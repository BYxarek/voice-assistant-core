use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Binary voice-activity classification for one mono frame.
pub enum VadDecision {
    /// Frame contains speech-like energy.
    Speech,
    /// Frame is treated as silence.
    Silence,
}

/// Replaceable synchronous VAD adapter used by the command collector.
pub trait VoiceActivityDetector: Send {
    /// Clears detector history between utterances.
    fn reset(&mut self);
    /// Classifies one mono frame.
    fn process(&mut self, frame: &[f32]) -> VadDecision;
}

/// Stateless RMS-threshold voice activity detector.
pub struct EnergyVad {
    threshold: f32,
}

impl EnergyVad {
    /// Creates a detector using normalized RMS threshold.
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold: threshold.max(0.0),
        }
    }
}

impl VoiceActivityDetector for EnergyVad {
    fn reset(&mut self) {}

    fn process(&mut self, frame: &[f32]) -> VadDecision {
        if frame.is_empty() {
            return VadDecision::Silence;
        }
        let rms =
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame.len() as f32).sqrt();
        if rms >= self.threshold {
            VadDecision::Speech
        } else {
            VadDecision::Silence
        }
    }
}

/// Fixed-capacity rolling mono sample buffer.
pub struct AudioRingBuffer {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl AudioRingBuffer {
    /// Creates a ring retaining at most `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Appends samples and evicts the oldest overflow.
    pub fn push(&mut self, input: &[f32]) {
        if self.capacity == 0 {
            return;
        }
        for &sample in input {
            if self.samples.len() == self.capacity {
                self.samples.pop_front();
            }
            self.samples.push_back(sample);
        }
    }

    /// Copies retained samples in chronological order.
    pub fn snapshot(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }
}

/// Collects one utterance until silence or the hard sample limit.
pub struct CommandCollector {
    samples: Vec<f32>,
    min_samples: usize,
    max_samples: usize,
    trailing_silence_samples: usize,
    silence_samples: usize,
}

/// Pure frame collector joining pre-roll, VAD and command-length policy.
pub struct CommandAudioPipeline<V> {
    ring: AudioRingBuffer,
    collector: Option<CommandCollector>,
    vad: V,
    pre_roll_samples: usize,
    min_samples: usize,
    max_samples: usize,
    trailing_silence_samples: usize,
}

impl<V: VoiceActivityDetector> CommandAudioPipeline<V> {
    /// Creates a bounded command collector from sample-count limits.
    pub fn new(
        vad: V,
        pre_roll_samples: usize,
        min_samples: usize,
        max_samples: usize,
        trailing_silence_samples: usize,
    ) -> Self {
        Self {
            ring: AudioRingBuffer::new(pre_roll_samples),
            collector: None,
            vad,
            pre_roll_samples,
            min_samples,
            max_samples,
            trailing_silence_samples,
        }
    }

    /// Reports whether frames are currently being appended to a command.
    pub fn is_collecting(&self) -> bool {
        self.collector.is_some()
    }

    /// Starts application-requested capture without wake-word pre-roll.
    pub fn start_manual(&mut self) {
        if self.collector.is_none() {
            self.ring = AudioRingBuffer::new(self.pre_roll_samples);
            self.vad.reset();
            self.collector = Some(CommandCollector::start(
                Vec::new(),
                self.min_samples,
                self.max_samples,
                self.trailing_silence_samples,
            ));
        }
    }

    /// Processes one frame and returns a completed command when policy stops capture.
    pub fn push(&mut self, frame: &[f32], wake_word_detected: bool) -> Option<Vec<f32>> {
        if let Some(collector) = self.collector.as_mut() {
            let result = collector.push(frame, self.vad.process(frame));
            if result.is_some() {
                self.reset();
            }
            return result;
        }
        self.ring.push(frame);
        if wake_word_detected {
            self.collector = Some(CommandCollector::start(
                self.ring.snapshot(),
                self.min_samples,
                self.max_samples,
                self.trailing_silence_samples,
            ));
        }
        None
    }

    /// Finishes an active manual capture, discarding audio shorter than the minimum.
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        let result = self.collector.take().and_then(CommandCollector::finish);
        self.reset();
        result
    }

    /// Clears pre-roll, active capture and detector state.
    pub fn reset(&mut self) {
        self.collector = None;
        self.ring = AudioRingBuffer::new(self.pre_roll_samples);
        self.vad.reset();
    }
}

impl CommandCollector {
    /// Starts collection with pre-roll and bounded duration thresholds.
    pub fn start(
        pre_roll: Vec<f32>,
        min_samples: usize,
        max_samples: usize,
        trailing_silence_samples: usize,
    ) -> Self {
        let samples = if pre_roll.len() > max_samples {
            pre_roll[pre_roll.len() - max_samples..].to_vec()
        } else {
            pre_roll
        };
        Self {
            samples,
            min_samples,
            max_samples,
            trailing_silence_samples,
            silence_samples: 0,
        }
    }

    /// Appends one classified frame and returns completed audio when finished.
    pub fn push(&mut self, frame: &[f32], decision: VadDecision) -> Option<Vec<f32>> {
        let remaining = self.max_samples.saturating_sub(self.samples.len());
        self.samples
            .extend_from_slice(&frame[..frame.len().min(remaining)]);
        self.silence_samples = match decision {
            VadDecision::Speech => 0,
            VadDecision::Silence => self.silence_samples.saturating_add(frame.len()),
        };
        let complete = self.samples.len() >= self.max_samples
            || (self.samples.len() >= self.min_samples
                && self.silence_samples >= self.trailing_silence_samples);
        complete.then(|| std::mem::take(&mut self.samples))
    }

    fn finish(mut self) -> Option<Vec<f32>> {
        (self.samples.len() >= self.min_samples).then(|| std::mem::take(&mut self.samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_only_pre_roll() {
        let mut ring = AudioRingBuffer::new(3);
        ring.push(&[1.0, 2.0]);
        ring.push(&[3.0, 4.0]);
        assert_eq!(ring.snapshot(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn collector_stops_after_silence() {
        let mut collector = CommandCollector::start(vec![1.0], 3, 10, 2);
        assert!(collector.push(&[1.0, 1.0], VadDecision::Speech).is_none());
        assert_eq!(
            collector.push(&[0.0, 0.0], VadDecision::Silence),
            Some(vec![1.0, 1.0, 1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn zero_capacity_ring_stays_empty() {
        let mut ring = AudioRingBuffer::new(0);
        ring.push(&[1.0]);
        assert!(ring.snapshot().is_empty());
    }

    #[test]
    fn frame_pipeline_collects_after_fake_wake_word() {
        let mut pipeline = CommandAudioPipeline::new(EnergyVad::new(0.5), 2, 3, 10, 2);
        assert!(pipeline.push(&[1.0, 1.0], true).is_none());
        assert!(pipeline.push(&[1.0], false).is_none());
        assert_eq!(
            pipeline.push(&[0.0, 0.0], false),
            Some(vec![1.0, 1.0, 1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn manual_finish_keeps_only_a_long_enough_capture() {
        let mut pipeline = CommandAudioPipeline::new(EnergyVad::new(0.5), 0, 3, 10, 2);
        pipeline.start_manual();
        pipeline.push(&[1.0], false);
        assert!(pipeline.finish().is_none());

        pipeline.start_manual();
        pipeline.push(&[1.0, 1.0, 1.0], false);
        assert_eq!(pipeline.finish(), Some(vec![1.0, 1.0, 1.0]));
        assert!(!pipeline.is_collecting());
    }
}
