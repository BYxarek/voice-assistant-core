use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::metrics::CoreMetrics;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// User-visible input-device descriptor.
pub struct AudioDeviceInfo {
    /// Stable backend identifier accepted by [`AudioInput::open`].
    pub id: String,
    /// Display name reported by the operating system.
    pub name: String,
    /// Whether this is the current default input device.
    pub is_default: bool,
}

#[derive(Debug, Error)]
/// Audio device, stream and WAV failures.
pub enum AudioError {
    /// Device enumeration or configuration failed.
    #[error("audio device error: {0}")]
    Device(String),
    /// Stream creation or capture failed.
    #[error("audio stream error: {0}")]
    Stream(String),
    /// The bounded stream channel disconnected.
    #[error("audio input disconnected")]
    Disconnected,
    /// WAV decoding or encoding failed.
    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),
}

/// Live input stream whose callback only downmixes and performs non-blocking enqueue.
pub struct AudioInput {
    receiver: mpsc::Receiver<Vec<f32>>,
    stream: Option<cpal::Stream>,
    source_rate: u32,
    failed: Arc<AtomicBool>,
    metrics: CoreMetrics,
}

impl AudioInput {
    /// Opens an input device and starts a bounded non-blocking callback.
    pub fn open(
        device_id: &str,
        queue_capacity: usize,
        metrics: CoreMetrics,
    ) -> Result<Self, AudioError> {
        let device = select_input_device(device_id)?;
        let config = device
            .default_input_config()
            .map_err(|error| AudioError::Device(error.to_string()))?;
        let source_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let (sender, receiver) = mpsc::sync_channel::<Vec<f32>>(queue_capacity);
        let failed = Arc::new(AtomicBool::new(false));
        let callback_failed = Arc::clone(&failed);
        let queue_metrics = metrics.clone();

        // ponytail: callback allocates one block; use a preallocated lock-free pool if profiling shows drops.
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    enqueue(
                        &sender,
                        &queue_metrics,
                        downmix(data.iter().copied(), channels),
                    );
                },
                move |error| {
                    callback_failed.store(true, Ordering::Release);
                    tracing::error!(%error, "audio input stream failed");
                },
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    enqueue(
                        &sender,
                        &queue_metrics,
                        downmix(
                            data.iter().map(|&sample| sample as f32 / i16::MAX as f32),
                            channels,
                        ),
                    );
                },
                move |error| {
                    callback_failed.store(true, Ordering::Release);
                    tracing::error!(%error, "audio input stream failed");
                },
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
                    enqueue(
                        &sender,
                        &queue_metrics,
                        downmix(
                            data.iter()
                                .map(|&sample| sample as f32 / u16::MAX as f32 * 2.0 - 1.0),
                            channels,
                        ),
                    );
                },
                move |error| {
                    callback_failed.store(true, Ordering::Release);
                    tracing::error!(%error, "audio input stream failed");
                },
                None,
            ),
            format => {
                return Err(AudioError::Stream(format!(
                    "unsupported sample format: {format:?}"
                )));
            }
        }
        .map_err(|error| AudioError::Stream(error.to_string()))?;
        stream
            .play()
            .map_err(|error| AudioError::Stream(error.to_string()))?;
        Ok(Self {
            receiver,
            stream: Some(stream),
            source_rate,
            failed,
            metrics,
        })
    }

    /// Native device sample rate before worker-side resampling.
    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Waits for one downmixed block without running work in the callback.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<Vec<f32>>, AudioError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(AudioError::Disconnected);
        }
        match self.receiver.recv_timeout(timeout) {
            Ok(block) => {
                self.metrics.audio_dequeued();
                Ok(Some(block))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(AudioError::Disconnected),
        }
    }
}

impl Drop for AudioInput {
    fn drop(&mut self) {
        self.stream.take();
        self.metrics.reset_audio_queue();
    }
}

/// Lists input devices and marks the current default.
pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, AudioError> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    host.input_devices()
        .map_err(|e| AudioError::Device(e.to_string()))?
        .map(|device| {
            let name = device
                .name()
                .map_err(|e| AudioError::Device(e.to_string()))?;
            Ok(AudioDeviceInfo {
                id: name.clone(),
                is_default: default_name.as_deref() == Some(&name),
                name,
            })
        })
        .collect()
}

/// Records one input device to a mono floating-point WAV at the target rate.
pub fn record_wav(
    device_id: &str,
    duration: Duration,
    target_sample_rate: u32,
    path: impl AsRef<Path>,
) -> Result<(), AudioError> {
    let input = AudioInput::open(device_id, 32, CoreMetrics::default())?;
    let source_rate = input.source_rate();
    let deadline = Instant::now() + duration;
    let mut samples = Vec::new();
    while Instant::now() < deadline {
        if let Some(block) = input.recv_timeout(Duration::from_millis(100))? {
            samples.extend(block);
        }
    }
    drop(input);

    let samples = resample_linear(&samples, source_rate, target_sample_rate);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: target_sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn select_input_device(device_id: &str) -> Result<cpal::Device, AudioError> {
    let host = cpal::default_host();
    if device_id == "default" {
        return host
            .default_input_device()
            .ok_or_else(|| AudioError::Device("no default input device".into()));
    }
    host.input_devices()
        .map_err(|error| AudioError::Device(error.to_string()))?
        .find(|device| device.name().is_ok_and(|name| name == device_id))
        .ok_or_else(|| AudioError::Device(format!("input device not found: {device_id}")))
}

fn enqueue(sender: &mpsc::SyncSender<Vec<f32>>, metrics: &CoreMetrics, block: Vec<f32>) {
    metrics.audio_queued();
    match sender.try_send(block) {
        Ok(()) => {}
        Err(mpsc::TrySendError::Full(_)) => {
            metrics.audio_dequeued();
            metrics.audio_dropped();
        }
        Err(mpsc::TrySendError::Disconnected(_)) => metrics.audio_dequeued(),
    }
}

/// Reads a WAV, converts supported samples to `f32` and downmixes to mono.
pub fn read_wav_mono(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), AudioError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(AudioError::Wav(hound::Error::FormatError(
            "only mono WAV is supported",
        )));
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| sample.map(|value| value as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
    };
    Ok((samples, spec.sample_rate))
}

fn downmix(samples: impl Iterator<Item = f32>, channels: usize) -> Vec<f32> {
    let samples: Vec<_> = samples.collect();
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resamples mono samples using bounded linear interpolation.
pub fn resample_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == target_rate {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64 * target_rate as u64 / source_rate as u64) as usize;
    (0..output_len)
        .map(|i| {
            let position = i as f64 * source_rate as f64 / target_rate as f64;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (position - left as f64) as f32;
            samples[left] * (1.0 - fraction) + samples[right] * fraction
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    #[test]
    fn resampling_preserves_duration() {
        for source_rate in [8_000, 44_100, 48_000, 96_000] {
            let input: Vec<_> = (0..source_rate)
                .map(|index| (index as f32 / source_rate as f32).sin())
                .collect();
            let output = resample_linear(&input, source_rate, 16_000);
            assert_eq!(output.len(), 16_000);
            assert!(output.iter().all(|sample| sample.is_finite()));
        }
    }
}
