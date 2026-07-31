use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};
use tokio::sync::{mpsc, oneshot};

use crate::{
    domain::{CoreError, SpeechRecognizer, Transcript, TranscriptionRequest},
    metrics::CoreMetrics,
};

enum RecognitionJob {
    Batch {
        request: TranscriptionRequest,
        response: oneshot::Sender<Result<Transcript, CoreError>>,
    },
    Begin {
        sample_rate: u32,
        response: oneshot::Sender<Result<(), CoreError>>,
    },
    Push {
        samples: Vec<f32>,
        response: oneshot::Sender<Result<(), CoreError>>,
    },
    Finish {
        response: oneshot::Sender<Result<Transcript, CoreError>>,
    },
}

/// Bounded streaming STT adapter backed by a supervised native worker.
pub struct SherpaOnnxRecognizer {
    jobs: Arc<RwLock<mpsc::Sender<RecognitionJob>>>,
    model_directory: PathBuf,
    threads: i32,
    queue_capacity: usize,
    metrics: CoreMetrics,
    restarts: AtomicU32,
    max_restarts: u32,
    restart_backoff: Duration,
    recovery: tokio::sync::Mutex<()>,
}

impl SherpaOnnxRecognizer {
    /// Starts one native worker, warms it up and creates a bounded request queue.
    pub fn new(
        model_directory: impl Into<PathBuf>,
        threads: i32,
        queue_capacity: usize,
        metrics: CoreMetrics,
    ) -> Result<Self, CoreError> {
        Self::new_supervised_with_control(
            model_directory,
            threads,
            queue_capacity,
            metrics,
            3,
            Duration::from_millis(500),
            Arc::new(AtomicBool::new(false)),
            |_, _| {},
        )
    }

    /// Loads with cooperative cancellation and the default restart policy.
    pub fn new_with_control(
        model_directory: impl Into<PathBuf>,
        threads: i32,
        queue_capacity: usize,
        metrics: CoreMetrics,
        cancelled: Arc<AtomicBool>,
        progress: impl FnMut(u8, &'static str) + Send + 'static,
    ) -> Result<Self, CoreError> {
        Self::new_supervised_with_control(
            model_directory,
            threads,
            queue_capacity,
            metrics,
            3,
            Duration::from_millis(500),
            cancelled,
            progress,
        )
    }

    /// Loads, warms and supervises a worker using an explicit bounded restart policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new_supervised_with_control(
        model_directory: impl Into<PathBuf>,
        threads: i32,
        queue_capacity: usize,
        metrics: CoreMetrics,
        max_restarts: u32,
        restart_backoff: Duration,
        cancelled: Arc<AtomicBool>,
        progress: impl FnMut(u8, &'static str) + Send + 'static,
    ) -> Result<Self, CoreError> {
        let model_directory = model_directory.into();
        validate_model(&model_directory, queue_capacity)?;
        let threads = effective_threads(threads);
        let jobs = spawn_worker(
            model_directory.clone(),
            threads,
            queue_capacity,
            metrics.clone(),
            cancelled,
            progress,
        )?;
        Ok(Self {
            jobs: Arc::new(RwLock::new(jobs)),
            model_directory,
            threads,
            queue_capacity,
            metrics,
            restarts: AtomicU32::new(0),
            max_restarts,
            restart_backoff,
            recovery: tokio::sync::Mutex::new(()),
        })
    }

    async fn sender(&self) -> Result<mpsc::Sender<RecognitionJob>, CoreError> {
        self.jobs
            .read()
            .map(|sender| sender.clone())
            .map_err(|_| CoreError::Recognition("STT worker lock poisoned".into()))
    }
}

#[async_trait]
impl SpeechRecognizer for SherpaOnnxRecognizer {
    async fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        validate_request(&request)?;
        let (response, result) = oneshot::channel();
        let sender = self.sender().await?;
        let permit = sender
            .reserve()
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?;
        self.metrics.stt_queued();
        permit.send(RecognitionJob::Batch { request, response });
        result
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?
    }

    async fn begin_stream(&self, sample_rate: u32) -> Result<bool, CoreError> {
        if sample_rate == 0 {
            return Err(CoreError::Recognition(
                "sample rate must be non-zero".into(),
            ));
        }
        let (response, result) = oneshot::channel();
        self.sender()
            .await?
            .send(RecognitionJob::Begin {
                sample_rate,
                response,
            })
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?;
        result
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))??;
        Ok(true)
    }

    async fn push_stream(&self, samples: Vec<f32>) -> Result<(), CoreError> {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(CoreError::Recognition(
                "stream chunk must contain finite samples".into(),
            ));
        }
        let (response, result) = oneshot::channel();
        self.sender()
            .await?
            .send(RecognitionJob::Push { samples, response })
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?;
        result
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?
    }

    async fn finish_stream(&self, request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        validate_request(&request)?;
        let (response, result) = oneshot::channel();
        let sender = self.sender().await?;
        let permit = sender
            .reserve()
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?;
        self.metrics.stt_queued();
        permit.send(RecognitionJob::Finish { response });
        result
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?
    }

    async fn recover(&self) -> Result<bool, CoreError> {
        let _guard = self.recovery.lock().await;
        let restart = self.restarts.fetch_add(1, Ordering::AcqRel) + 1;
        if restart > self.max_restarts {
            self.metrics.stt_faulted();
            return Err(CoreError::Unavailable(
                "STT restart budget exhausted".into(),
            ));
        }
        // ponytail: Rust cannot terminate a stuck native thread; each abandoned generation is
        // bounded by `max_restarts`, and a process restart is the upgrade path for hard hangs.
        tokio::time::sleep(restart_delay(self.restart_backoff, restart)).await;
        let model_directory = self.model_directory.clone();
        let threads = self.threads;
        let queue_capacity = self.queue_capacity;
        let metrics = self.metrics.clone();
        let sender = tokio::task::spawn_blocking(move || {
            spawn_worker(
                model_directory,
                threads,
                queue_capacity,
                metrics,
                Arc::new(AtomicBool::new(false)),
                |_, _| {},
            )
        })
        .await
        .map_err(|error| CoreError::Recognition(error.to_string()))??;
        *self
            .jobs
            .write()
            .map_err(|_| CoreError::Recognition("STT worker lock poisoned".into()))? = sender;
        self.metrics.reset_stt_queue();
        self.metrics.stt_restarted();
        Ok(true)
    }
}

fn spawn_worker(
    model_directory: PathBuf,
    threads: i32,
    queue_capacity: usize,
    metrics: CoreMetrics,
    cancelled: Arc<AtomicBool>,
    mut progress: impl FnMut(u8, &'static str) + Send + 'static,
) -> Result<mpsc::Sender<RecognitionJob>, CoreError> {
    let (jobs, mut receiver) = mpsc::channel::<RecognitionJob>(queue_capacity);
    let (ready, initialized) = std::sync::mpsc::sync_channel(1);
    std::thread::Builder::new()
        .name("assistant-stt".into())
        .spawn(move || {
            progress(20, "validating model");
            if cancelled.load(Ordering::Acquire) {
                let _ = ready.send(Err("STT loading cancelled".into()));
                return;
            }
            progress(50, "initializing STT engine");
            let recognizer = create_recognizer(&model_directory, threads);
            match recognizer {
                Ok(recognizer) => {
                    progress(75, "warming STT engine");
                    warm_up(&recognizer);
                    if cancelled.load(Ordering::Acquire) {
                        let _ = ready.send(Err("STT loading cancelled".into()));
                        return;
                    }
                    progress(100, "STT engine ready");
                    if ready.send(Ok(())).is_err() {
                        return;
                    }
                    worker_loop(recognizer, &mut receiver, metrics);
                }
                Err(message) => {
                    let _ = ready.send(Err(message));
                }
            }
        })
        .map_err(|error| CoreError::Recognition(error.to_string()))?;
    initialized
        .recv()
        .map_err(|_| CoreError::Recognition("STT loading worker stopped".into()))?
        .map_err(CoreError::Recognition)?;
    Ok(jobs)
}

fn worker_loop(
    recognizer: OnlineRecognizer,
    receiver: &mut mpsc::Receiver<RecognitionJob>,
    metrics: CoreMetrics,
) {
    let mut streaming: Option<(OnlineStream, i32)> = None;
    while let Some(job) = receiver.blocking_recv() {
        match job {
            RecognitionJob::Batch { request, response } => {
                metrics.stt_dequeued();
                let _ = response.send(recognize(&recognizer, request));
            }
            RecognitionJob::Begin {
                sample_rate,
                response,
            } => {
                let result = i32::try_from(sample_rate)
                    .map(|sample_rate| streaming = Some((recognizer.create_stream(), sample_rate)))
                    .map_err(|_| CoreError::Recognition("sample rate is too large".into()));
                let _ = response.send(result);
            }
            RecognitionJob::Push { samples, response } => {
                let result = streaming
                    .as_ref()
                    .ok_or_else(|| CoreError::Recognition("no STT stream is active".into()))
                    .map(|(stream, sample_rate)| {
                        stream.accept_waveform(*sample_rate, &samples);
                        decode_ready(&recognizer, stream);
                    });
                let _ = response.send(result);
            }
            RecognitionJob::Finish { response } => {
                metrics.stt_dequeued();
                let result = streaming
                    .take()
                    .ok_or_else(|| CoreError::Recognition("no STT stream is active".into()))
                    .and_then(|(stream, _)| finish(&recognizer, stream));
                let _ = response.send(result);
            }
        }
    }
}

fn validate_model(directory: &Path, queue_capacity: usize) -> Result<(), CoreError> {
    for path in [
        "am-onnx/encoder.int8.onnx",
        "am-onnx/decoder.int8.onnx",
        "am-onnx/joiner.int8.onnx",
        "lang/tokens.txt",
    ] {
        if !directory.join(path).is_file() {
            return Err(CoreError::Recognition(format!(
                "model file is missing: {path}"
            )));
        }
    }
    if queue_capacity == 0 {
        return Err(CoreError::Recognition(
            "STT queue capacity must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn validate_request(request: &TranscriptionRequest) -> Result<(), CoreError> {
    if request.sample_rate == 0
        || request.samples.is_empty()
        || request.samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(CoreError::Recognition(
            "audio must contain finite samples at a non-zero sample rate".into(),
        ));
    }
    Ok(())
}

/// Resolves zero to host parallelism while retaining an explicit calibration knob.
pub fn effective_threads(configured: i32) -> i32 {
    if configured > 0 {
        return configured;
    }
    std::thread::available_parallelism()
        .map(|threads| threads.get().min(64) as i32)
        .unwrap_or(1)
}

fn restart_delay(initial: Duration, attempt: u32) -> Duration {
    initial.saturating_mul(1_u32 << attempt.saturating_sub(1).min(10))
}

fn create_recognizer(directory: &Path, threads: i32) -> Result<OnlineRecognizer, String> {
    let path = |relative: &str| directory.join(relative).to_string_lossy().into_owned();
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(path("am-onnx/encoder.int8.onnx"));
    config.model_config.transducer.decoder = Some(path("am-onnx/decoder.int8.onnx"));
    config.model_config.transducer.joiner = Some(path("am-onnx/joiner.int8.onnx"));
    config.model_config.tokens = Some(path("lang/tokens.txt"));
    config.model_config.num_threads = threads.max(1);
    config.enable_endpoint = true;
    config.decoding_method = Some("greedy_search".into());
    OnlineRecognizer::create(&config).ok_or_else(|| "sherpa-onnx initialization failed".into())
}

fn warm_up(recognizer: &OnlineRecognizer) {
    let stream = recognizer.create_stream();
    stream.accept_waveform(16_000, &[0.0; 1_600]);
    stream.input_finished();
    decode_ready(recognizer, &stream);
}

fn recognize(
    recognizer: &OnlineRecognizer,
    request: TranscriptionRequest,
) -> Result<Transcript, CoreError> {
    let stream = recognizer.create_stream();
    let sample_rate = i32::try_from(request.sample_rate)
        .map_err(|_| CoreError::Recognition("sample rate is too large".into()))?;
    for chunk in request
        .samples
        .chunks((request.sample_rate / 50).max(1) as usize)
    {
        stream.accept_waveform(sample_rate, chunk);
        decode_ready(recognizer, &stream);
    }
    stream.input_finished();
    finish(recognizer, stream)
}

fn decode_ready(recognizer: &OnlineRecognizer, stream: &OnlineStream) {
    while recognizer.is_ready(stream) {
        recognizer.decode(stream);
    }
}

fn finish(recognizer: &OnlineRecognizer, stream: OnlineStream) -> Result<Transcript, CoreError> {
    stream.input_finished();
    decode_ready(recognizer, &stream);
    let result = recognizer
        .get_result(&stream)
        .ok_or_else(|| CoreError::Recognition("sherpa-onnx returned no result".into()))?;
    Ok(Transcript {
        text: result.text,
        confidence: None,
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{effective_threads, restart_delay};

    #[test]
    fn automatic_thread_count_is_bounded() {
        assert!((1..=64).contains(&effective_threads(0)));
        assert_eq!(effective_threads(3), 3);
    }

    #[test]
    fn worker_restart_backoff_is_bounded() {
        assert_eq!(
            restart_delay(Duration::from_millis(500), 1),
            Duration::from_millis(500)
        );
        assert_eq!(
            restart_delay(Duration::from_millis(500), 3),
            Duration::from_secs(2)
        );
        assert_eq!(
            restart_delay(Duration::from_millis(500), 50),
            Duration::from_secs(512)
        );
    }
}
