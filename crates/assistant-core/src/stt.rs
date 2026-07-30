use std::path::{Path, PathBuf};

use async_trait::async_trait;
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};
use tokio::sync::{mpsc, oneshot};

use crate::{
    domain::{CoreError, SpeechRecognizer, Transcript, TranscriptionRequest},
    metrics::CoreMetrics,
};

struct RecognitionJob {
    request: TranscriptionRequest,
    response: oneshot::Sender<Result<Transcript, CoreError>>,
}

/// Bounded STT adapter backed by one long-lived native inference worker.
pub struct SherpaOnnxRecognizer {
    jobs: mpsc::Sender<RecognitionJob>,
    metrics: CoreMetrics,
}

impl SherpaOnnxRecognizer {
    /// Starts one native worker and a bounded request queue.
    pub fn new(
        model_directory: impl Into<PathBuf>,
        threads: i32,
        queue_capacity: usize,
        metrics: CoreMetrics,
    ) -> Result<Self, CoreError> {
        let model_directory = model_directory.into();
        for path in [
            "am-onnx/encoder.int8.onnx",
            "am-onnx/decoder.int8.onnx",
            "am-onnx/joiner.int8.onnx",
            "lang/tokens.txt",
        ] {
            if !model_directory.join(path).is_file() {
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

        let (jobs, mut receiver) = mpsc::channel::<RecognitionJob>(queue_capacity);
        let worker_metrics = metrics.clone();
        std::thread::Builder::new()
            .name("assistant-stt".into())
            .spawn(move || {
                let recognizer = create_recognizer(&model_directory, threads.max(1));
                while let Some(job) = receiver.blocking_recv() {
                    worker_metrics.stt_dequeued();
                    let result = match &recognizer {
                        Ok(recognizer) => recognize(recognizer, job.request),
                        Err(message) => Err(CoreError::Recognition(message.clone())),
                    };
                    let _ = job.response.send(result);
                }
            })
            .map_err(|error| CoreError::Recognition(error.to_string()))?;
        Ok(Self { jobs, metrics })
    }
}

#[async_trait]
impl SpeechRecognizer for SherpaOnnxRecognizer {
    async fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        if request.sample_rate == 0
            || request.samples.is_empty()
            || request.samples.iter().any(|sample| !sample.is_finite())
        {
            return Err(CoreError::Recognition(
                "audio must contain finite samples at a non-zero sample rate".into(),
            ));
        }
        let (response, result) = oneshot::channel();
        let permit = self
            .jobs
            .reserve()
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?;
        self.metrics.stt_queued();
        permit.send(RecognitionJob { request, response });
        result
            .await
            .map_err(|_| CoreError::Recognition("STT worker stopped".into()))?
    }
}

fn create_recognizer(directory: &Path, threads: i32) -> Result<OnlineRecognizer, String> {
    let path = |relative: &str| directory.join(relative).to_string_lossy().into_owned();
    let mut config = OnlineRecognizerConfig::default();
    config.model_config.transducer.encoder = Some(path("am-onnx/encoder.int8.onnx"));
    config.model_config.transducer.decoder = Some(path("am-onnx/decoder.int8.onnx"));
    config.model_config.transducer.joiner = Some(path("am-onnx/joiner.int8.onnx"));
    config.model_config.tokens = Some(path("lang/tokens.txt"));
    config.model_config.num_threads = threads;
    config.enable_endpoint = true;
    config.decoding_method = Some("greedy_search".into());
    OnlineRecognizer::create(&config).ok_or_else(|| "sherpa-onnx initialization failed".into())
}

fn recognize(
    recognizer: &OnlineRecognizer,
    request: TranscriptionRequest,
) -> Result<Transcript, CoreError> {
    let stream = recognizer.create_stream();
    let sample_rate = i32::try_from(request.sample_rate)
        .map_err(|_| CoreError::Recognition("sample rate is too large".into()))?;
    stream.accept_waveform(sample_rate, &request.samples);
    stream.input_finished();
    while recognizer.is_ready(&stream) {
        recognizer.decode(&stream);
    }
    let result = recognizer
        .get_result(&stream)
        .ok_or_else(|| CoreError::Recognition("sherpa-onnx returned no result".into()))?;
    Ok(Transcript {
        text: result.text,
        confidence: None,
    })
}
