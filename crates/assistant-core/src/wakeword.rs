use std::path::{Path, PathBuf};

use sentencepiece_rs::SentencePieceProcessor;
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, OnlineStream};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
/// One accepted wake-word occurrence.
pub struct WakeWordDetection {
    /// Detector keyword text.
    pub keyword: String,
    /// Approximate start time in seconds within detector input.
    pub start_time: f32,
}

#[derive(Debug, Error)]
/// Wake-word model or keyword configuration failure.
pub enum WakeWordError {
    /// Local model files or engine parameters are invalid.
    #[error("wake-word configuration failed: {0}")]
    Configuration(String),
}

/// Streaming sherpa-onnx open-vocabulary wake-word adapter.
pub struct SherpaWakeWordDetector {
    spotter: KeywordSpotter,
    stream: OnlineStream,
    sample_rate: i32,
}

impl SherpaWakeWordDetector {
    /// Loads a verified model directory and encodes one keyword.
    pub fn new(
        model_directory: impl Into<PathBuf>,
        keyword: &str,
        score: f32,
        threshold: f32,
        threads: i32,
        sample_rate: u32,
    ) -> Result<Self, WakeWordError> {
        let directory = model_directory.into();
        let keywords = tokenize_keyword(&directory, keyword, score, threshold)?;
        let path = |relative: &str| directory.join(relative).to_string_lossy().into_owned();
        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder = Some(path("am-onnx/encoder.int8.onnx"));
        config.model_config.transducer.decoder = Some(path("am-onnx/decoder.int8.onnx"));
        config.model_config.transducer.joiner = Some(path("am-onnx/joiner.int8.onnx"));
        config.model_config.tokens = Some(path("lang/tokens.txt"));
        config.model_config.num_threads = threads.max(1);
        config.keywords_buf = Some(keywords);
        config.keywords_score = score;
        config.keywords_threshold = threshold;
        let spotter = KeywordSpotter::create(&config)
            .ok_or_else(|| WakeWordError::Configuration("sherpa-onnx rejected model".into()))?;
        let stream = spotter.create_stream();
        let sample_rate = i32::try_from(sample_rate)
            .map_err(|_| WakeWordError::Configuration("sample rate is too large".into()))?;
        Ok(Self {
            spotter,
            stream,
            sample_rate,
        })
    }

    /// Clears decoder state after capture, cooldown or suspension.
    pub fn reset(&mut self) {
        self.spotter.reset(&self.stream);
    }

    /// Processes one mono frame and returns an accepted detection.
    pub fn process(&mut self, frame: &[f32]) -> Option<WakeWordDetection> {
        self.stream.accept_waveform(self.sample_rate, frame);
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
        }
        let result = self.spotter.get_result(&self.stream)?;
        if result.keyword.is_empty() {
            return None;
        }
        let detection = WakeWordDetection {
            keyword: result.keyword,
            start_time: result.start_time,
        };
        self.reset();
        Some(detection)
    }
}

fn tokenize_keyword(
    directory: &Path,
    keyword: &str,
    score: f32,
    threshold: f32,
) -> Result<String, WakeWordError> {
    let tokenizer = SentencePieceProcessor::open(directory.join("lang/bpe.model"))
        .map_err(|error| WakeWordError::Configuration(error.to_string()))?;
    let pieces = tokenizer
        .encode(keyword)
        .map_err(|error| WakeWordError::Configuration(error.to_string()))?;
    if pieces.is_empty() {
        return Err(WakeWordError::Configuration(
            "keyword produced no tokens".into(),
        ));
    }
    Ok(format!(
        "{} :{score} #{threshold} @{keyword}",
        pieces.join(" ")
    ))
}
