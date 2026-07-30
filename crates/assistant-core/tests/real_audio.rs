#![cfg(feature = "stt-sherpa-onnx")]

use std::{env, path::PathBuf};

use assistant_core::{
    CoreMetrics, SpeechRecognizer, TranscriptionRequest, audio::read_wav_mono,
    stt::SherpaOnnxRecognizer, wakeword::SherpaWakeWordDetector,
};

#[tokio::test]
#[ignore = "requires a fixed external Russian WAV and installed model"]
async fn fixed_real_wav_detects_wake_word_and_expected_text() {
    let wav = PathBuf::from(env::var_os("VOICE_ASSISTANT_TEST_WAV").expect("test WAV path"));
    let model = PathBuf::from(env::var_os("VOICE_ASSISTANT_TEST_MODEL").expect("model path"));
    let expected = env::var("VOICE_ASSISTANT_TEST_TRANSCRIPT").expect("expected transcript");
    let (samples, sample_rate) = read_wav_mono(wav).expect("valid WAV");

    let mut detector = SherpaWakeWordDetector::new(&model, "ассистент", 1.5, 0.25, 1, sample_rate)
        .expect("wake-word detector");
    assert!(
        samples
            .chunks((sample_rate / 50) as usize)
            .any(|frame| detector.process(frame).is_some())
    );

    let transcript = SherpaOnnxRecognizer::new(model, 2, 1, CoreMetrics::default())
        .expect("recognizer")
        .transcribe(TranscriptionRequest {
            samples,
            sample_rate,
        })
        .await
        .expect("transcription");
    assert_eq!(transcript.text, expected);
}
