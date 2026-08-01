use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Context;
use assistant_core::{
    AppPaths, CommandExecutor, CommandRegistry, CoreConfig, CoreIpcClient, CoreMetrics,
    CoreRequest, CoreResponse, MockRecognizer, Runtime, TranscriptionRequest,
    audio::{AudioInput, list_input_devices, record_wav},
    builtin_handlers,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "Voice Assistant Core diagnostics")]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    models: Option<PathBuf>,
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
enum CliCommand {
    Devices,
    Record {
        #[arg(long, default_value = "default")]
        device: String,
        #[arg(long, default_value_t = 5)]
        seconds: u64,
        #[arg(long, default_value = "recording.wav")]
        output: PathBuf,
    },
    ValidateConfig,
    Status,
    Handlers,
    Soak {
        #[arg(long, default_value_t = 60)]
        seconds: u64,
    },
    RunText {
        text: String,
    },
    Models {
        #[command(subcommand)]
        command: ModelCommand,
    },
    #[cfg(feature = "stt-sherpa-onnx")]
    Transcribe {
        input: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        threads: i32,
    },
    #[cfg(feature = "stt-sherpa-onnx")]
    TestWakeWord {
        input: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
    },
    #[cfg(feature = "stt-sherpa-onnx")]
    Evaluate {
        input: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    List,
    Install { model: Option<String> },
    Verify { directory: Option<PathBuf> },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("assistant_core=info")
        .init();
    let args = Args::parse();
    let paths = AppPaths::discover()?;
    paths.ensure_directories()?;
    let config_path = args.config.unwrap_or(paths.config);
    let models_path = args.models.unwrap_or(paths.models);
    if !config_path.is_file() {
        CoreConfig::bundled_example()?.save_atomic(&config_path)?;
    }
    match args.command {
        CliCommand::Devices => {
            for device in list_input_devices()? {
                println!(
                    "{}{}",
                    device.name,
                    if device.is_default { " (default)" } else { "" }
                );
            }
        }
        CliCommand::Record {
            device,
            seconds,
            output,
        } => {
            let config = CoreConfig::load(&config_path)?;
            record_wav(
                &device,
                Duration::from_secs(seconds),
                config.audio.target_sample_rate,
                &output,
            )?;
            println!("{}", output.display());
        }
        CliCommand::ValidateConfig => {
            let config = CoreConfig::load(&config_path)?;
            config.validate_with_handlers(&builtin_handlers(&config.commands)?)?;
            assistant_core::models::model_spec(&config.inference.model)?;
            let wake_model = assistant_core::models::model_spec(&config.wake_word.model)?;
            anyhow::ensure!(
                wake_model.supports_wake_word(),
                "model {} cannot be used for wake-word detection",
                wake_model.id()
            );
            println!("configuration is valid");
        }
        CliCommand::Status => {
            let config = CoreConfig::load(&config_path)?;
            let client = CoreIpcClient::new(
                config.ipc.pipe_name,
                Duration::from_millis(config.ipc.io_timeout_ms),
            )?;
            match client.request(CoreRequest::GetHealth).await? {
                CoreResponse::Health { health } => println!("{health:?}"),
                response => anyhow::bail!("unexpected response: {response:?}"),
            }
        }
        CliCommand::Handlers => {
            let config = CoreConfig::load(&config_path)?;
            let client = CoreIpcClient::new(
                config.ipc.pipe_name,
                Duration::from_millis(config.ipc.io_timeout_ms),
            )?;
            match client.request(CoreRequest::ListHandlers).await? {
                CoreResponse::Handlers { handlers } => {
                    for handler in handlers {
                        println!(
                            "{} required={:?} optional={:?}",
                            handler.name, handler.required_parameters, handler.optional_parameters
                        );
                    }
                }
                response => anyhow::bail!("unexpected response: {response:?}"),
            }
        }
        CliCommand::Soak { seconds } => {
            let config = CoreConfig::load(&config_path)?;
            let metrics = CoreMetrics::default();
            let deadline = std::time::Instant::now() + Duration::from_secs(seconds);
            while std::time::Instant::now() < deadline {
                match AudioInput::open(
                    &config.audio.device_id,
                    config.audio.queue_capacity_frames,
                    metrics.clone(),
                ) {
                    Ok(input) => {
                        let mut last_audio = std::time::Instant::now();
                        while std::time::Instant::now() < deadline {
                            match input.recv_timeout(Duration::from_millis(500)) {
                                Ok(Some(_)) => last_audio = std::time::Instant::now(),
                                Ok(None)
                                    if last_audio.elapsed()
                                        >= Duration::from_millis(config.audio.stall_timeout_ms) =>
                                {
                                    metrics.audio_reconnected();
                                    break;
                                }
                                Ok(None) => {}
                                Err(_) => {
                                    metrics.audio_reconnected();
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        metrics.audio_reconnected();
                        std::thread::sleep(Duration::from_millis(config.audio.reconnect_delay_ms));
                    }
                }
            }
            println!("{:?}", metrics.snapshot());
        }
        CliCommand::RunText { text } => {
            let config = CoreConfig::load(&config_path)?;
            let registry =
                CommandRegistry::new(config.commands.clone(), config.matching.normalize_yo);
            let executor: Arc<dyn CommandExecutor> = Arc::new(builtin_handlers(&config.commands)?);
            let mut runtime = Runtime::new_with_wake_words(
                Arc::new(MockRecognizer { text }),
                registry,
                executor,
                std::iter::once(config.wake_word.keyword.clone())
                    .chain(config.wake_word.aliases.iter().cloned())
                    .collect(),
                config.policy.clone(),
                Duration::from_millis(config.inference.timeout_ms),
                Duration::from_millis(config.wake_word.cooldown_ms),
                assistant_core::CoreMetrics::default(),
            );
            let mut events = runtime.subscribe();
            runtime.start()?;
            runtime
                .process_command_audio(TranscriptionRequest {
                    samples: vec![],
                    sample_rate: config.audio.target_sample_rate,
                })
                .await
                .context("pipeline failed")?;
            while let Ok(event) = events.try_recv() {
                println!("{event:?}");
            }
        }
        CliCommand::Models { command } => {
            use assistant_core::models::{ALPHACEP_MODELS, ModelManager};
            let manager = ModelManager::new(&models_path);
            match command {
                ModelCommand::List => {
                    for model in ALPHACEP_MODELS {
                        println!(
                            "{} language={} mode={:?} wake_word={}",
                            model.id(),
                            model.language(),
                            model.mode(),
                            model.supports_wake_word()
                        );
                    }
                }
                ModelCommand::Install { model } => {
                    let model = model.unwrap_or(CoreConfig::load(&config_path)?.inference.model);
                    println!("{}", manager.install(&model)?.display());
                }
                ModelCommand::Verify { directory } => {
                    let model = CoreConfig::load(&config_path)?.inference.model;
                    let directory = match directory {
                        Some(directory) => directory,
                        None => manager.resolve(&model)?,
                    };
                    let manifest = manager.verify(directory)?;
                    println!(
                        "{} {} {}",
                        manifest.model_id, manifest.revision, manifest.license
                    );
                }
            }
        }
        #[cfg(feature = "stt-sherpa-onnx")]
        CliCommand::Transcribe {
            input,
            model,
            threads,
        } => {
            use assistant_core::{
                SpeechRecognizer, audio::read_wav_mono, stt::SherpaOnnxRecognizer,
            };
            let (samples, sample_rate) = read_wav_mono(input)?;
            let config = CoreConfig::load(&config_path)?;
            let spec = assistant_core::models::model_spec(&config.inference.model)?;
            let model = match model {
                Some(model) => model,
                None => assistant_core::models::ModelManager::new(&models_path)
                    .resolve(&config.inference.model)?,
            };
            let transcript = SherpaOnnxRecognizer::new_for_model(
                model,
                spec,
                threads,
                config.inference.queue_capacity,
                assistant_core::CoreMetrics::default(),
            )?
            .transcribe(TranscriptionRequest {
                samples,
                sample_rate,
            })
            .await?;
            println!("{}", transcript.text);
        }
        #[cfg(feature = "stt-sherpa-onnx")]
        CliCommand::TestWakeWord { input, model } => {
            use assistant_core::{audio::read_wav_mono, wakeword::SherpaWakeWordDetector};
            let config = CoreConfig::load(&config_path)?;
            let spec = assistant_core::models::model_spec(&config.wake_word.model)?;
            let (samples, sample_rate) = read_wav_mono(input)?;
            let model = match model {
                Some(model) => model,
                None => assistant_core::models::ModelManager::new(&models_path)
                    .resolve(&config.wake_word.model)?,
            };
            let mut detector = SherpaWakeWordDetector::new_for_model_with_aliases(
                model,
                spec,
                std::iter::once(config.wake_word.keyword.as_str()),
                config.wake_word.score,
                config.wake_word.threshold,
                1,
                sample_rate,
            )?;
            let detection = samples
                .chunks((sample_rate / 50) as usize)
                .find_map(|frame| detector.process(frame));
            match detection {
                Some(detection) => {
                    println!("{} at {:.2}s", detection.keyword, detection.start_time)
                }
                None => println!("wake word not detected"),
            }
        }
        #[cfg(feature = "stt-sherpa-onnx")]
        CliCommand::Evaluate { input, model } => {
            use assistant_core::{
                SpeechRecognizer, audio::read_wav_mono, stt::SherpaOnnxRecognizer,
                wakeword::SherpaWakeWordDetector,
            };
            let config = CoreConfig::load(&config_path)?;
            let stt_spec = assistant_core::models::model_spec(&config.inference.model)?;
            let wake_spec = assistant_core::models::model_spec(&config.wake_word.model)?;
            let (samples, sample_rate) = read_wav_mono(input)?;
            let model = match model {
                Some(model) => model,
                None => assistant_core::models::ModelManager::new(&models_path)
                    .resolve(&config.inference.model)?,
            };
            let wake_model = assistant_core::models::ModelManager::new(&models_path)
                .resolve(&config.wake_word.model)?;
            let mut detector = SherpaWakeWordDetector::new_for_model_with_aliases(
                wake_model,
                wake_spec,
                std::iter::once(config.wake_word.keyword.as_str()),
                config.wake_word.score,
                config.wake_word.threshold,
                1,
                sample_rate,
            )?;
            let wake_word = samples
                .chunks((sample_rate / 50) as usize)
                .find_map(|frame| detector.process(frame));
            let transcript = SherpaOnnxRecognizer::new_for_model(
                model,
                stt_spec,
                config.inference.threads,
                config.inference.queue_capacity,
                CoreMetrics::default(),
            )?
            .transcribe(TranscriptionRequest {
                samples,
                sample_rate,
            })
            .await?;
            println!("wake_word={wake_word:?}");
            println!("transcript={}", transcript.text);
        }
    }
    Ok(())
}
