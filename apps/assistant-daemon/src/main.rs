use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use assistant_core::{
    AppPaths, AssistantEvent, AssistantState, CORE_API_VERSION, CORE_VERSION, CommandExecutor,
    CommandRegistry, CoreConfig, CoreError, CoreMetrics, HealthSnapshot, ModelStatus,
    PROTOCOL_VERSION, Runtime, RuntimeComponents, RuntimeHandle, RuntimeUpdate, SpeechRecognizer,
    TranscriptUnavailableReason, TranscriptionRequest, UnavailableRecognizer,
    audio::{
        AudioDeviceInfo, AudioInput, default_input_device, list_input_devices, resample_linear,
    },
    builtin_handlers,
    config::CURRENT_CONFIG_VERSION,
    ipc::{
        CoreRequest, CoreResponse, IpcErrorCode,
        windows::{RequestHandler, serve_named_pipe},
    },
    models::{
        ALPHACEP_STREAMING_RU_FILES, ALPHACEP_STREAMING_RU_REVISION, ModelError, ModelManager,
    },
    signal::{CommandAudioPipeline, EnergyVad, rms},
    spawn_runtime_service,
    stt::SherpaOnnxRecognizer,
    wakeword::SherpaWakeWordDetector,
};
use clap::Parser;
use tokio::sync::{mpsc, watch};

#[derive(Parser)]
struct Args {
    /// Defaults to %LOCALAPPDATA%\VoiceAssistantCore\assistant.toml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Defaults to %LOCALAPPDATA%\VoiceAssistantCore\models.
    #[arg(long)]
    models: Option<PathBuf>,
}

#[derive(Clone)]
struct AudioSettings {
    config: Arc<CoreConfig>,
    model: Option<PathBuf>,
}

#[derive(Default)]
struct ActiveAudioDevice {
    observed: bool,
    device: Option<AudioDeviceInfo>,
}

#[derive(Clone)]
struct DaemonContext {
    runtime: RuntimeHandle,
    config: Arc<RwLock<Arc<CoreConfig>>>,
    config_path: Arc<PathBuf>,
    model_manager: ModelManager,
    model_status: Arc<Mutex<ModelStatus>>,
    model_cancel: Arc<AtomicBool>,
    model_installing: Arc<AtomicBool>,
    stt_cancel: Arc<AtomicBool>,
    stt_loading: Arc<AtomicBool>,
    apply_lock: Arc<tokio::sync::Mutex<()>>,
    audio_settings: Arc<RwLock<AudioSettings>>,
    audio_generation: Arc<AtomicU64>,
    audio_ready: Arc<AtomicBool>,
    active_audio_device: Arc<Mutex<ActiveAudioDevice>>,
    manual_capture: Arc<AtomicU8>,
    listening: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    metrics: CoreMetrics,
    last_error: Arc<Mutex<Option<String>>>,
    process_shutdown: watch::Sender<bool>,
}

enum PipelineMessage {
    WakeWord,
    Command(TranscriptionRequest),
    CaptureCancelled,
    CaptureUnavailable(TranscriptUnavailableReason),
    TranscriptUnavailable(TranscriptUnavailableReason),
    AudioLevel(f32),
    AudioDeviceChanged(Option<AudioDeviceInfo>),
    AudioDeviceFallback {
        requested_device_id: String,
        device: AudioDeviceInfo,
    },
    AudioError(String),
}

const MANUAL_CAPTURE_IDLE: u8 = 0;
const MANUAL_CAPTURE_RESERVING: u8 = 1;
const MANUAL_CAPTURE_BEGIN: u8 = 2;
const MANUAL_CAPTURE_ACTIVE: u8 = 3;
const MANUAL_CAPTURE_FINISH: u8 = 4;

#[derive(Default)]
struct MissedWakeWord {
    speech: bool,
    silence_samples: usize,
}

impl MissedWakeWord {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn observe(
        &mut self,
        frame_rms: f32,
        threshold: f32,
        frame_samples: usize,
        trailing_silence_samples: usize,
    ) -> bool {
        if frame_rms >= threshold {
            self.speech = true;
            self.silence_samples = 0;
        } else if self.speech {
            self.silence_samples = self.silence_samples.saturating_add(frame_samples);
        }
        let missed = self.speech && self.silence_samples >= trailing_silence_samples;
        if missed {
            self.reset();
        }
        missed
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("assistant_core=info,assistant_daemon=info")
        .init();
    let args = Args::parse();
    let paths = AppPaths::discover()?;
    paths.ensure_directories()?;
    let config_path = args.config.unwrap_or(paths.config);
    let model_root = args.models.unwrap_or(paths.models);
    if !config_path.is_file() {
        CoreConfig::bundled_example()?.save_atomic(&config_path)?;
    }
    let config = Arc::new(CoreConfig::load(&config_path)?);
    let model_manager = ModelManager::new(model_root);
    let metrics = CoreMetrics::default();
    validate_stock_config(&config)?;
    let components = build_runtime_components_without_stt(&config)?;
    let runtime = Runtime::new(
        components.recognizer,
        components.commands,
        components.executor,
        components.wake_word,
        components.policy,
        components.stt_timeout,
        components.cooldown,
        metrics.clone(),
    );
    let (runtime, runtime_task) =
        spawn_runtime_service(runtime, config.inference.queue_capacity.max(8))?;
    let model_status = ModelStatus::Loading {
        progress: 0,
        stage: "locating model".into(),
        active: false,
    };
    let listening = Arc::new(AtomicBool::new(false));
    let stopping = Arc::new(AtomicBool::new(false));
    let audio_ready = Arc::new(AtomicBool::new(false));
    let audio_settings = Arc::new(RwLock::new(AudioSettings {
        config: Arc::clone(&config),
        model: None,
    }));
    let audio_generation = Arc::new(AtomicU64::new(0));
    let manual_capture = Arc::new(AtomicU8::new(MANUAL_CAPTURE_IDLE));
    let (process_shutdown, mut process_shutdown_rx) = watch::channel(false);
    let context = DaemonContext {
        runtime: runtime.clone(),
        config: Arc::new(RwLock::new(config)),
        config_path: Arc::new(config_path),
        model_manager,
        model_status: Arc::new(Mutex::new(model_status)),
        model_cancel: Arc::new(AtomicBool::new(false)),
        model_installing: Arc::new(AtomicBool::new(false)),
        stt_cancel: Arc::new(AtomicBool::new(false)),
        stt_loading: Arc::new(AtomicBool::new(false)),
        apply_lock: Arc::new(tokio::sync::Mutex::new(())),
        audio_settings: Arc::clone(&audio_settings),
        audio_generation: Arc::clone(&audio_generation),
        audio_ready: Arc::clone(&audio_ready),
        active_audio_device: Arc::new(Mutex::new(ActiveAudioDevice::default())),
        manual_capture: Arc::clone(&manual_capture),
        listening: Arc::clone(&listening),
        stopping: Arc::clone(&stopping),
        metrics: metrics.clone(),
        last_error: Arc::new(Mutex::new(None)),
        process_shutdown,
    };
    let (pipeline_tx, mut pipeline_rx) = mpsc::channel(8);
    let mut audio_task = {
        let settings = Arc::clone(&audio_settings);
        let generation = Arc::clone(&audio_generation);
        let listening = Arc::clone(&listening);
        let stopping = Arc::clone(&stopping);
        let ready = Arc::clone(&audio_ready);
        let manual_capture = Arc::clone(&manual_capture);
        let metrics = metrics.clone();
        tokio::task::spawn_blocking(move || {
            audio_loop(
                settings,
                generation,
                pipeline_tx,
                listening,
                stopping,
                ready,
                manual_capture,
                metrics,
            )
        })
    };
    let bridge_context = context.clone();
    let bridge = tokio::spawn(async move {
        while let Some(message) = pipeline_rx.recv().await {
            match message {
                PipelineMessage::WakeWord => {
                    if let Err(error) = bridge_context.runtime.begin_capture().await {
                        tracing::debug!(%error, "wake word ignored while runtime is busy");
                    }
                }
                PipelineMessage::Command(request) => {
                    if let Err(error) = bridge_context.runtime.captured_audio(request).await {
                        tracing::warn!(%error, "command pipeline recovered");
                    }
                }
                PipelineMessage::CaptureCancelled => {
                    if let Err(error) = bridge_context.runtime.cancel_capture().await {
                        tracing::debug!(%error, "manual capture cancellation ignored");
                    }
                }
                PipelineMessage::CaptureUnavailable(reason) => {
                    bridge_context
                        .runtime
                        .publish_event(AssistantEvent::TranscriptUnavailable { reason });
                    if let Err(error) = bridge_context.runtime.cancel_capture().await {
                        tracing::debug!(%error, "capture cancellation ignored");
                    }
                }
                PipelineMessage::TranscriptUnavailable(reason) => bridge_context
                    .runtime
                    .publish_event(AssistantEvent::TranscriptUnavailable { reason }),
                PipelineMessage::AudioLevel(rms) => bridge_context
                    .runtime
                    .publish_event(AssistantEvent::AudioLevel { rms }),
                PipelineMessage::AudioDeviceChanged(device) => {
                    set_active_audio_device(&bridge_context, device);
                }
                PipelineMessage::AudioDeviceFallback {
                    requested_device_id,
                    device,
                } => bridge_context
                    .runtime
                    .publish_event(AssistantEvent::AudioDeviceFallback {
                        requested_device_id,
                        device,
                    }),
                PipelineMessage::AudioError(message) => {
                    set_last_error(&bridge_context, Some(message.clone()));
                    bridge_context.runtime.report_recoverable("audio", message);
                }
            }
        }
    });
    let handler = ipc_handler(context.clone());
    let ipc_config = current_config(&context).map_err(anyhow::Error::msg)?;
    let ipc = serve_named_pipe(
        &ipc_config.ipc.pipe_name,
        Duration::from_millis(ipc_config.ipc.io_timeout_ms),
        handler,
        runtime.event_sender(),
    );
    tokio::pin!(ipc);
    let initialization = tokio::spawn(initialize_stt(context.clone()));
    let mut completed_audio = None;
    tracing::info!(
        config = %context.config_path.display(),
        "voice assistant core started; press Ctrl+C to stop"
    );

    loop {
        tokio::select! {
            result = &mut ipc => {
                result?;
                break;
            }
            result = &mut audio_task => {
                completed_audio = Some(result);
                break;
            }
            result = tokio::signal::ctrl_c() => {
                result?;
                break;
            }
            result = process_shutdown_rx.changed() => {
                if result.is_err() || *process_shutdown_rx.borrow() {
                    break;
                }
            }
        }
    }

    stopping.store(true, Ordering::Release);
    context.stt_cancel.store(true, Ordering::Release);
    context.model_cancel.store(true, Ordering::Release);
    runtime.signal_shutdown();
    initialization.abort();
    bridge.abort();
    let audio_result = match completed_audio {
        Some(result) => result,
        None => audio_task.await,
    };
    match audio_result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(%error, "audio worker stopped"),
        Err(error) => tracing::warn!(%error, "audio worker join failed"),
    }
    let abort = runtime_task.abort_handle();
    if tokio::time::timeout(Duration::from_secs(2), runtime_task.join())
        .await
        .is_err()
    {
        abort.abort();
    }
    tracing::info!("voice assistant core stopped");
    Ok(())
}

fn ipc_handler(context: DaemonContext) -> RequestHandler {
    Arc::new(move |request| {
        let context = context.clone();
        Box::pin(async move { handle_request(context, request).await })
    })
}

async fn handle_request(context: DaemonContext, request: CoreRequest) -> CoreResponse {
    match request {
        CoreRequest::GetStatus => CoreResponse::Status {
            state: context.runtime.state(),
        },
        CoreRequest::GetHealth => CoreResponse::Health {
            health: health(&context),
        },
        CoreRequest::GetConfig => match current_config(&context) {
            Ok(config) => CoreResponse::Config {
                config: (*config).clone(),
            },
            Err(error) => error_response(IpcErrorCode::Internal, error),
        },
        CoreRequest::ValidateConfig { config } => {
            match config.migrate().and_then(|config| {
                validate_stock_config(&config)?;
                Ok(config)
            }) {
                Ok(_) => CoreResponse::Accepted,
                Err(error) => error_response(IpcErrorCode::Configuration, error),
            }
        }
        CoreRequest::ApplyConfig { config } => match apply_config(&context, config).await {
            Ok(()) => CoreResponse::Accepted,
            Err((code, error)) => error_response(code, error),
        },
        CoreRequest::ListAudioDevices => {
            match tokio::task::spawn_blocking(assistant_core::audio::list_input_devices).await {
                Ok(Ok(devices)) => CoreResponse::AudioDevices { devices },
                Ok(Err(error)) => error_response(IpcErrorCode::Audio, error),
                Err(error) => error_response(IpcErrorCode::Internal, error),
            }
        }
        CoreRequest::GetMetrics => CoreResponse::Metrics {
            metrics: context.metrics.snapshot(),
        },
        CoreRequest::GetModelStatus => CoreResponse::ModelStatus {
            status: current_model_status(&context),
        },
        CoreRequest::InstallModel => match start_model_install(context.clone()) {
            Ok(()) => CoreResponse::Accepted,
            Err(error) => error_response(IpcErrorCode::Model, error),
        },
        CoreRequest::CancelModelInstall => {
            if context.model_installing.load(Ordering::Acquire) {
                context.model_cancel.store(true, Ordering::Release);
                CoreResponse::Accepted
            } else if context.stt_loading.load(Ordering::Acquire) {
                context.stt_cancel.store(true, Ordering::Release);
                CoreResponse::Accepted
            } else {
                error_response(
                    IpcErrorCode::Model,
                    "no model installation or loading is running",
                )
            }
        }
        CoreRequest::VerifyModel => match verify_model(&context).await {
            Ok(status) => CoreResponse::ModelStatus { status },
            Err(error) => error_response(IpcErrorCode::Model, error),
        },
        CoreRequest::SuspendListening => match context.runtime.suspend().await {
            Ok(()) => {
                context.listening.store(false, Ordering::Release);
                CoreResponse::Accepted
            }
            Err(error) => core_error(error),
        },
        CoreRequest::ResumeListening => match context.runtime.resume().await {
            Ok(()) => {
                context.listening.store(true, Ordering::Release);
                CoreResponse::Accepted
            }
            Err(error) => core_error(error),
        },
        CoreRequest::BeginCapture => {
            if !context.listening.load(Ordering::Acquire)
                || !context.audio_ready.load(Ordering::Acquire)
            {
                if !matches!(
                    context.model_status.lock().as_deref(),
                    Ok(ModelStatus::Ready { .. })
                ) {
                    context
                        .runtime
                        .publish_event(AssistantEvent::TranscriptUnavailable {
                            reason: TranscriptUnavailableReason::ModelUnavailable,
                        });
                }
                return error_response(IpcErrorCode::Audio, "audio input is not ready");
            }
            if context
                .manual_capture
                .compare_exchange(
                    MANUAL_CAPTURE_IDLE,
                    MANUAL_CAPTURE_RESERVING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                return error_response(
                    IpcErrorCode::RuntimeBusy,
                    "manual capture is already active",
                );
            }
            match context.runtime.begin_manual_capture().await {
                Ok(()) => {
                    if context
                        .manual_capture
                        .compare_exchange(
                            MANUAL_CAPTURE_RESERVING,
                            MANUAL_CAPTURE_BEGIN,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        CoreResponse::Accepted
                    } else {
                        let _ = context.runtime.cancel_capture().await;
                        error_response(
                            IpcErrorCode::Audio,
                            "audio input changed while starting manual capture",
                        )
                    }
                }
                Err(error) => {
                    context
                        .manual_capture
                        .store(MANUAL_CAPTURE_IDLE, Ordering::Release);
                    core_error(error)
                }
            }
        }
        CoreRequest::EndCapture => {
            if context
                .manual_capture
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                    matches!(
                        state,
                        MANUAL_CAPTURE_BEGIN | MANUAL_CAPTURE_ACTIVE | MANUAL_CAPTURE_FINISH
                    )
                    .then_some(MANUAL_CAPTURE_FINISH)
                })
                .is_ok()
            {
                CoreResponse::Accepted
            } else {
                error_response(IpcErrorCode::RuntimeBusy, "manual capture is not active")
            }
        }
        CoreRequest::SubmitText { text } => {
            if text.trim().is_empty() || text.len() > 4_096 {
                error_response(
                    IpcErrorCode::InvalidRequest,
                    "submitted text must contain 1..=4096 UTF-8 bytes",
                )
            } else {
                match context.runtime.submit_text(text).await {
                    Ok(()) => CoreResponse::Accepted,
                    Err(error) => core_error(error),
                }
            }
        }
        CoreRequest::ConfirmCommand { confirmation_id } => {
            match context.runtime.confirm(confirmation_id).await {
                Ok(()) => CoreResponse::Accepted,
                Err(error) => error_response(IpcErrorCode::Confirmation, error),
            }
        }
        CoreRequest::CancelCommand { confirmation_id } => {
            match context.runtime.cancel(confirmation_id).await {
                Ok(()) => CoreResponse::Accepted,
                Err(error) => error_response(IpcErrorCode::Confirmation, error),
            }
        }
        CoreRequest::SubscribeEvents => CoreResponse::Accepted,
        CoreRequest::Shutdown => {
            context.stopping.store(true, Ordering::Release);
            context.stt_cancel.store(true, Ordering::Release);
            context.model_cancel.store(true, Ordering::Release);
            context.runtime.signal_shutdown();
            context.process_shutdown.send_replace(true);
            CoreResponse::Accepted
        }
    }
}

async fn apply_config(
    context: &DaemonContext,
    config: CoreConfig,
) -> Result<(), (IpcErrorCode, String)> {
    let config = config
        .migrate()
        .map_err(|error| (IpcErrorCode::Configuration, error.to_string()))?;
    validate_stock_config(&config)
        .map_err(|error| (IpcErrorCode::Configuration, error.to_string()))?;
    if !matches!(
        context.runtime.state(),
        AssistantState::IdleListening | AssistantState::Suspended
    ) {
        return Err((IpcErrorCode::RuntimeBusy, "runtime is busy".into()));
    }
    let _guard = context.apply_lock.lock().await;
    if !matches!(
        context.runtime.state(),
        AssistantState::IdleListening | AssistantState::Suspended
    ) {
        return Err((IpcErrorCode::RuntimeBusy, "runtime is busy".into()));
    }
    let old = current_config(context).map_err(|error| (IpcErrorCode::Internal, error))?;
    if config.ipc != old.ipc {
        return Err((
            IpcErrorCode::Configuration,
            "IPC pipe name and timeout require daemon restart".into(),
        ));
    }
    let model = current_model_path(context);
    let model_ready = model.is_some();
    let replace_stt = inference_worker_changed(&old, &config);
    let recognizer = if replace_stt {
        match model.as_deref() {
            Some(path) => Some(
                load_stt_recognizer(context, path.to_path_buf(), &config, true)
                    .await
                    .map_err(|error| (IpcErrorCode::Model, error))?,
            ),
            None => None,
        }
    } else {
        None
    };
    let update = build_runtime_update(&old, &config, recognizer).map_err(|error| {
        restore_ready_status(context, replace_stt && model_ready);
        (IpcErrorCode::Configuration, error.to_string())
    })?;
    if let Err(error) = context.runtime.reconfigure_partial(update).await {
        restore_ready_status(context, replace_stt && model_ready);
        return Err((IpcErrorCode::RuntimeBusy, error.to_string()));
    }
    if let Err(error) = config.save_atomic(context.config_path.as_ref()) {
        let rollback_recognizer = if replace_stt {
            match model.as_deref() {
                Some(path) => load_stt_recognizer(context, path.to_path_buf(), &old, true)
                    .await
                    .ok(),
                None => Some(Arc::new(UnavailableRecognizer {
                    message: "speech model is not installed".into(),
                }) as Arc<dyn SpeechRecognizer>),
            }
        } else {
            None
        };
        if let Ok(rollback) = build_runtime_update(&config, &old, rollback_recognizer) {
            let _ = context.runtime.reconfigure_partial(rollback).await;
        }
        restore_ready_status(context, replace_stt && model_ready);
        return Err((IpcErrorCode::Configuration, error.to_string()));
    }
    let config = Arc::new(config);
    *context
        .config
        .write()
        .map_err(|_| (IpcErrorCode::Internal, "config lock poisoned".into()))? =
        Arc::clone(&config);
    let restart_audio = old.audio != config.audio || old.wake_word != config.wake_word;
    update_audio_settings(context, config, model, restart_audio);
    if replace_stt && model_ready {
        set_model_ready(context);
    }
    Ok(())
}

fn start_model_install(context: DaemonContext) -> Result<(), String> {
    if context.model_installing.swap(true, Ordering::AcqRel) {
        return Err("model installation is already running".into());
    }
    context.model_cancel.store(false, Ordering::Release);
    set_model_status(
        &context,
        ModelStatus::Installing {
            completed_files: 0,
            total_files: ALPHACEP_STREAMING_RU_FILES.len(),
            file: None,
        },
    );
    tokio::spawn(async move {
        let worker_context = context.clone();
        let manager = context.model_manager.clone();
        let cancelled = Arc::clone(&context.model_cancel);
        let result = tokio::task::spawn_blocking(move || {
            manager.install_alphacep_streaming_ru_with_control(&cancelled, |progress| {
                let status = ModelStatus::Installing {
                    completed_files: progress.completed_files,
                    total_files: progress.total_files,
                    file: progress.file.clone(),
                };
                set_model_status(&worker_context, status);
                worker_context
                    .runtime
                    .publish_event(AssistantEvent::ModelInstallProgress {
                        completed_files: progress.completed_files,
                        total_files: progress.total_files,
                        file: progress.file,
                    });
            })
        })
        .await;
        match result {
            Ok(Ok(path)) => {
                if let Err(error) = activate_model(&context, path).await {
                    if context.stt_cancel.load(Ordering::Acquire) {
                        set_model_status(&context, ModelStatus::Cancelled);
                    } else {
                        set_model_status(
                            &context,
                            ModelStatus::Failed {
                                message: error.clone(),
                            },
                        );
                        set_last_error(&context, Some(error));
                    }
                }
            }
            Ok(Err(ModelError::Cancelled)) => {
                set_model_status(&context, ModelStatus::Cancelled);
            }
            Ok(Err(error)) => {
                set_model_status(
                    &context,
                    ModelStatus::Failed {
                        message: error.to_string(),
                    },
                );
                set_last_error(&context, Some(error.to_string()));
            }
            Err(error) => {
                set_model_status(
                    &context,
                    ModelStatus::Failed {
                        message: error.to_string(),
                    },
                );
            }
        }
        context.model_installing.store(false, Ordering::Release);
    });
    Ok(())
}

async fn initialize_stt(context: DaemonContext) {
    match resolve_model(&context).await {
        Ok(path) => {
            if let Err(error) = activate_model(&context, path).await
                && !context.stopping.load(Ordering::Acquire)
            {
                if context.stt_cancel.load(Ordering::Acquire) {
                    set_model_status(&context, ModelStatus::Cancelled);
                } else {
                    set_model_status(
                        &context,
                        ModelStatus::Failed {
                            message: error.clone(),
                        },
                    );
                    set_last_error(&context, Some(error));
                }
            }
        }
        Err(_) => set_model_status(&context, ModelStatus::Missing),
    }
    if !context.stopping.load(Ordering::Acquire) {
        if let Err(error) = context.runtime.start().await {
            set_last_error(&context, Some(error.to_string()));
        } else {
            context.listening.store(true, Ordering::Release);
        }
    }
}

async fn resolve_model(context: &DaemonContext) -> Result<PathBuf, String> {
    let manager = context.model_manager.clone();
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("assistant-model-resolve".into())
        .spawn(move || {
            let _ = sender.send(manager.resolve_alphacep_streaming_ru());
        })
        .map_err(|error| error.to_string())?;
    receiver
        .await
        .map_err(|_| "model resolver stopped".to_string())?
        .map_err(|error| error.to_string())
}

async fn load_stt_recognizer(
    context: &DaemonContext,
    path: PathBuf,
    config: &CoreConfig,
    active: bool,
) -> Result<Arc<dyn SpeechRecognizer>, String> {
    if context.stt_loading.swap(true, Ordering::AcqRel) {
        return Err("STT loading is already running".into());
    }
    context.stt_cancel.store(false, Ordering::Release);
    set_stt_load_progress(context, 5, "scheduling STT load", active);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let worker_context = context.clone();
    let cancelled = Arc::clone(&context.stt_cancel);
    let metrics = context.metrics.clone();
    let threads = config.inference.threads;
    let queue_capacity = config.inference.queue_capacity;
    let worker = std::thread::Builder::new()
        .name("assistant-stt-loader".into())
        .spawn(move || {
            let progress_context = worker_context.clone();
            let result = SherpaOnnxRecognizer::new_with_control(
                path,
                threads,
                queue_capacity,
                metrics,
                cancelled,
                move |progress, stage| {
                    set_stt_load_progress(&progress_context, progress, stage, active);
                },
            )
            .map(|recognizer| Arc::new(recognizer) as Arc<dyn SpeechRecognizer>)
            .map_err(|error| error.to_string());
            worker_context.stt_loading.store(false, Ordering::Release);
            let _ = sender.send(result);
        });
    if let Err(error) = worker {
        context.stt_loading.store(false, Ordering::Release);
        return Err(error.to_string());
    }
    let result = receiver
        .await
        .map_err(|_| "STT loading worker stopped".to_string())?;
    if result.is_err() && active {
        set_model_status(
            context,
            ModelStatus::Ready {
                revision: ALPHACEP_STREAMING_RU_REVISION.into(),
            },
        );
    }
    result
}

fn set_stt_load_progress(context: &DaemonContext, progress: u8, stage: &str, active: bool) {
    set_model_status(
        context,
        ModelStatus::Loading {
            progress,
            stage: stage.into(),
            active,
        },
    );
    context
        .runtime
        .publish_event(AssistantEvent::ModelLoadProgress {
            progress,
            stage: stage.into(),
        });
}

async fn activate_model(context: &DaemonContext, path: PathBuf) -> Result<(), String> {
    let _guard = context.apply_lock.lock().await;
    let config = current_config(context)?;
    while !matches!(
        context.runtime.state(),
        AssistantState::Starting | AssistantState::IdleListening | AssistantState::Suspended
    ) {
        if context.stopping.load(Ordering::Acquire) {
            return Err("STT loading cancelled".into());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let recognizer = load_stt_recognizer(
        context,
        path.clone(),
        &config,
        current_model_path(context).is_some(),
    )
    .await?;
    context
        .runtime
        .reconfigure_partial(RuntimeUpdate {
            recognizer: Some(recognizer),
            ..RuntimeUpdate::default()
        })
        .await
        .map_err(|error| error.to_string())?;
    update_audio_settings(context, config, Some(path), true);
    set_model_ready(context);
    Ok(())
}

async fn verify_model(context: &DaemonContext) -> Result<ModelStatus, String> {
    match resolve_model(context).await {
        Ok(path) => {
            activate_model(context, path).await?;
            Ok(current_model_status(context))
        }
        Err(error) => Err(error),
    }
}

fn build_runtime_components_without_stt(
    config: &CoreConfig,
) -> Result<RuntimeComponents, CoreError> {
    let handlers = builtin_handlers(&config.commands)?;
    config
        .validate_with_handlers(&handlers)
        .map_err(|error| CoreError::Command(error.to_string()))?;
    let executor: Arc<dyn CommandExecutor> = Arc::new(handlers);
    Ok(RuntimeComponents {
        recognizer: Arc::new(UnavailableRecognizer {
            message: "speech model is not installed".into(),
        }),
        commands: CommandRegistry::new(config.commands.clone(), config.matching.normalize_yo),
        executor,
        wake_word: config.wake_word.keyword.clone(),
        policy: config.policy.clone(),
        stt_timeout: Duration::from_millis(config.inference.timeout_ms),
        cooldown: Duration::from_millis(config.wake_word.cooldown_ms),
    })
}

fn build_runtime_update(
    old: &CoreConfig,
    new: &CoreConfig,
    recognizer: Option<Arc<dyn SpeechRecognizer>>,
) -> Result<RuntimeUpdate, CoreError> {
    let commands = if old.commands != new.commands || old.matching != new.matching {
        let handlers = builtin_handlers(&new.commands)?;
        Some((
            CommandRegistry::new(new.commands.clone(), new.matching.normalize_yo),
            Arc::new(handlers) as Arc<dyn CommandExecutor>,
        ))
    } else {
        None
    };
    Ok(RuntimeUpdate {
        recognizer,
        commands,
        wake_word: (old.wake_word.keyword != new.wake_word.keyword)
            .then(|| new.wake_word.keyword.clone()),
        policy: (old.policy != new.policy).then(|| new.policy.clone()),
        stt_timeout: (old.inference.timeout_ms != new.inference.timeout_ms)
            .then(|| Duration::from_millis(new.inference.timeout_ms)),
        cooldown: (old.wake_word.cooldown_ms != new.wake_word.cooldown_ms)
            .then(|| Duration::from_millis(new.wake_word.cooldown_ms)),
    })
}

fn inference_worker_changed(old: &CoreConfig, new: &CoreConfig) -> bool {
    old.inference.threads != new.inference.threads
        || old.inference.queue_capacity != new.inference.queue_capacity
}

fn validate_stock_config(config: &CoreConfig) -> Result<(), assistant_core::config::ConfigError> {
    let handlers = builtin_handlers(&config.commands)
        .map_err(|error| assistant_core::config::ConfigError::Validation(error.to_string()))?;
    config.validate_with_handlers(&handlers)
}

fn health(context: &DaemonContext) -> HealthSnapshot {
    HealthSnapshot {
        state: context.runtime.state(),
        core_version: CORE_VERSION.into(),
        audio_ready: context.audio_ready.load(Ordering::Acquire),
        active_audio_device: context
            .active_audio_device
            .lock()
            .ok()
            .and_then(|state| state.device.clone()),
        model_ready: matches!(
            current_model_status(context),
            ModelStatus::Ready { .. } | ModelStatus::Loading { active: true, .. }
        ),
        config_version: CURRENT_CONFIG_VERSION,
        core_api_version: CORE_API_VERSION,
        protocol_version: PROTOCOL_VERSION,
        last_error: context
            .last_error
            .lock()
            .ok()
            .and_then(|error| error.clone()),
    }
}

fn current_config(context: &DaemonContext) -> Result<Arc<CoreConfig>, String> {
    context
        .config
        .read()
        .map(|config| Arc::clone(&config))
        .map_err(|_| "config lock poisoned".into())
}

fn current_model_status(context: &DaemonContext) -> ModelStatus {
    context
        .model_status
        .lock()
        .map(|status| status.clone())
        .unwrap_or_else(|_| ModelStatus::Failed {
            message: "model status lock poisoned".into(),
        })
}

fn current_model_path(context: &DaemonContext) -> Option<PathBuf> {
    context
        .audio_settings
        .read()
        .ok()
        .and_then(|settings| settings.model.clone())
}

fn set_model_status(context: &DaemonContext, status: ModelStatus) {
    if let Ok(mut current) = context.model_status.lock() {
        *current = status;
    }
}

fn set_model_ready(context: &DaemonContext) {
    set_model_status(
        context,
        ModelStatus::Ready {
            revision: ALPHACEP_STREAMING_RU_REVISION.into(),
        },
    );
    context.runtime.publish_event(AssistantEvent::ModelReady {
        revision: ALPHACEP_STREAMING_RU_REVISION.into(),
    });
    set_last_error(context, None);
}

fn restore_ready_status(context: &DaemonContext, ready: bool) {
    if ready {
        set_model_status(
            context,
            ModelStatus::Ready {
                revision: ALPHACEP_STREAMING_RU_REVISION.into(),
            },
        );
    }
}

fn set_last_error(context: &DaemonContext, error: Option<String>) {
    if let Ok(mut current) = context.last_error.lock() {
        *current = error;
    }
}

fn set_active_audio_device(context: &DaemonContext, device: Option<AudioDeviceInfo>) {
    let changed = context
        .active_audio_device
        .lock()
        .map(|mut current| {
            if current.observed && current.device == device {
                false
            } else {
                current.observed = true;
                current.device = device.clone();
                true
            }
        })
        .unwrap_or(false);
    if changed {
        context
            .runtime
            .publish_event(AssistantEvent::AudioDeviceChanged { device });
    }
}

fn update_audio_settings(
    context: &DaemonContext,
    config: Arc<CoreConfig>,
    model: Option<PathBuf>,
    restart: bool,
) {
    if let Ok(mut settings) = context.audio_settings.write() {
        *settings = AudioSettings { config, model };
        if restart {
            context.audio_generation.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn core_error(error: CoreError) -> CoreResponse {
    let code = match error.code() {
        assistant_core::CoreErrorCode::InvalidState | assistant_core::CoreErrorCode::Busy => {
            IpcErrorCode::RuntimeBusy
        }
        assistant_core::CoreErrorCode::Recognition | assistant_core::CoreErrorCode::Unavailable => {
            IpcErrorCode::Model
        }
        assistant_core::CoreErrorCode::Confirmation => IpcErrorCode::Confirmation,
        assistant_core::CoreErrorCode::Command => IpcErrorCode::Internal,
    };
    error_response(code, error)
}

fn error_response(code: IpcErrorCode, error: impl std::fmt::Display) -> CoreResponse {
    CoreResponse::Error {
        code,
        message: error.to_string(),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the worker receives independent shared lifecycle and audio signals"
)]
fn audio_loop(
    settings: Arc<RwLock<AudioSettings>>,
    generation: Arc<AtomicU64>,
    commands: mpsc::Sender<PipelineMessage>,
    listening: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    manual_capture: Arc<AtomicU8>,
    metrics: CoreMetrics,
) -> Result<(), String> {
    while !stopping.load(Ordering::Acquire) {
        let current_generation = generation.load(Ordering::Acquire);
        let current = settings
            .read()
            .map_err(|_| "audio settings lock poisoned".to_string())?
            .clone();
        let Some(model) = current.model else {
            ready.store(false, Ordering::Release);
            std::thread::sleep(Duration::from_millis(200));
            continue;
        };
        let mut detector = match SherpaWakeWordDetector::new(
            model,
            &current.config.wake_word.keyword,
            current.config.wake_word.score,
            current.config.wake_word.threshold,
            1,
            current.config.audio.target_sample_rate,
        ) {
            Ok(detector) => detector,
            Err(error) => {
                let _ = commands.blocking_send(PipelineMessage::AudioError(error.to_string()));
                std::thread::sleep(Duration::from_millis(
                    current.config.audio.reconnect_delay_ms,
                ));
                continue;
            }
        };
        match audio_session(
            &current.config,
            &mut detector,
            &commands,
            &listening,
            &stopping,
            &ready,
            &manual_capture,
            &metrics,
            &generation,
            current_generation,
        ) {
            Ok(SessionEnd::Stopped) => return Ok(()),
            Ok(SessionEnd::Reconfigured | SessionEnd::InputChanged) => continue,
            Err(error) => {
                ready.store(false, Ordering::Release);
                manual_capture.store(MANUAL_CAPTURE_IDLE, Ordering::Release);
                if stopping.load(Ordering::Acquire) {
                    return Ok(());
                }
                let _ = commands.blocking_send(PipelineMessage::AudioDeviceChanged(None));
                let _ = commands.blocking_send(PipelineMessage::AudioError(error));
                metrics.audio_reconnected();
                std::thread::sleep(Duration::from_millis(
                    current.config.audio.reconnect_delay_ms,
                ));
                detector.reset();
            }
        }
    }
    Ok(())
}

enum SessionEnd {
    Stopped,
    Reconfigured,
    InputChanged,
}

#[allow(clippy::too_many_arguments)]
fn audio_session(
    config: &CoreConfig,
    detector: &mut SherpaWakeWordDetector,
    commands: &mpsc::Sender<PipelineMessage>,
    listening: &AtomicBool,
    stopping: &AtomicBool,
    ready: &AtomicBool,
    manual_capture: &AtomicU8,
    metrics: &CoreMetrics,
    generation: &AtomicU64,
    expected_generation: u64,
) -> Result<SessionEnd, String> {
    let requested_device_id = &config.audio.device_id;
    let (input, using_fallback) = match AudioInput::open(
        &config.audio.device_id,
        config.audio.queue_capacity_frames,
        metrics.clone(),
    ) {
        Ok(input) => (input, false),
        Err(selected_error)
            if requested_device_id != "default"
                && !list_input_devices()
                    .map_err(|error| error.to_string())?
                    .iter()
                    .any(|device| device.id.as_str() == requested_device_id) =>
        {
            let fallback = AudioInput::open(
                "default",
                config.audio.queue_capacity_frames,
                metrics.clone(),
            )
            .map_err(|fallback_error| {
                format!("{selected_error}; system default fallback failed: {fallback_error}")
            })?;
            commands
                .blocking_send(PipelineMessage::AudioDeviceFallback {
                    requested_device_id: requested_device_id.clone(),
                    device: fallback.device().clone(),
                })
                .map_err(|_| "runtime command queue closed".to_string())?;
            (fallback, true)
        }
        Err(error) => return Err(error.to_string()),
    };
    commands
        .blocking_send(PipelineMessage::AudioDeviceChanged(Some(
            input.device().clone(),
        )))
        .map_err(|_| "runtime command queue closed".to_string())?;
    ready.store(true, Ordering::Release);
    let target_rate = config.audio.target_sample_rate;
    let frame_samples = samples_for_ms(target_rate, config.audio.frame_ms);
    let mut buffered = VecDeque::new();
    let mut pipeline = CommandAudioPipeline::new(
        EnergyVad::new(config.audio.vad_threshold),
        samples_for_ms(target_rate, config.audio.pre_roll_ms),
        samples_for_ms(target_rate, config.audio.command_min_ms),
        samples_for_ms(target_rate, config.audio.command_max_ms),
        samples_for_ms(target_rate, config.audio.trailing_silence_ms),
    );
    let mut cooldown_until = Instant::now();
    let mut last_audio = Instant::now();
    let mut last_device_check = Instant::now();
    let mut last_level = Instant::now() - Duration::from_millis(100);
    let mut missed_wake_word = MissedWakeWord::default();

    while !stopping.load(Ordering::Acquire) {
        if generation.load(Ordering::Acquire) != expected_generation {
            ready.store(false, Ordering::Release);
            return Ok(SessionEnd::Reconfigured);
        }
        // ponytail: poll once per second; use IMMNotificationClient if sub-second switching matters.
        if (config.audio.device_id == "default" || using_fallback)
            && last_device_check.elapsed() >= Duration::from_secs(1)
        {
            last_device_check = Instant::now();
            let current_default = default_input_device().map_err(|error| error.to_string())?;
            let reopen = should_reopen_audio(
                &config.audio.device_id,
                &input.device().id,
                &current_default.id,
                using_fallback,
                using_fallback
                    && list_input_devices()
                        .map_err(|error| error.to_string())?
                        .iter()
                        .any(|device| device.id == config.audio.device_id),
            );
            if reopen {
                ready.store(false, Ordering::Release);
                let manual_active = manual_capture.swap(MANUAL_CAPTURE_IDLE, Ordering::AcqRel)
                    != MANUAL_CAPTURE_IDLE;
                if pipeline.is_collecting() || manual_active {
                    commands
                        .blocking_send(PipelineMessage::CaptureCancelled)
                        .map_err(|_| "runtime command queue closed".to_string())?;
                }
                return Ok(SessionEnd::InputChanged);
            }
        }
        let block = input
            .recv_timeout(Duration::from_millis(250))
            .map_err(|error| error.to_string())?;
        let Some(block) = block else {
            if last_audio.elapsed() >= Duration::from_millis(config.audio.stall_timeout_ms) {
                return Err("audio callback stalled".into());
            }
            continue;
        };
        last_audio = Instant::now();
        // ponytail: replace this bounded blockwise resampler only when real WAV evaluation
        // demonstrates a measurable KWS/STT regression.
        buffered.extend(resample_linear(&block, input.source_rate(), target_rate));
        while buffered.len() >= frame_samples {
            let frame: Vec<f32> = buffered.drain(..frame_samples).collect();
            let frame_rms = rms(&frame);
            if last_level.elapsed() >= Duration::from_millis(100) {
                let _ = commands.try_send(PipelineMessage::AudioLevel(frame_rms));
                last_level = Instant::now();
            }
            if !listening.load(Ordering::Acquire) {
                pipeline.reset();
                detector.reset();
                manual_capture.store(MANUAL_CAPTURE_IDLE, Ordering::Release);
                missed_wake_word.reset();
                continue;
            }
            let manual_state = manual_capture.load(Ordering::Acquire);
            if manual_state == MANUAL_CAPTURE_BEGIN {
                if manual_capture
                    .compare_exchange(
                        MANUAL_CAPTURE_BEGIN,
                        MANUAL_CAPTURE_ACTIVE,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    pipeline.start_manual();
                    detector.reset();
                }
            } else if manual_state == MANUAL_CAPTURE_FINISH {
                if !pipeline.is_collecting() {
                    pipeline.start_manual();
                }
                let captured = pipeline.finish();
                manual_capture.store(MANUAL_CAPTURE_IDLE, Ordering::Release);
                let message = match captured {
                    None => {
                        PipelineMessage::CaptureUnavailable(TranscriptUnavailableReason::TooShort)
                    }
                    Some(samples) if rms(&samples) < config.audio.vad_threshold => {
                        PipelineMessage::CaptureUnavailable(TranscriptUnavailableReason::Silence)
                    }
                    Some(samples) => PipelineMessage::Command(TranscriptionRequest {
                        samples,
                        sample_rate: target_rate,
                    }),
                };
                commands
                    .blocking_send(message)
                    .map_err(|_| "runtime command queue closed".to_string())?;
                cooldown_until =
                    Instant::now() + Duration::from_millis(config.wake_word.cooldown_ms);
                continue;
            }
            let detected = if manual_capture.load(Ordering::Acquire) == MANUAL_CAPTURE_IDLE
                && !pipeline.is_collecting()
                && Instant::now() >= cooldown_until
            {
                let started = Instant::now();
                let detected = detector.process(&frame).is_some();
                metrics.observe_kws(started.elapsed().as_micros() as u64);
                if detected {
                    commands
                        .blocking_send(PipelineMessage::WakeWord)
                        .map_err(|_| "runtime command queue closed".to_string())?;
                }
                detected
            } else {
                false
            };
            if detected {
                missed_wake_word.reset();
            } else if manual_capture.load(Ordering::Acquire) == MANUAL_CAPTURE_IDLE
                && !pipeline.is_collecting()
                && Instant::now() >= cooldown_until
                && missed_wake_word.observe(
                    frame_rms,
                    config.audio.vad_threshold,
                    frame.len(),
                    samples_for_ms(target_rate, config.audio.trailing_silence_ms),
                )
            {
                let _ = commands.try_send(PipelineMessage::TranscriptUnavailable(
                    TranscriptUnavailableReason::WakeWordNotDetected,
                ));
            }
            let was_manual = manual_capture.load(Ordering::Acquire) != MANUAL_CAPTURE_IDLE;
            if let Some(samples) = pipeline.push(&frame, detected) {
                manual_capture.store(MANUAL_CAPTURE_IDLE, Ordering::Release);
                let message = if was_manual && rms(&samples) < config.audio.vad_threshold {
                    PipelineMessage::CaptureUnavailable(TranscriptUnavailableReason::Silence)
                } else {
                    PipelineMessage::Command(TranscriptionRequest {
                        samples,
                        sample_rate: target_rate,
                    })
                };
                commands
                    .blocking_send(message)
                    .map_err(|_| "runtime command queue closed".to_string())?;
                cooldown_until =
                    Instant::now() + Duration::from_millis(config.wake_word.cooldown_ms);
            }
        }
    }
    ready.store(false, Ordering::Release);
    Ok(SessionEnd::Stopped)
}

fn should_reopen_audio(
    configured_id: &str,
    active_id: &str,
    current_default_id: &str,
    using_fallback: bool,
    selected_available: bool,
) -> bool {
    (configured_id == "default" || using_fallback) && active_id != current_default_id
        || using_fallback && selected_available
}

fn samples_for_ms(sample_rate: u32, milliseconds: u32) -> usize {
    (u64::from(sample_rate) * u64::from(milliseconds) / 1_000) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millisecond_conversion_is_exact_for_canonical_audio() {
        assert_eq!(samples_for_ms(16_000, 20), 320);
        assert_eq!(samples_for_ms(16_000, 900), 14_400);
    }

    #[test]
    fn only_a_changed_default_device_reopens_the_session() {
        assert!(should_reopen_audio(
            "default",
            "microphone-a",
            "microphone-b",
            false,
            false
        ));
        assert!(!should_reopen_audio(
            "default",
            "microphone-a",
            "microphone-a",
            false,
            false
        ));
        assert!(!should_reopen_audio(
            "microphone-a",
            "microphone-a",
            "microphone-b",
            false,
            false
        ));
        assert!(should_reopen_audio(
            "microphone-a",
            "default-a",
            "default-a",
            true,
            true
        ));
    }

    #[test]
    fn missed_wake_word_is_reported_after_trailing_silence() {
        let mut tracker = MissedWakeWord::default();
        assert!(!tracker.observe(0.2, 0.1, 10, 20));
        assert!(!tracker.observe(0.0, 0.1, 10, 20));
        assert!(tracker.observe(0.0, 0.1, 10, 20));
        assert!(!tracker.observe(0.0, 0.1, 10, 20));
    }

    #[test]
    fn non_inference_config_changes_do_not_replace_stt() {
        let old = CoreConfig::bundled_example().unwrap();
        for changed in [
            {
                let mut config = old.clone();
                config.commands[0].enabled = !config.commands[0].enabled;
                config
            },
            {
                let mut config = old.clone();
                config.policy.confirmations_enabled = !config.policy.confirmations_enabled;
                config
            },
            {
                let mut config = old.clone();
                config.audio.device_id = "another-device".into();
                config
            },
        ] {
            assert!(!inference_worker_changed(&old, &changed));
            assert!(
                build_runtime_update(&old, &changed, None)
                    .unwrap()
                    .recognizer
                    .is_none()
            );
        }
    }

    #[test]
    fn only_worker_inference_settings_replace_stt() {
        let old = CoreConfig::bundled_example().unwrap();
        let mut timeout = old.clone();
        timeout.inference.timeout_ms += 1;
        assert!(!inference_worker_changed(&old, &timeout));

        let mut threads = old.clone();
        threads.inference.threads += 1;
        assert!(inference_worker_changed(&old, &threads));
    }
}
