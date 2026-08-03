use std::{collections::BTreeMap, time::Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::audio::AudioDeviceInfo;

/// Current incompatible-version boundary for serialized IPC envelopes.
pub const PROTOCOL_VERSION: u16 = 6;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Readiness state of one supervised daemon component.
pub enum ComponentStatus {
    /// Component is accepting work.
    Ready,
    /// Component is initializing or restarting with bounded backoff.
    Recovering,
    /// Restart budget was exhausted or external action is required.
    Faulted,
    /// Component is intentionally inactive.
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Component-level health exposed through IPC.
pub struct ComponentHealth {
    /// Stable component name.
    pub name: String,
    /// Current supervised status.
    pub status: ComponentStatus,
    /// Automatic restarts since daemon startup.
    pub restart_count: u32,
    /// Most recent component-specific failure.
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Readiness and compatibility information returned to applications.
pub struct HealthSnapshot {
    /// Current runtime state.
    pub state: AssistantState,
    /// Semantic version of the running core build.
    pub core_version: String,
    /// Whether an input stream is currently producing frames.
    pub audio_ready: bool,
    /// Currently opened input endpoint, when available.
    #[serde(default)]
    pub active_audio_device: Option<AudioDeviceInfo>,
    /// Whether a verified recognizer model is active.
    pub model_ready: bool,
    /// Loaded configuration schema.
    pub config_version: u16,
    /// Public Rust extension API version.
    pub core_api_version: u16,
    /// Serialized IPC protocol version.
    pub protocol_version: u16,
    /// Last recoverable component error, when present.
    pub last_error: Option<String>,
    /// Per-component supervised readiness and restart state.
    #[serde(default)]
    pub components: Vec<ComponentHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
/// Durable model lifecycle state.
pub enum ModelStatus {
    /// No verified model is installed.
    Missing,
    /// Installation is active.
    Installing {
        /// Number of allowlisted files copied and verified so far.
        completed_files: usize,
        /// Total allowlisted file count.
        total_files: usize,
        /// Current or most recently completed relative file.
        file: Option<String>,
    },
    /// A verified model is being loaded into the native STT engine.
    Loading {
        /// Coarse completion percentage.
        progress: u8,
        /// Stable user-visible loading stage.
        stage: String,
        /// Whether the previous recognizer remains active during replacement.
        active: bool,
    },
    /// A verified revision is active.
    Ready {
        /// Pinned model commit SHA.
        revision: String,
    },
    /// Installation or activation failed.
    Failed {
        /// User-visible failure description.
        message: String,
    },
    /// The last installation or STT loading was cancelled cooperatively.
    Cancelled,
}

#[derive(Debug, Clone)]
/// Timestamped interleaved audio frame at a device-native format.
pub struct AudioFrame {
    /// Interleaved normalized samples.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
    /// Monotonic capture timestamp.
    pub captured_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Observable runtime state machine.
pub enum AssistantState {
    /// Runtime adapters are being initialized.
    Starting,
    /// Ready and waiting for speech.
    IdleListening,
    /// Voice activity began.
    SpeechDetected,
    /// Configured keyword was detected.
    WakeWordDetected,
    /// Post-keyword command audio is being collected.
    CapturingCommand,
    /// Captured audio is in the STT worker.
    Transcribing,
    /// Transcript is being matched to configured phrases.
    MatchingCommand,
    /// A matched invocation waits for a one-time confirmation token.
    AwaitingConfirmation,
    /// A typed handler is running.
    ExecutingCommand,
    /// New wake words are temporarily suppressed.
    Cooldown,
    /// Listening is explicitly paused.
    Suspended,
    /// A recoverable component is being reset.
    Recovering,
    /// Recovery failed and external action is required.
    Faulted,
    /// Runtime shutdown has started.
    ShuttingDown,
}

impl AssistantState {
    /// Returns whether the state machine permits a direct transition.
    pub fn can_transition_to(self, next: Self) -> bool {
        use AssistantState::*;
        if next == ShuttingDown && self != ShuttingDown {
            return true;
        }
        matches!(
            (self, next),
            (Starting, IdleListening)
                | (
                    IdleListening,
                    SpeechDetected
                        | CapturingCommand
                        | MatchingCommand
                        | Suspended
                        | Recovering
                        | ShuttingDown
                )
                | (
                    SpeechDetected,
                    WakeWordDetected | IdleListening | Recovering
                )
                | (WakeWordDetected, CapturingCommand | Recovering)
                | (CapturingCommand, Transcribing | IdleListening | Recovering)
                | (Transcribing, MatchingCommand | Recovering)
                | (
                    MatchingCommand,
                    ExecutingCommand | AwaitingConfirmation | Cooldown
                )
                | (
                    AwaitingConfirmation,
                    CapturingCommand | ExecutingCommand | Cooldown | Recovering
                )
                | (Transcribing, AwaitingConfirmation)
                | (ExecutingCommand, Cooldown | Recovering)
                | (Cooldown, IdleListening | Suspended)
                | (Suspended, IdleListening | ShuttingDown)
                | (Recovering, IdleListening | Faulted | ShuttingDown)
                | (Faulted, Recovering | ShuttingDown)
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
/// Ordered events emitted by runtime and model lifecycle operations.
pub enum AssistantEvent {
    /// Runtime entered a new state.
    StateChanged {
        /// State before the transition.
        previous: AssistantState,
        /// State after the transition.
        current: AssistantState,
    },
    /// STT produced a transcript.
    TranscriptReady {
        /// Normalized recognizer output.
        text: String,
        /// Optional recognizer confidence.
        confidence: Option<f32>,
    },
    /// Streaming STT produced a non-final transcript update.
    TranscriptPartial {
        /// Current normalized recognizer output.
        text: String,
        /// Optional recognizer confidence.
        confidence: Option<f32>,
    },
    /// Current normalized microphone energy, throttled by the audio worker.
    AudioLevel {
        /// Root mean square of the latest mono frame.
        rms: f32,
    },
    /// Audio did not produce a transcript.
    TranscriptUnavailable {
        /// Stable reason suitable for GUI messaging.
        reason: TranscriptUnavailableReason,
    },
    /// Exact matching selected a command.
    CommandMatched {
        /// Configured command identifier.
        command_id: String,
    },
    /// Handler execution started.
    CommandStarted {
        /// Configured command identifier.
        command_id: String,
    },
    /// An invocation requires explicit confirmation.
    ConfirmationRequired {
        /// Configured command identifier.
        command_id: String,
        /// One-time invocation token accepted by confirm or cancel.
        confirmation_id: String,
    },
    /// A pending confirmation was cancelled explicitly, by timeout or recovery.
    ConfirmationCancelled {
        /// Configured command identifier.
        command_id: String,
        /// Invalidated one-time invocation token.
        confirmation_id: String,
        /// Why the token was invalidated.
        reason: ConfirmationCancelReason,
    },
    /// Speech captured for confirmation matched neither an accept nor a cancel phrase.
    ConfirmationUnrecognized {
        /// Recognized confirmation text.
        text: String,
    },
    /// Handler execution completed successfully.
    CommandFinished {
        /// Configured command identifier.
        command_id: String,
        /// Typed handler result.
        result: CommandResult,
    },
    /// No exact command matched the transcript.
    CommandNotMatched {
        /// Unmatched transcript.
        text: String,
    },
    /// A component failed but runtime recovery remains possible.
    RecoverableError {
        /// Stable component label such as `audio` or `stt`.
        component: String,
        /// Diagnostic failure description.
        message: String,
    },
    /// The active input endpoint changed or became unavailable.
    AudioDeviceChanged {
        /// Newly opened input endpoint, or `None` while no endpoint is available.
        device: Option<AudioDeviceInfo>,
    },
    /// A missing explicitly selected endpoint was temporarily replaced by the system default.
    AudioDeviceFallback {
        /// Configured endpoint that could not be opened.
        requested_device_id: String,
        /// System-default endpoint opened instead.
        device: AudioDeviceInfo,
    },
    /// Model installation advanced between allowlisted files.
    ModelInstallProgress {
        /// Completed allowlisted file count.
        completed_files: usize,
        /// Total allowlisted file count.
        total_files: usize,
        /// Current relative file, if any.
        file: Option<String>,
    },
    /// Native STT initialization advanced to a new stage.
    ModelLoadProgress {
        /// Coarse completion percentage.
        progress: u8,
        /// Stable user-visible loading stage.
        stage: String,
    },
    /// A verified model revision became active.
    ModelReady {
        /// Pinned model commit SHA.
        revision: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Stable reason why no transcript was produced for observed or requested audio.
pub enum TranscriptUnavailableReason {
    /// Speech ended without the configured wake word.
    WakeWordNotDetected,
    /// Capture ended before `audio.command_min_ms`.
    TooShort,
    /// Capture contained no audio above the configured VAD threshold.
    Silence,
    /// No verified speech-recognition model is active.
    ModelUnavailable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Reason a pending one-time confirmation token became invalid.
pub enum ConfirmationCancelReason {
    /// Application explicitly cancelled it.
    User,
    /// Confirmation deadline elapsed.
    Expired,
    /// Runtime recovery invalidated it.
    Recovery,
    /// Listening was suspended.
    Suspended,
    /// Runtime components or policy changed.
    Reconfigured,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Speech-recognition output.
pub struct Transcript {
    /// Recognized UTF-8 text.
    pub text: String,
    /// Optional engine-specific confidence in the range `[0, 1]`.
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
/// Owned mono audio submitted to a [`SpeechRecognizer`].
pub struct TranscriptionRequest {
    /// Normalized mono samples.
    pub samples: Vec<f32>,
    /// Samples per second.
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// Successful typed handler result.
pub struct CommandResult {
    /// Short user-visible result message.
    pub message: String,
}

/// Validated string parameters passed to a typed command handler.
pub type CommandParameters = BTreeMap<String, String>;

#[derive(Debug, Error)]
/// Typed failures returned across core adapter boundaries.
pub enum CoreError {
    /// State transition violates [`AssistantState::can_transition_to`].
    #[error("invalid state transition: {0:?} -> {1:?}")]
    InvalidTransition(AssistantState, AssistantState),
    /// Speech recognizer failed.
    #[error("speech recognition failed: {0}")]
    Recognition(String),
    /// Matching, handler validation or handler execution failed.
    #[error("command failed: {0}")]
    Command(String),
    /// Operation cannot run in the current non-idle state.
    #[error("runtime is busy: {0}")]
    Busy(String),
    /// Confirmation token is missing, expired or does not match.
    #[error("confirmation failed: {0}")]
    Confirmation(String),
    /// Required adapter or resource is not available.
    #[error("component is unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Stable machine-readable category for [`CoreError`].
pub enum CoreErrorCode {
    /// Invalid runtime state transition.
    InvalidState,
    /// Speech-recognition failure.
    Recognition,
    /// Command validation or execution failure.
    Command,
    /// Runtime is busy.
    Busy,
    /// Invalid confirmation operation.
    Confirmation,
    /// Adapter or resource unavailable.
    Unavailable,
}

impl CoreError {
    /// Stable machine-readable category for logs, IPC and extensions.
    pub fn code(&self) -> CoreErrorCode {
        match self {
            Self::InvalidTransition(_, _) => CoreErrorCode::InvalidState,
            Self::Recognition(_) => CoreErrorCode::Recognition,
            Self::Command(_) => CoreErrorCode::Command,
            Self::Busy(_) => CoreErrorCode::Busy,
            Self::Confirmation(_) => CoreErrorCode::Confirmation,
            Self::Unavailable(_) => CoreErrorCode::Unavailable,
        }
    }
}

#[async_trait]
/// Replaceable asynchronous speech-recognition adapter.
pub trait SpeechRecognizer: Send + Sync {
    /// Transcribes owned mono samples without blocking the audio callback.
    async fn transcribe(&self, request: TranscriptionRequest) -> Result<Transcript, CoreError>;

    /// Starts an optional incremental recognition session.
    async fn begin_stream(&self, _sample_rate: u32) -> Result<bool, CoreError> {
        Ok(false)
    }

    /// Adds one owned mono chunk to an active incremental session.
    async fn push_stream(&self, _samples: Vec<f32>) -> Result<(), CoreError> {
        Ok(())
    }

    /// Adds a chunk and returns a changed non-final hypothesis when supported.
    async fn push_stream_partial(
        &self,
        samples: Vec<f32>,
    ) -> Result<Option<Transcript>, CoreError> {
        self.push_stream(samples).await?;
        Ok(None)
    }

    /// Finishes an incremental session, falling back to the complete request by default.
    async fn finish_stream(&self, request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        self.transcribe(request).await
    }

    /// Replaces a failed native worker when the adapter supports supervised recovery.
    async fn recover(&self) -> Result<bool, CoreError> {
        Ok(false)
    }
}

#[async_trait]
/// Internal execution boundary implemented by [`crate::HandlerRegistry`].
pub trait CommandExecutor: Send + Sync {
    /// Executes a named handler with already validated parameters.
    async fn execute(
        &self,
        handler: &str,
        parameters: &CommandParameters,
    ) -> Result<CommandResult, CoreError>;
}
