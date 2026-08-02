use serde::{Deserialize, Serialize};

use crate::{
    audio::AudioDeviceInfo,
    commands::HandlerSchema,
    config::CoreConfig,
    domain::{AssistantEvent, AssistantState, HealthSnapshot, ModelStatus, PROTOCOL_VERSION},
    metrics::MetricsSnapshot,
};

#[cfg(windows)]
/// Windows named-pipe server and reconnecting client.
pub mod windows;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Versioned request, response or event frame.
pub struct Envelope<T> {
    /// Must equal [`PROTOCOL_VERSION`].
    pub protocol_version: u16,
    /// Caller-generated identifier echoed by the response.
    pub request_id: String,
    /// Typed frame body.
    pub payload: T,
}

impl<T> Envelope<T> {
    /// Wraps a payload using the current protocol version.
    pub fn new(request_id: impl Into<String>, payload: T) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.into(),
            payload,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// Versioned requests accepted by the core IPC server.
pub enum CoreRequest {
    /// Reads only the current runtime state.
    GetStatus,
    /// Reads readiness, versions and last recoverable error.
    GetHealth,
    /// Reads the active configuration.
    GetConfig,
    /// Validates without changing runtime or disk.
    ValidateConfig {
        /// Candidate configuration.
        config: CoreConfig,
    },
    /// Validates, differentially reconfigures idle runtime and atomically persists.
    ApplyConfig {
        /// Candidate configuration.
        config: CoreConfig,
    },
    /// Lists current Windows input devices.
    ListAudioDevices,
    /// Reads a point-in-time operational metrics snapshot.
    GetMetrics,
    /// Lists command-handler capability contracts installed in the daemon.
    ListHandlers,
    /// Reads durable model lifecycle status.
    GetModelStatus,
    /// Starts pinned model installation if none is active.
    InstallModel,
    /// Requests cooperative cancellation of active installation or STT loading.
    CancelModelInstall,
    /// Verifies the installed pinned revision and activates it when possible.
    VerifyModel,
    /// Pauses listening.
    SuspendListening,
    /// Resumes listening.
    ResumeListening,
    /// Starts microphone capture without requiring the wake word.
    BeginCapture,
    /// Stops manual capture and submits audio that meets the configured minimum duration.
    EndCapture,
    /// Matches validated application text through the same command policy as recognized speech.
    SubmitText {
        /// Command phrase without a required wake-word prefix.
        text: String,
    },
    /// Executes the invocation addressed by a one-time token.
    ConfirmCommand {
        /// Token from `confirmation_required`.
        confirmation_id: String,
    },
    /// Cancels the invocation addressed by a one-time token.
    CancelCommand {
        /// Token from `confirmation_required`.
        confirmation_id: String,
    },
    /// Switches the connection to event-only streaming after `Accepted`.
    SubscribeEvents,
    /// Requests daemon and runtime shutdown.
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
/// Versioned responses and event frames returned by the core IPC server.
pub enum CoreResponse {
    /// Response to `get_status`.
    Status {
        /// Current runtime state.
        state: AssistantState,
    },
    /// Response to `get_health`.
    Health {
        /// Current readiness and compatibility snapshot.
        health: HealthSnapshot,
    },
    /// Response to `get_config`.
    Config {
        /// Active configuration.
        config: CoreConfig,
    },
    /// Response to `list_audio_devices`.
    AudioDevices {
        /// Current input devices.
        devices: Vec<AudioDeviceInfo>,
    },
    /// Response to `get_metrics`.
    Metrics {
        /// Point-in-time counters and process usage.
        metrics: MetricsSnapshot,
    },
    /// Response to `list_handlers`.
    Handlers {
        /// Installed handler schemas sorted by stable name.
        handlers: Vec<HandlerSchema>,
    },
    /// Response to `get_model_status`.
    ModelStatus {
        /// Durable model lifecycle status.
        status: ModelStatus,
    },
    /// Mutation accepted or event subscription established.
    Accepted,
    /// Typed request failure.
    Error {
        /// Stable machine-readable category.
        code: IpcErrorCode,
        /// User-visible diagnostic text.
        message: String,
    },
    /// Event-only subscription frame.
    Event {
        /// Ordered runtime or model event.
        event: AssistantEvent,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Stable IPC error category.
pub enum IpcErrorCode {
    /// Malformed or unsupported request.
    InvalidRequest,
    /// Envelope protocol version does not match.
    ProtocolVersion,
    /// Configuration failed parsing, migration or validation.
    Configuration,
    /// Audio device operation failed.
    Audio,
    /// Model operation failed.
    Model,
    /// Runtime cannot accept the operation in its current state.
    RuntimeBusy,
    /// Confirmation token operation failed.
    Confirmation,
    /// Unexpected server failure.
    Internal,
}

/// Serializes an envelope as UTF-8 JSON without the transport length prefix.
pub fn encode<T: Serialize>(message: &Envelope<T>) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(message)
}

/// Deserializes one complete UTF-8 JSON envelope.
pub fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<Envelope<T>, serde_json::Error> {
    serde_json::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_roundtrip() {
        let message = Envelope::new("42", CoreRequest::GetStatus);
        let decoded: Envelope<CoreRequest> = decode(&encode(&message).unwrap()).unwrap();
        assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
        assert_eq!(decoded.request_id, "42");
    }

    #[test]
    fn protocol_version_is_part_of_the_contract() {
        let mut message = Envelope::new("42", CoreRequest::GetStatus);
        message.protocol_version += 1;
        let decoded: Envelope<CoreRequest> = decode(&encode(&message).unwrap()).unwrap();
        assert_ne!(decoded.protocol_version, PROTOCOL_VERSION);
    }

    #[test]
    fn get_status_wire_format_is_stable() {
        let message = Envelope::new("golden-1", CoreRequest::GetStatus);
        assert_eq!(
            String::from_utf8(encode(&message).unwrap()).unwrap(),
            r#"{"protocol_version":5,"request_id":"golden-1","payload":{"type":"get_status"}}"#
        );
    }

    #[test]
    fn manual_input_wire_formats_are_stable() {
        let begin = Envelope::new("manual-1", CoreRequest::BeginCapture);
        assert_eq!(
            String::from_utf8(encode(&begin).unwrap()).unwrap(),
            r#"{"protocol_version":5,"request_id":"manual-1","payload":{"type":"begin_capture"}}"#
        );
        let text = Envelope::new(
            "manual-2",
            CoreRequest::SubmitText {
                text: "open notepad".into(),
            },
        );
        assert_eq!(
            String::from_utf8(encode(&text).unwrap()).unwrap(),
            r#"{"protocol_version":5,"request_id":"manual-2","payload":{"type":"submit_text","text":"open notepad"}}"#
        );
    }

    #[test]
    fn microphone_diagnostic_events_have_stable_wire_formats() {
        let level = Envelope::new(
            "event-1",
            CoreResponse::Event {
                event: AssistantEvent::AudioLevel { rms: 0.25 },
            },
        );
        assert_eq!(
            String::from_utf8(encode(&level).unwrap()).unwrap(),
            r#"{"protocol_version":5,"request_id":"event-1","payload":{"type":"event","event":{"type":"audio_level","rms":0.25}}}"#
        );
        let unavailable = Envelope::new(
            "event-2",
            CoreResponse::Event {
                event: AssistantEvent::TranscriptUnavailable {
                    reason: crate::TranscriptUnavailableReason::Silence,
                },
            },
        );
        assert_eq!(
            String::from_utf8(encode(&unavailable).unwrap()).unwrap(),
            r#"{"protocol_version":5,"request_id":"event-2","payload":{"type":"event","event":{"type":"transcript_unavailable","reason":"silence"}}}"#
        );
    }
}
