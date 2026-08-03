use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::commands::{HandlerRegistry, normalize};

/// Current on-disk and IPC configuration schema.
pub const CURRENT_CONFIG_VERSION: u16 = 6;

/// Default pinned Alphacep model used for both STT and wake-word detection.
pub const DEFAULT_MODEL_ID: &str = "alphacep/vosk-model-streaming-ru";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// Complete versioned core configuration.
pub struct CoreConfig {
    /// Configuration schema version.
    pub schema_version: u16,
    /// Audio capture and VAD settings.
    #[serde(default)]
    pub audio: AudioConfig,
    /// Text matching settings.
    #[serde(default)]
    pub matching: MatchingConfig,
    /// Wake-word detector settings.
    #[serde(default)]
    pub wake_word: WakeWordConfig,
    /// Native inference worker limits.
    #[serde(default)]
    pub inference: InferenceConfig,
    /// Command confirmation policy.
    #[serde(default)]
    pub policy: PolicyConfig,
    /// Windows named-pipe settings.
    #[serde(default)]
    pub ipc: IpcConfig,
    #[serde(default)]
    /// Configured typed commands.
    pub commands: Vec<CommandConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Audio capture, buffering, VAD and recovery settings.
pub struct AudioConfig {
    /// Device identifier or `default`.
    pub device_id: String,
    /// Canonical worker sample rate; currently 16 kHz.
    pub target_sample_rate: u32,
    /// Canonical channel count; currently mono.
    pub channels: u16,
    /// Processing frame duration in milliseconds.
    pub frame_ms: u32,
    /// Audio retained before wake-word detection.
    pub pre_roll_ms: u32,
    /// Capacity of the callback-to-worker bounded queue in frames.
    pub queue_capacity_frames: usize,
    /// Energy VAD RMS threshold in normalized sample units.
    pub vad_threshold: f32,
    /// Minimum captured command duration.
    pub command_min_ms: u32,
    /// Maximum captured command duration.
    pub command_max_ms: u32,
    /// Silence required to finish command capture.
    pub trailing_silence_ms: u32,
    /// Delay before reopening a failed input device.
    pub reconnect_delay_ms: u64,
    /// Maximum time without callback data before reopening the device.
    pub stall_timeout_ms: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device_id: "default".into(),
            target_sample_rate: 16_000,
            channels: 1,
            frame_ms: 20,
            pre_roll_ms: 900,
            queue_capacity_frames: 250,
            vad_threshold: 0.015,
            command_min_ms: 300,
            command_max_ms: 10_000,
            trailing_silence_ms: 700,
            reconnect_delay_ms: 1_000,
            stall_timeout_ms: 3_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Transcript normalization settings.
pub struct MatchingConfig {
    /// Whether Russian `ё` is normalized to `е`.
    pub normalize_yo: bool,
}

impl Default for MatchingConfig {
    fn default() -> Self {
        Self { normalize_yo: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Open-vocabulary wake-word detector settings.
pub struct WakeWordConfig {
    /// Pinned model catalog identifier used by the keyword spotter.
    pub model: String,
    /// Phrase required before a command.
    pub keyword: String,
    /// Additional phrases accepted by the same detector and command matcher.
    pub aliases: Vec<String>,
    /// Keyword boosting score passed to sherpa-onnx.
    pub score: f32,
    /// Detector activation threshold.
    pub threshold: f32,
    /// Delay before accepting another wake word.
    pub cooldown_ms: u64,
}

impl Default for WakeWordConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL_ID.into(),
            keyword: "ассистент".into(),
            aliases: Vec::new(),
            score: 1.5,
            threshold: 0.35,
            cooldown_ms: 1_200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Native STT worker limits.
pub struct InferenceConfig {
    /// Pinned model catalog identifier used for transcription.
    pub model: String,
    /// Native inference thread count; zero selects the host parallelism automatically.
    pub threads: i32,
    /// Maximum queued STT requests.
    pub queue_capacity: usize,
    /// End-to-end STT request timeout.
    pub timeout_ms: u64,
    /// Maximum automatic worker restarts before the component is faulted.
    pub max_restarts: u32,
    /// Initial exponential-backoff delay between worker restarts.
    pub restart_backoff_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL_ID.into(),
            threads: 0,
            queue_capacity: 2,
            timeout_ms: 30_000,
            max_restarts: 3,
            restart_backoff_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
/// Confirmation policy shared by all commands.
pub struct PolicyConfig {
    /// When false, even high-risk commands execute without confirmation.
    pub confirmations_enabled: bool,
    /// Lifetime of one pending confirmation.
    pub confirmation_timeout_ms: u64,
    /// Whether a pending invocation accepts speech after another wake word.
    pub voice_confirmations_enabled: bool,
    /// Normalized phrases that confirm a pending invocation.
    pub confirmation_accept_phrases: Vec<String>,
    /// Normalized phrases that cancel a pending invocation.
    pub confirmation_cancel_phrases: Vec<String>,
    /// Minimum delay between starts of the same handler; zero disables it.
    pub handler_rate_limit_ms: u64,
    /// Consecutive failures that open a handler circuit; zero disables it.
    pub handler_failure_threshold: u32,
    /// Delay before an open handler circuit permits a probe call.
    pub handler_circuit_breaker_ms: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            confirmations_enabled: true,
            confirmation_timeout_ms: 15_000,
            voice_confirmations_enabled: true,
            confirmation_accept_phrases: vec!["да".into(), "подтверждаю".into()],
            confirmation_cancel_phrases: vec!["нет".into(), "отмена".into()],
            handler_rate_limit_ms: 0,
            handler_failure_threshold: 3,
            handler_circuit_breaker_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
/// Windows named-pipe transport settings.
pub struct IpcConfig {
    /// Named pipe suffix, without `\\.\pipe\`.
    pub pipe_name: String,
    /// Per-read and per-write timeout.
    pub io_timeout_ms: u64,
}

impl Default for IpcConfig {
    fn default() -> Self {
        Self {
            pipe_name: "voice-assistant-core".into(),
            io_timeout_ms: 30_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// One exact-match command and its typed handler parameters.
pub struct CommandConfig {
    /// Stable application-defined command identifier.
    pub id: String,
    #[serde(default = "default_true")]
    /// Whether matching can select this command.
    pub enabled: bool,
    /// Exact phrases accepted after normalization.
    pub phrases: Vec<String>,
    /// Typed single-token slots referenced as `{name}` in phrases and parameters.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub slots: std::collections::BTreeMap<String, SlotConfig>,
    /// Registered [`crate::CommandHandler`] name for a single-action command.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub handler: String,
    /// Security classification used by confirmation policy.
    #[serde(default)]
    pub risk: RiskLevel,
    #[serde(default)]
    /// Whether this command explicitly requests confirmation.
    pub requires_confirmation: bool,
    #[serde(default = "default_timeout_ms")]
    /// Maximum handler execution time.
    pub timeout_ms: u64,
    #[serde(default)]
    /// Handler-specific values checked against [`crate::HandlerSchema`].
    pub parameters: std::collections::BTreeMap<String, String>,
    /// Ordered actions; mutually exclusive with `handler` and `parameters`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<CommandActionConfig>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Runtime type used to validate one captured command slot.
pub enum SlotType {
    /// One normalized value from the configured allowlist.
    Text,
    /// A signed integer within inclusive bounds.
    Integer,
    /// The literal value `true` or `false`.
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Validation policy for one command phrase slot.
pub struct SlotConfig {
    /// Slot value type.
    #[serde(rename = "type")]
    pub value_type: SlotType,
    /// Inclusive minimum for an integer slot.
    #[serde(default)]
    pub min: Option<i64>,
    /// Inclusive maximum for an integer slot.
    #[serde(default)]
    pub max: Option<i64>,
    /// Allowed normalized values for a text slot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<String>,
}

impl SlotConfig {
    /// Validates and canonicalizes one recognized slot token.
    pub fn canonicalize(&self, value: &str, normalize_yo: bool) -> Option<String> {
        match self.value_type {
            SlotType::Text => {
                let value = normalize(value, normalize_yo);
                self.values
                    .iter()
                    .map(|allowed| normalize(allowed, normalize_yo))
                    .find(|allowed| *allowed == value)
            }
            SlotType::Integer => value
                .parse::<i64>()
                .ok()
                .filter(|value| self.min.is_none_or(|min| *value >= min))
                .filter(|value| self.max.is_none_or(|max| *value <= max))
                .map(|value| value.to_string()),
            SlotType::Boolean => match value {
                "true" | "false" => Some(value.into()),
                _ => None,
            },
        }
    }

    pub(crate) fn sample(&self, normalize_yo: bool) -> Option<String> {
        match self.value_type {
            SlotType::Text => self
                .values
                .first()
                .map(|value| normalize(value, normalize_yo)),
            SlotType::Integer => self.min.map(|value| value.to_string()),
            SlotType::Boolean => Some("false".into()),
        }
    }
}

impl CommandConfig {
    /// Returns the single legacy action or the configured ordered actions.
    pub fn resolved_actions(&self) -> Vec<CommandActionConfig> {
        if self.actions.is_empty() {
            vec![CommandActionConfig {
                handler: self.handler.clone(),
                delay_ms: 0,
                parameters: self.parameters.clone(),
            }]
        } else {
            self.actions.clone()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
/// One typed action in an ordered command sequence.
pub struct CommandActionConfig {
    /// Registered [`crate::CommandHandler`] name.
    pub handler: String,
    /// Delay before this action, in milliseconds.
    #[serde(default)]
    pub delay_ms: u64,
    /// Handler-specific values checked against [`crate::HandlerSchema`].
    #[serde(default)]
    pub parameters: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Security risk classification used by confirmation policy.
pub enum RiskLevel {
    #[default]
    /// Routine local action.
    Low,
    /// Action with meaningful but limited side effects.
    Medium,
    /// Destructive or otherwise sensitive action.
    High,
}

fn default_true() -> bool {
    true
}

fn default_timeout_ms() -> u64 {
    10_000
}

#[derive(Debug, Error)]
/// Configuration I/O, parsing and validation failures.
pub enum ConfigError {
    /// Filesystem operation failed.
    #[error("cannot read config: {0}")]
    Io(#[from] std::io::Error),
    /// TOML decoding failed.
    #[error("invalid TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// A decoded value violates a core or handler invariant.
    #[error("invalid config: {0}")]
    Validation(String),
}

/// Per-user locations used by packaged applications and the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    /// `%LOCALAPPDATA%\VoiceAssistantCore`.
    pub root: PathBuf,
    /// Default `assistant.toml` path.
    pub config: PathBuf,
    /// Default model storage root.
    pub models: PathBuf,
    /// Default diagnostic log directory.
    pub logs: PathBuf,
}

impl AppPaths {
    /// Resolves `%LOCALAPPDATA%\VoiceAssistantCore` without depending on CWD.
    pub fn discover() -> Result<Self, ConfigError> {
        let local = std::env::var_os("LOCALAPPDATA").ok_or_else(|| {
            ConfigError::Validation("LOCALAPPDATA is unavailable on Windows".into())
        })?;
        let root = PathBuf::from(local).join("VoiceAssistantCore");
        Ok(Self {
            config: root.join("assistant.toml"),
            models: root.join("models"),
            logs: root.join("logs"),
            root,
        })
    }

    /// Creates application directories but never creates a configuration file.
    pub fn ensure_directories(&self) -> Result<(), ConfigError> {
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(&self.models)?;
        fs::create_dir_all(&self.logs)?;
        Ok(())
    }
}

impl CoreConfig {
    /// Loads, migrates and validates a TOML configuration.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config = Self::from_toml_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    /// Parses supported historical schemas and returns the current representation.
    pub fn from_toml_str(text: &str) -> Result<Self, ConfigError> {
        let mut value: toml::Value = toml::from_str(text)?;
        let version = value
            .get("schema_version")
            .and_then(toml::Value::as_integer)
            .ok_or_else(|| ConfigError::Validation("schema_version is required".into()))?;
        match version {
            1..=5 => {
                value["schema_version"] = toml::Value::Integer(CURRENT_CONFIG_VERSION.into());
            }
            version if version == i64::from(CURRENT_CONFIG_VERSION) => {}
            version => {
                return Err(ConfigError::Validation(format!(
                    "unsupported schema_version {version}"
                )));
            }
        }
        value.try_into().map_err(ConfigError::Toml)
    }

    /// Returns the repository's safe stock configuration.
    pub fn bundled_example() -> Result<Self, ConfigError> {
        Self::from_toml_str(include_str!("../../../config/assistant.example.toml"))
    }

    /// Migrates an IPC-supplied configuration object to the current schema.
    pub fn migrate(mut self) -> Result<Self, ConfigError> {
        match self.schema_version {
            1..=5 => self.schema_version = CURRENT_CONFIG_VERSION,
            CURRENT_CONFIG_VERSION => {}
            version => {
                return Err(ConfigError::Validation(format!(
                    "unsupported schema_version {version}"
                )));
            }
        }
        Ok(self)
    }

    /// Writes a validated configuration with same-directory atomic replacement.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        self.validate()?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .ok_or_else(|| ConfigError::Validation("config path has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| ConfigError::Validation(error.to_string()))?
            .as_nanos();
        let temporary = parent.join(format!(".assistant-{}-{nonce}.tmp", std::process::id()));
        fs::write(
            &temporary,
            toml::to_string_pretty(self)
                .map_err(|error| ConfigError::Validation(error.to_string()))?,
        )?;
        let result = replace_file(&temporary, path);
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    /// Validates handler names and parameters against installed extensions.
    pub fn validate_with_handlers(&self, handlers: &HandlerRegistry) -> Result<(), ConfigError> {
        self.validate()?;
        for command in &self.commands {
            handlers.validate_command(command).map_err(|error| {
                ConfigError::Validation(format!("command {}: {error}", command.id))
            })?;
            if command.risk != RiskLevel::High
                && command
                    .resolved_actions()
                    .iter()
                    .any(|action| action.handler == "click_mouse")
            {
                return Err(ConfigError::Validation(format!(
                    "command {}: click_mouse requires risk = \"high\"",
                    command.id
                )));
            }
        }
        Ok(())
    }

    /// Validates all structural ranges and cross-field invariants.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != CURRENT_CONFIG_VERSION {
            return Err(ConfigError::Validation(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        if self.audio.target_sample_rate != 16_000 || self.audio.channels != 1 {
            return Err(ConfigError::Validation(
                "audio must use the canonical 16 kHz mono format".into(),
            ));
        }
        if self.audio.device_id.trim().is_empty() {
            return Err(ConfigError::Validation(
                "audio.device_id must be non-empty; use \"default\" for the system input device"
                    .into(),
            ));
        }
        if !(5..=100).contains(&self.audio.frame_ms)
            || self.audio.pre_roll_ms > 5_000
            || !(1..=4_096).contains(&self.audio.queue_capacity_frames)
            || !self.audio.vad_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.audio.vad_threshold)
            || self.audio.command_min_ms < self.audio.frame_ms
            || self.audio.command_min_ms > self.audio.command_max_ms
            || !(250..=60_000).contains(&self.audio.command_max_ms)
            || self.audio.trailing_silence_ms < self.audio.frame_ms
            || self.audio.trailing_silence_ms > self.audio.command_max_ms
            || !(100..=60_000).contains(&self.audio.reconnect_delay_ms)
            || !(500..=60_000).contains(&self.audio.stall_timeout_ms)
        {
            return Err(ConfigError::Validation(
                "audio timing, queue capacity or VAD threshold is out of range".into(),
            ));
        }
        let wake_words = std::iter::once(&self.wake_word.keyword).chain(&self.wake_word.aliases);
        let mut normalized_wake_words = std::collections::HashSet::new();
        if wake_words.clone().any(|word| {
            let normalized = normalize(word, self.matching.normalize_yo);
            normalized.is_empty() || !normalized_wake_words.insert(normalized)
        }) || !self.wake_word.score.is_finite()
            || !self.wake_word.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.wake_word.threshold)
            || self.wake_word.score <= 0.0
            || self.wake_word.score > 100.0
            || self.wake_word.cooldown_ms > 60_000
        {
            return Err(ConfigError::Validation(
                "wake word must be non-empty, score positive and threshold within 0..=1".into(),
            ));
        }
        if self.wake_word.model.trim().is_empty() || self.inference.model.trim().is_empty() {
            return Err(ConfigError::Validation(
                "wake-word and STT model identifiers must be non-empty".into(),
            ));
        }
        if !(0..=64).contains(&self.inference.threads)
            || !(1..=64).contains(&self.inference.queue_capacity)
            || !(100..=300_000).contains(&self.inference.timeout_ms)
            || self.inference.max_restarts > 20
            || !(100..=60_000).contains(&self.inference.restart_backoff_ms)
        {
            return Err(ConfigError::Validation(
                "inference threads, queue capacity or timeout is out of range".into(),
            ));
        }
        let mut confirmation_phrases = std::collections::HashSet::new();
        let confirmation_phrases_valid = self
            .policy
            .confirmation_accept_phrases
            .iter()
            .chain(&self.policy.confirmation_cancel_phrases)
            .all(|phrase| {
                let phrase = normalize(phrase, self.matching.normalize_yo);
                !phrase.is_empty() && confirmation_phrases.insert(phrase)
            });
        if !(100..=300_000).contains(&self.policy.confirmation_timeout_ms)
            || !confirmation_phrases_valid
            || self.policy.confirmation_accept_phrases.is_empty()
            || self.policy.confirmation_cancel_phrases.is_empty()
            || self.policy.handler_rate_limit_ms > 300_000
            || self.policy.handler_failure_threshold > 100
            || !(100..=3_600_000).contains(&self.policy.handler_circuit_breaker_ms)
        {
            return Err(ConfigError::Validation(
                "confirmation or handler protection policy is invalid".into(),
            ));
        }
        if self.ipc.pipe_name.is_empty()
            || !self
                .ipc
                .pipe_name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '-')
            || !(100..=300_000).contains(&self.ipc.io_timeout_ms)
        {
            return Err(ConfigError::Validation(
                "IPC pipe name or timeout is invalid".into(),
            ));
        }
        let mut ids = std::collections::HashSet::new();
        let mut phrases = std::collections::HashMap::new();
        for command in &self.commands {
            if command.id.trim().is_empty()
                || command.phrases.is_empty()
                || !ids.insert(&command.id)
            {
                return Err(ConfigError::Validation(format!(
                    "command ids must be unique and phrases non-empty: {}",
                    command.id
                )));
            }
            if command.actions.len() > 32
                || (!command.actions.is_empty()
                    && (!command.handler.is_empty() || !command.parameters.is_empty()))
            {
                return Err(ConfigError::Validation(format!(
                    "command {} must use either one handler or 1..=32 actions",
                    command.id
                )));
            }
            for (name, slot) in &command.slots {
                let valid_name = !name.is_empty()
                    && name.chars().all(|character| {
                        character.is_ascii_lowercase()
                            || character.is_ascii_digit()
                            || character == '_'
                    });
                let valid_policy = match slot.value_type {
                    SlotType::Text => {
                        let mut values = std::collections::HashSet::new();
                        slot.min.is_none()
                            && slot.max.is_none()
                            && !slot.values.is_empty()
                            && slot.values.len() <= 256
                            && slot.values.iter().all(|value| {
                                let value = normalize(value, self.matching.normalize_yo);
                                !value.is_empty()
                                    && !value.contains(char::is_whitespace)
                                    && values.insert(value)
                            })
                    }
                    SlotType::Integer => {
                        slot.values.is_empty()
                            && slot.min.is_some()
                            && slot.max.is_some()
                            && slot.min <= slot.max
                    }
                    SlotType::Boolean => {
                        slot.values.is_empty() && slot.min.is_none() && slot.max.is_none()
                    }
                };
                if !valid_name || !valid_policy {
                    return Err(ConfigError::Validation(format!(
                        "command {} has invalid slot {name}",
                        command.id
                    )));
                }
            }
            for action in command.resolved_actions() {
                if action.handler.trim().is_empty() {
                    return Err(ConfigError::Validation(format!(
                        "handler is empty: {}",
                        command.id
                    )));
                }
                if action.delay_ms > 60_000 {
                    return Err(ConfigError::Validation(format!(
                        "action delay_ms is out of range: {}",
                        command.id
                    )));
                }
            }
            if !(1..=300_000).contains(&command.timeout_ms) {
                return Err(ConfigError::Validation(format!(
                    "timeout_ms is out of range: {}",
                    command.id
                )));
            }
            for phrase in &command.phrases {
                let normalized =
                    normalize_pattern(phrase, self.matching.normalize_yo, &command.slots)
                        .ok_or_else(|| {
                            ConfigError::Validation(format!(
                                "command {} has an invalid slot placeholder",
                                command.id
                            ))
                        })?;
                if normalized.is_empty() {
                    return Err(ConfigError::Validation(format!(
                        "command phrase is empty after normalization: {}",
                        command.id
                    )));
                }
                if let Some(previous) = phrases.insert(normalized, &command.id) {
                    return Err(ConfigError::Validation(format!(
                        "command phrase is ambiguous: {previous} and {}",
                        command.id
                    )));
                }
            }
            for action in command.resolved_actions() {
                for value in action.parameters.values() {
                    if let Some(name) = placeholder(value)
                        && (!command.slots.contains_key(name)
                            || command.phrases.iter().any(|phrase| {
                                !phrase
                                    .split_whitespace()
                                    .any(|token| placeholder(token) == Some(name))
                            }))
                    {
                        return Err(ConfigError::Validation(format!(
                            "command {} parameter references slot {name} not captured by every phrase",
                            command.id
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn placeholder(value: &str) -> Option<&str> {
    value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .filter(|value| !value.is_empty() && !value.contains('{') && !value.contains('}'))
}

fn normalize_pattern(
    phrase: &str,
    normalize_yo: bool,
    slots: &std::collections::BTreeMap<String, SlotConfig>,
) -> Option<String> {
    let mut normalized = Vec::new();
    for token in phrase.split_whitespace() {
        if let Some(name) = placeholder(token) {
            if !slots.contains_key(name) {
                return None;
            }
            normalized.push("{}".into());
        } else if token.contains('{') || token.contains('}') {
            return None;
        } else {
            normalized.extend(
                normalize(token, normalize_yo)
                    .split_whitespace()
                    .map(str::to_owned),
            );
        }
    }
    Some(normalized.join(" "))
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> Result<(), ConfigError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let source: Vec<u16> = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: both paths are valid NUL-terminated UTF-16 strings.
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        return Err(ConfigError::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> Result<(), ConfigError> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> CoreConfig {
        CoreConfig::bundled_example().unwrap()
    }

    #[test]
    fn example_is_valid() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn empty_audio_device_id_is_rejected_with_actionable_error() {
        let mut config = valid_config();
        config.audio.device_id = "  ".into();
        assert_eq!(
            config.validate().unwrap_err().to_string(),
            "invalid config: audio.device_id must be non-empty; use \"default\" for the system input device"
        );
    }

    #[test]
    fn normalized_phrase_collisions_are_rejected() {
        let mut config = valid_config();
        let mut duplicate = config.commands[0].clone();
        duplicate.id = "duplicate".into();
        duplicate.phrases = vec!["ОТКРОЙ, БЛОКНОТ!".into()];
        config.commands.push(duplicate);
        assert!(config.validate().is_err());
    }

    #[test]
    fn parameter_slot_must_be_captured_by_every_phrase() {
        let mut config = valid_config();
        let command = config
            .commands
            .iter_mut()
            .find(|command| command.id == "set_volume_voice")
            .unwrap();
        command.phrases.push("установи обычную громкость".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn schema_one_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml").replacen(
            "schema_version = 6",
            "schema_version = 1",
            1,
        );
        assert_eq!(
            CoreConfig::from_toml_str(&text).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
    }

    #[test]
    fn schema_two_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml").replacen(
            "schema_version = 6",
            "schema_version = 2",
            1,
        );
        assert_eq!(
            CoreConfig::from_toml_str(&text).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
    }

    #[test]
    fn schema_three_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml")
            .replacen("schema_version = 6", "schema_version = 3", 1)
            .replace("model = \"alphacep/vosk-model-streaming-ru\"\r\n", "")
            .replace("model = \"alphacep/vosk-model-streaming-ru\"\n", "");
        let config = CoreConfig::from_toml_str(&text).unwrap();
        assert_eq!(config.schema_version, CURRENT_CONFIG_VERSION);
        assert_eq!(config.inference.model, DEFAULT_MODEL_ID);
        assert_eq!(config.wake_word.model, DEFAULT_MODEL_ID);
    }

    #[test]
    fn schema_four_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml").replacen(
            "schema_version = 6",
            "schema_version = 4",
            1,
        );
        assert_eq!(
            CoreConfig::from_toml_str(&text).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
    }

    #[test]
    fn schema_five_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml").replacen(
            "schema_version = 6",
            "schema_version = 5",
            1,
        );
        assert_eq!(
            CoreConfig::from_toml_str(&text).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
    }

    #[test]
    fn duplicate_normalized_wake_word_alias_is_rejected() {
        let mut config = valid_config();
        config.wake_word.aliases = vec!["АССИСТЕНТ!".into()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn mouse_click_requires_high_risk_command() {
        let mut config = valid_config();
        let command = config
            .commands
            .iter_mut()
            .find(|command| command.id == "browser_demo")
            .unwrap();
        command.risk = RiskLevel::Low;
        let handlers = crate::builtin_handlers(&config.commands).unwrap();
        assert!(config.validate_with_handlers(&handlers).is_err());
    }

    #[test]
    fn atomic_save_roundtrips() {
        let directory =
            std::env::temp_dir().join(format!("assistant-config-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("assistant.toml");
        let config = valid_config();
        config.save_atomic(&path).unwrap();
        assert_eq!(
            CoreConfig::load(&path).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
