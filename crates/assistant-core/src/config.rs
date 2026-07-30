use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::commands::{HandlerRegistry, normalize};

/// Current on-disk and IPC configuration schema.
pub const CURRENT_CONFIG_VERSION: u16 = 2;

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
    /// Phrase required before a command.
    pub keyword: String,
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
            keyword: "ассистент".into(),
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
    /// Native inference thread count.
    pub threads: i32,
    /// Maximum queued STT requests.
    pub queue_capacity: usize,
    /// End-to-end STT request timeout.
    pub timeout_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            threads: 4,
            queue_capacity: 2,
            timeout_ms: 30_000,
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
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            confirmations_enabled: true,
            confirmation_timeout_ms: 15_000,
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
    /// Registered [`crate::CommandHandler`] name.
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
            1 => {
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
            1 => self.schema_version = CURRENT_CONFIG_VERSION,
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
        if self.wake_word.keyword.trim().is_empty()
            || !self.wake_word.score.is_finite()
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
        if !(1..=64).contains(&self.inference.threads)
            || !(1..=64).contains(&self.inference.queue_capacity)
            || !(100..=300_000).contains(&self.inference.timeout_ms)
        {
            return Err(ConfigError::Validation(
                "inference threads, queue capacity or timeout is out of range".into(),
            ));
        }
        if !(100..=300_000).contains(&self.policy.confirmation_timeout_ms) {
            return Err(ConfigError::Validation(
                "confirmation_timeout_ms is out of range".into(),
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
            if command.handler.trim().is_empty() {
                return Err(ConfigError::Validation(format!(
                    "handler is empty: {}",
                    command.handler
                )));
            }
            if !(1..=300_000).contains(&command.timeout_ms) {
                return Err(ConfigError::Validation(format!(
                    "timeout_ms is out of range: {}",
                    command.id
                )));
            }
            for phrase in &command.phrases {
                let normalized = normalize(phrase, self.matching.normalize_yo);
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
        }
        Ok(())
    }
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
    fn normalized_phrase_collisions_are_rejected() {
        let mut config = valid_config();
        let mut duplicate = config.commands[0].clone();
        duplicate.id = "duplicate".into();
        duplicate.phrases = vec!["ОТКРОЙ, БЛОКНОТ!".into()];
        config.commands.push(duplicate);
        assert!(config.validate().is_err());
    }

    #[test]
    fn schema_one_is_migrated() {
        let text = include_str!("../../../config/assistant.example.toml").replacen(
            "schema_version = 2",
            "schema_version = 1",
            1,
        );
        assert_eq!(
            CoreConfig::from_toml_str(&text).unwrap().schema_version,
            CURRENT_CONFIG_VERSION
        );
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
