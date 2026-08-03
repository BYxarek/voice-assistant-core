use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    process::Command,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    config::{CommandConfig, SlotConfig, SlotType},
    domain::{CommandExecutor, CommandParameters, CommandResult, CoreError},
};

/// Normalizes case, punctuation, whitespace and optionally Russian `ё`.
pub fn normalize(text: &str, normalize_yo: bool) -> String {
    let lowered = text.to_lowercase();
    let text = if normalize_yo {
        lowered.replace('ё', "е")
    } else {
        lowered
    };
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// One matched command and its validated typed slot values.
pub struct CommandMatch<'a> {
    /// Matched command configuration.
    pub command: &'a CommandConfig,
    /// Canonical values captured from phrase placeholders.
    pub slots: CommandParameters,
}

/// Exact and typed-slot registry built from validated command configuration.
pub struct CommandRegistry {
    commands: Vec<CommandConfig>,
    normalize_yo: bool,
}

impl CommandRegistry {
    /// Builds a registry and selects whether `ё` is normalized to `е`.
    pub fn new(commands: Vec<CommandConfig>, normalize_yo: bool) -> Self {
        Self {
            commands,
            normalize_yo,
        }
    }

    /// Applies the registry's configured text normalization policy.
    pub fn normalized(&self, text: &str) -> String {
        normalize(text, self.normalize_yo)
    }

    /// Finds an enabled command by exact normalized phrase.
    pub fn find(&self, text: &str) -> Option<&CommandConfig> {
        self.match_text(text).map(|matched| matched.command)
    }

    /// Matches an exact command, optionally after a known wake-word prefix.
    pub fn find_after_wake_word(&self, text: &str, wake_word: &str) -> Option<&CommandConfig> {
        self.find_after_wake_words(text, std::iter::once(wake_word))
    }

    /// Matches an exact command, optionally after any known wake-word prefix.
    pub fn find_after_wake_words<'a>(
        &self,
        text: &str,
        wake_words: impl IntoIterator<Item = &'a str>,
    ) -> Option<&CommandConfig> {
        self.match_after_wake_words(text, wake_words)
            .map(|matched| matched.command)
    }

    /// Matches a command and returns validated values captured by `{slot}` tokens.
    pub fn match_text(&self, text: &str) -> Option<CommandMatch<'_>> {
        let text = normalize(text, self.normalize_yo);
        self.match_normalized(&text)
    }

    /// Matches a command with slots, optionally after any known wake-word prefix.
    pub fn match_after_wake_words<'a>(
        &self,
        text: &str,
        wake_words: impl IntoIterator<Item = &'a str>,
    ) -> Option<CommandMatch<'_>> {
        let text = normalize(text, self.normalize_yo);
        self.match_normalized(&text).or_else(|| {
            wake_words.into_iter().find_map(|wake_word| {
                let wake_word = normalize(wake_word, self.normalize_yo);
                text.strip_prefix(&wake_word)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .and_then(|text| self.match_normalized(text))
            })
        })
    }

    fn match_normalized(&self, text: &str) -> Option<CommandMatch<'_>> {
        self.commands.iter().find_map(|command| {
            command.enabled.then_some(command).and_then(|command| {
                command.phrases.iter().find_map(|phrase| {
                    match_phrase(phrase, text, command, self.normalize_yo)
                        .map(|slots| CommandMatch { command, slots })
                })
            })
        })
    }
}

enum PatternToken {
    Literal(String),
    Slot(String),
}

fn match_phrase(
    phrase: &str,
    text: &str,
    command: &CommandConfig,
    normalize_yo: bool,
) -> Option<CommandParameters> {
    let mut pattern = Vec::new();
    for token in phrase.split_whitespace() {
        if let Some(name) = placeholder(token) {
            pattern.push(PatternToken::Slot(name.into()));
        } else {
            pattern.extend(
                normalize(token, normalize_yo)
                    .split_whitespace()
                    .map(|token| PatternToken::Literal(token.into())),
            );
        }
    }
    let words: Vec<_> = text.split_whitespace().collect();
    if pattern.len() != words.len() {
        return None;
    }
    let mut slots = CommandParameters::new();
    for (pattern, word) in pattern.iter().zip(words) {
        match pattern {
            PatternToken::Literal(literal) if literal == word => {}
            PatternToken::Slot(name) => {
                let value = command.slots.get(name)?.canonicalize(word, normalize_yo)?;
                if slots
                    .insert(name.clone(), value.clone())
                    .is_some_and(|old| old != value)
                {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(slots)
}

fn placeholder(value: &str) -> Option<&str> {
    value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .filter(|value| !value.is_empty() && !value.contains('{') && !value.contains('}'))
}

/// Stable parameter contract published by a command handler.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HandlerSchema {
    /// Stable handler name used by command configuration.
    pub name: String,
    /// Parameters that must be present.
    pub required_parameters: BTreeSet<String>,
    /// Additional parameters accepted by the handler.
    pub optional_parameters: BTreeSet<String>,
}

impl HandlerSchema {
    /// Creates a schema from required and optional parameter names.
    pub fn new(
        name: impl Into<String>,
        required_parameters: impl IntoIterator<Item = impl Into<String>>,
        optional_parameters: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            required_parameters: required_parameters.into_iter().map(Into::into).collect(),
            optional_parameters: optional_parameters.into_iter().map(Into::into).collect(),
        }
    }

    /// Validates configured parameters without accepting internal runtime keys.
    pub fn validate(&self, parameters: &CommandParameters) -> Result<(), CoreError> {
        let configured: BTreeSet<_> = parameters
            .keys()
            .filter(|key| !key.starts_with('_'))
            .cloned()
            .collect();
        if !self.required_parameters.is_subset(&configured) {
            return Err(CoreError::Command(format!(
                "handler {} is missing required parameters",
                self.name
            )));
        }
        let allowed = self
            .required_parameters
            .union(&self.optional_parameters)
            .cloned()
            .collect::<BTreeSet<_>>();
        if !configured.is_subset(&allowed) {
            return Err(CoreError::Command(format!(
                "handler {} received unknown parameters",
                self.name
            )));
        }
        if parameters
            .iter()
            .any(|(key, value)| !key.starts_with('_') && value.trim().is_empty())
        {
            return Err(CoreError::Command(format!(
                "handler {} received an empty parameter",
                self.name
            )));
        }
        Ok(())
    }
}

/// Public extension point for application-specific typed actions.
#[async_trait]
pub trait CommandHandler: Send + Sync {
    /// Declares the stable name and parameter allowlist.
    fn schema(&self) -> HandlerSchema;

    /// Validates both the schema and handler-specific parameter values.
    fn validate_parameters(&self, parameters: &CommandParameters) -> Result<(), CoreError> {
        self.schema().validate(parameters)
    }

    /// Executes already validated typed parameters.
    async fn execute(&self, parameters: &CommandParameters) -> Result<CommandResult, CoreError>;
}

/// Runtime registry of explicitly installed command handlers.
#[derive(Clone, Default)]
pub struct HandlerRegistry {
    handlers: HashMap<String, Arc<dyn CommandHandler>>,
}

impl HandlerRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one handler, rejecting duplicate or unsafe names and schemas.
    pub fn register(&mut self, handler: Arc<dyn CommandHandler>) -> Result<(), CoreError> {
        let schema = handler.schema();
        if schema.name.is_empty()
            || !schema
                .name
                .chars()
                .all(|character| character.is_ascii_lowercase() || character == '_')
            || schema
                .required_parameters
                .intersection(&schema.optional_parameters)
                .next()
                .is_some()
            || self.handlers.contains_key(&schema.name)
        {
            return Err(CoreError::Command(format!(
                "invalid or duplicate handler schema: {}",
                schema.name
            )));
        }
        self.handlers.insert(schema.name, handler);
        Ok(())
    }

    /// Validates one command against the registered extension contract.
    pub fn validate_command(&self, command: &CommandConfig) -> Result<(), CoreError> {
        for action in command.resolved_actions() {
            let mut parameters = action.parameters;
            for value in parameters.values_mut() {
                if let Some(name) = placeholder(value) {
                    *value = command
                        .slots
                        .get(name)
                        .and_then(|slot| slot.sample(true))
                        .ok_or_else(|| CoreError::Command(format!("invalid slot: {name}")))?;
                }
            }
            self.validate_action(&action.handler, &parameters)?;
        }
        Ok(())
    }

    /// Validates one action against its installed extension contract.
    pub fn validate_action(
        &self,
        handler: &str,
        parameters: &CommandParameters,
    ) -> Result<(), CoreError> {
        self.handlers
            .get(handler)
            .ok_or_else(|| CoreError::Command(format!("unknown handler: {handler}")))?
            .validate_parameters(parameters)
    }

    /// Returns registered handler names for diagnostics.
    pub fn handler_names(&self) -> impl Iterator<Item = &str> {
        self.handlers.keys().map(String::as_str)
    }

    /// Returns stable handler contracts sorted by name for diagnostics and GUI discovery.
    pub fn schemas(&self) -> Vec<HandlerSchema> {
        let mut schemas: Vec<_> = self
            .handlers
            .values()
            .map(|handler| handler.schema())
            .collect();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }
}

#[async_trait]
impl CommandExecutor for HandlerRegistry {
    async fn execute(
        &self,
        handler: &str,
        parameters: &CommandParameters,
    ) -> Result<CommandResult, CoreError> {
        let handler = self
            .handlers
            .get(handler)
            .ok_or_else(|| CoreError::Command(format!("unknown handler: {handler}")))?;
        handler.validate_parameters(parameters)?;
        handler.execute(parameters).await
    }
}

/// Legacy built-in executor for configured application allowlist entries.
pub struct LaunchAppExecutor {
    allowlist: ParameterAllowlist,
}

/// Built-in `launch_app` extension restricted to configured command IDs.
pub struct LaunchAppHandler(LaunchAppExecutor);

impl LaunchAppHandler {
    /// Builds the executable allowlist from validated commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        Self(LaunchAppExecutor::from_commands(commands))
    }
}

#[async_trait]
impl CommandHandler for LaunchAppHandler {
    fn schema(&self) -> HandlerSchema {
        HandlerSchema::new("launch_app", ["executable"], std::iter::empty::<&str>())
    }

    async fn execute(&self, parameters: &CommandParameters) -> Result<CommandResult, CoreError> {
        self.0.execute("launch_app", parameters).await
    }
}

/// Built-in `open_url` extension restricted to configured HTTP(S) URLs.
pub struct OpenUrlHandler {
    allowlist: ParameterAllowlist,
}

impl OpenUrlHandler {
    /// Builds the URL allowlist from validated commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        Self {
            allowlist: parameter_allowlist(commands, "open_url"),
        }
    }
}

#[async_trait]
impl CommandHandler for OpenUrlHandler {
    fn schema(&self) -> HandlerSchema {
        HandlerSchema::new("open_url", ["url"], std::iter::empty::<&str>())
    }

    fn validate_parameters(&self, parameters: &CommandParameters) -> Result<(), CoreError> {
        self.schema().validate(parameters)?;
        validate_url(parameter(parameters, "url")?)
    }

    async fn execute(&self, parameters: &CommandParameters) -> Result<CommandResult, CoreError> {
        require_allowlisted(&self.allowlist, parameters)?;
        let url = parameter(parameters, "url")?.to_owned();
        tokio::task::spawn_blocking(move || open_url(&url))
            .await
            .map_err(|error| CoreError::Command(format!("open_url worker failed: {error}")))??;
        Ok(CommandResult {
            message: "opened URL".into(),
        })
    }
}

/// Built-in `set_volume` extension for the default Windows render endpoint.
pub struct SetVolumeHandler {
    allowlist: ParameterAllowlist,
}

impl SetVolumeHandler {
    /// Builds the level allowlist from validated commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        Self {
            allowlist: parameter_allowlist(commands, "set_volume"),
        }
    }
}

#[async_trait]
impl CommandHandler for SetVolumeHandler {
    fn schema(&self) -> HandlerSchema {
        HandlerSchema::new("set_volume", ["level"], std::iter::empty::<&str>())
    }

    fn validate_parameters(&self, parameters: &CommandParameters) -> Result<(), CoreError> {
        self.schema().validate(parameters)?;
        parse_volume(parameters).map(|_| ())
    }

    async fn execute(&self, parameters: &CommandParameters) -> Result<CommandResult, CoreError> {
        require_allowlisted(&self.allowlist, parameters)?;
        let level = parse_volume(parameters)?;
        set_volume(level)?;
        Ok(CommandResult {
            message: format!("volume set to {level}"),
        })
    }
}

/// Built-in high-risk mouse click with bounded smooth cursor movement.
pub struct ClickMouseHandler {
    allowlist: ParameterAllowlist,
}

impl ClickMouseHandler {
    /// Builds the click allowlist from validated commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        Self {
            allowlist: parameter_allowlist(commands, "click_mouse"),
        }
    }
}

#[async_trait]
impl CommandHandler for ClickMouseHandler {
    fn schema(&self) -> HandlerSchema {
        HandlerSchema::new("click_mouse", ["x", "y"], ["duration_ms"])
    }

    fn validate_parameters(&self, parameters: &CommandParameters) -> Result<(), CoreError> {
        self.schema().validate(parameters)?;
        parse_click(parameters).map(|_| ())
    }

    async fn execute(&self, parameters: &CommandParameters) -> Result<CommandResult, CoreError> {
        require_allowlisted(&self.allowlist, parameters)?;
        let (x, y, duration_ms) = parse_click(parameters)?;
        click_mouse(x, y, duration_ms).await?;
        Ok(CommandResult {
            message: format!("clicked at {x},{y}"),
        })
    }
}

/// Creates the built-in handler set used by the stock daemon.
pub fn builtin_handlers(commands: &[CommandConfig]) -> Result<HandlerRegistry, CoreError> {
    let mut handlers = HandlerRegistry::new();
    handlers.register(Arc::new(LaunchAppHandler::from_commands(commands)))?;
    handlers.register(Arc::new(OpenUrlHandler::from_commands(commands)))?;
    handlers.register(Arc::new(SetVolumeHandler::from_commands(commands)))?;
    handlers.register(Arc::new(ClickMouseHandler::from_commands(commands)))?;
    Ok(handlers)
}

impl LaunchAppExecutor {
    /// Builds an executable allowlist from enabled `launch_app` commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        Self {
            allowlist: parameter_allowlist(commands, "launch_app"),
        }
    }
}

#[async_trait]
impl CommandExecutor for LaunchAppExecutor {
    async fn execute(
        &self,
        handler: &str,
        parameters: &CommandParameters,
    ) -> Result<CommandResult, CoreError> {
        if handler != "launch_app" {
            return Err(CoreError::Command("handler is not allowlisted".into()));
        }
        require_allowlisted(&self.allowlist, parameters)?;
        let executable = parameter(parameters, "executable")?;
        Command::new(executable)
            .spawn()
            .map_err(|e| CoreError::Command(e.to_string()))?;
        Ok(CommandResult {
            message: format!("launched {}", parameters["_command_id"]),
        })
    }
}

#[derive(Clone)]
enum AllowedValue {
    Exact(String),
    Slot(SlotConfig),
}

type AllowedParameters = BTreeMap<String, AllowedValue>;
type ParameterAllowlist = HashMap<String, Vec<AllowedParameters>>;

fn parameter_allowlist(commands: &[CommandConfig], handler: &str) -> ParameterAllowlist {
    let mut allowlist = ParameterAllowlist::new();
    for command in commands.iter().filter(|command| command.enabled) {
        for action in command.resolved_actions() {
            if action.handler == handler {
                let parameters = action
                    .parameters
                    .into_iter()
                    .map(|(name, value)| {
                        let allowed = placeholder(&value)
                            .and_then(|slot| command.slots.get(slot))
                            .cloned()
                            .map(AllowedValue::Slot)
                            .unwrap_or(AllowedValue::Exact(value));
                        (name, allowed)
                    })
                    .collect();
                allowlist
                    .entry(command.id.clone())
                    .or_default()
                    .push(parameters);
            }
        }
    }
    allowlist
}

fn require_allowlisted(
    allowlist: &ParameterAllowlist,
    parameters: &CommandParameters,
) -> Result<(), CoreError> {
    let command_id = parameter(parameters, "_command_id")?;
    let configured: BTreeMap<_, _> = parameters
        .iter()
        .filter(|(key, _)| !key.starts_with('_'))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if allowlist.get(command_id).is_some_and(|entries| {
        entries.iter().any(|allowed| {
            allowed.len() == configured.len()
                && allowed.iter().all(|(name, rule)| {
                    configured
                        .get(name)
                        .is_some_and(|value| allowed_value_matches(rule, value))
                })
        })
    }) {
        Ok(())
    } else {
        Err(CoreError::Command("action is not allowlisted".into()))
    }
}

fn allowed_value_matches(rule: &AllowedValue, value: &str) -> bool {
    match rule {
        AllowedValue::Exact(expected) => expected == value,
        AllowedValue::Slot(slot) if slot.value_type == SlotType::Text => {
            slot.values.iter().any(|allowed| {
                normalize(allowed, true) == normalize(value, true)
                    || normalize(allowed, false) == normalize(value, false)
            })
        }
        AllowedValue::Slot(slot) => slot.canonicalize(value, false).is_some(),
    }
}

fn parameter<'a>(parameters: &'a CommandParameters, name: &str) -> Result<&'a str, CoreError> {
    parameters
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| CoreError::Command(format!("missing parameter: {name}")))
}

fn validate_url(url: &str) -> Result<(), CoreError> {
    let (scheme, remainder) = url
        .split_once("://")
        .ok_or_else(|| CoreError::Command("url must use http or https".into()))?;
    if !(scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
        || remainder.is_empty()
        || url.chars().any(char::is_whitespace)
    {
        return Err(CoreError::Command(
            "url must be a non-empty HTTP(S) URL without whitespace".into(),
        ));
    }
    Ok(())
}

fn parse_volume(parameters: &CommandParameters) -> Result<u8, CoreError> {
    parameter(parameters, "level")?
        .parse::<u8>()
        .ok()
        .filter(|level| *level <= 100)
        .ok_or_else(|| CoreError::Command("level must be an integer from 0 to 100".into()))
}

fn parse_click(parameters: &CommandParameters) -> Result<(i32, i32, u64), CoreError> {
    let coordinate = |name| {
        parameter(parameters, name)?
            .parse::<i32>()
            .map_err(|_| CoreError::Command(format!("{name} must be a 32-bit integer")))
    };
    let duration_ms = parameters
        .get("duration_ms")
        .map(|value| value.parse::<u64>())
        .transpose()
        .map_err(|_| CoreError::Command("duration_ms must be an integer".into()))?
        .unwrap_or(350);
    if !(100..=10_000).contains(&duration_ms) {
        return Err(CoreError::Command(
            "duration_ms must be from 100 to 10000".into(),
        ));
    }
    Ok((coordinate("x")?, coordinate("y")?, duration_ms))
}

#[cfg(windows)]
fn open_url(url: &str) -> Result<(), CoreError> {
    use windows::{
        Win32::UI::{
            Shell::{SEE_MASK_ASYNCOK, SEE_MASK_FLAG_NO_UI, SHELLEXECUTEINFOW, ShellExecuteExW},
            WindowsAndMessaging::SW_SHOWNORMAL,
        },
        core::PCWSTR,
    };

    let operation: Vec<u16> = "open\0".encode_utf16().collect();
    let url: Vec<u16> = url.encode_utf16().chain(std::iter::once(0)).collect();
    let mut info = SHELLEXECUTEINFOW {
        cbSize: std::mem::size_of::<SHELLEXECUTEINFOW>() as u32,
        fMask: SEE_MASK_ASYNCOK | SEE_MASK_FLAG_NO_UI,
        lpVerb: PCWSTR(operation.as_ptr()),
        lpFile: PCWSTR(url.as_ptr()),
        nShow: SW_SHOWNORMAL.0,
        ..Default::default()
    };
    // SAFETY: all pointers reference NUL-terminated UTF-16 for the duration of the call.
    unsafe { ShellExecuteExW(&mut info) }
        .map_err(|error| CoreError::Command(format!("cannot open URL: {error}")))?;
    Ok(())
}

#[cfg(not(windows))]
fn open_url(_: &str) -> Result<(), CoreError> {
    Err(CoreError::Unavailable("open_url requires Windows".into()))
}

#[cfg(windows)]
fn set_volume(level: u8) -> Result<(), CoreError> {
    use windows::Win32::{
        Media::Audio::{
            Endpoints::IAudioEndpointVolume, IMMDeviceEnumerator, MMDeviceEnumerator, eMultimedia,
            eRender,
        },
        System::Com::{
            CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
        },
    };

    struct ComGuard;
    impl Drop for ComGuard {
        fn drop(&mut self) {
            // SAFETY: paired with the successful CoInitializeEx call on this thread.
            unsafe { CoUninitialize() };
        }
    }

    // SAFETY: COM objects are created, used and released on this thread.
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED)
            .ok()
            .map_err(|error| CoreError::Command(format!("cannot initialize COM: {error}")))?;
        let _guard = ComGuard;
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|error| CoreError::Command(format!("cannot enumerate audio: {error}")))?;
        let device = enumerator
            .GetDefaultAudioEndpoint(eRender, eMultimedia)
            .map_err(|error| CoreError::Command(format!("cannot get audio endpoint: {error}")))?;
        let volume: IAudioEndpointVolume = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|error| CoreError::Command(format!("cannot control volume: {error}")))?;
        volume
            .SetMasterVolumeLevelScalar(f32::from(level) / 100.0, std::ptr::null())
            .map_err(|error| CoreError::Command(format!("cannot set volume: {error}")))?;
    }
    Ok(())
}

#[cfg(not(windows))]
fn set_volume(_: u8) -> Result<(), CoreError> {
    Err(CoreError::Unavailable("set_volume requires Windows".into()))
}

#[cfg(windows)]
async fn click_mouse(x: i32, y: i32, duration_ms: u64) -> Result<(), CoreError> {
    use windows::Win32::{
        Foundation::POINT,
        UI::{
            Input::KeyboardAndMouse::{MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, mouse_event},
            WindowsAndMessaging::{GetCursorPos, SetCursorPos},
        },
    };

    let mut start = POINT::default();
    // SAFETY: start points to writable memory.
    unsafe { GetCursorPos(&mut start) }
        .map_err(|error| CoreError::Command(format!("cannot read cursor position: {error}")))?;
    let steps = (duration_ms / 10).max(1);
    for step in 1..=steps {
        let t = step as f64 / steps as f64;
        let eased = t * t * (3.0 - 2.0 * t);
        let next_x = f64::from(start.x) + (f64::from(x) - f64::from(start.x)) * eased;
        let next_y = f64::from(start.y) + (f64::from(y) - f64::from(start.y)) * eased;
        // SAFETY: SetCursorPos accepts any pair of screen coordinates and clamps as needed.
        unsafe { SetCursorPos(next_x.round() as i32, next_y.round() as i32) }
            .map_err(|error| CoreError::Command(format!("cannot move cursor: {error}")))?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // SAFETY: mouse_event is called with documented button flags and no pointer payload.
    unsafe { mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0) };
    // SAFETY: releases the button pressed immediately above.
    unsafe { mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0) };
    Ok(())
}

#[cfg(not(windows))]
async fn click_mouse(_: i32, _: i32, _: u64) -> Result<(), CoreError> {
    Err(CoreError::Unavailable(
        "click_mouse requires Windows".into(),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::config::{CommandConfig, RiskLevel};

    #[test]
    fn normalizes_russian_phrase() {
        assert_eq!(normalize("  Открой, Ёлку! ", true), "открой елку");
    }

    #[test]
    fn matches_command_after_known_wake_word_only() {
        let registry = CommandRegistry::new(
            vec![CommandConfig {
                id: "open".into(),
                enabled: true,
                phrases: vec!["открой блокнот".into()],
                slots: BTreeMap::new(),
                handler: "launch_app".into(),
                risk: RiskLevel::Low,
                requires_confirmation: false,
                timeout_ms: 1_000,
                parameters: BTreeMap::from([("executable".into(), "notepad.exe".into())]),
                actions: Vec::new(),
            }],
            true,
        );
        assert!(
            registry
                .find_after_wake_word("ассистент, открой блокнот", "ассистент")
                .is_some()
        );
        assert!(
            registry
                .find_after_wake_word("что-нибудь открой блокнот", "ассистент")
                .is_none()
        );
        assert!(
            registry
                .find_after_wake_words("помощник, открой блокнот", ["ассистент", "помощник"],)
                .is_some()
        );
    }

    #[test]
    fn captures_and_validates_typed_integer_slot() {
        let registry = CommandRegistry::new(
            vec![CommandConfig {
                id: "volume".into(),
                enabled: true,
                phrases: vec!["громкость {level}".into()],
                slots: BTreeMap::from([(
                    "level".into(),
                    SlotConfig {
                        value_type: SlotType::Integer,
                        min: Some(0),
                        max: Some(100),
                        values: Vec::new(),
                    },
                )]),
                handler: "set_volume".into(),
                risk: RiskLevel::Low,
                requires_confirmation: false,
                timeout_ms: 1_000,
                parameters: BTreeMap::from([("level".into(), "{level}".into())]),
                actions: Vec::new(),
            }],
            true,
        );
        let matched = registry.match_text("Громкость 35").unwrap();
        assert_eq!(matched.slots["level"], "35");
        assert!(registry.match_text("громкость 101").is_none());
    }

    struct EchoHandler;

    #[async_trait]
    impl CommandHandler for EchoHandler {
        fn schema(&self) -> HandlerSchema {
            HandlerSchema::new("echo", ["value"], std::iter::empty::<&str>())
        }

        async fn execute(
            &self,
            parameters: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            Ok(CommandResult {
                message: parameters["value"].clone(),
            })
        }
    }

    #[tokio::test]
    async fn extension_handler_is_schema_checked_and_executed() {
        let mut handlers = HandlerRegistry::new();
        handlers.register(Arc::new(EchoHandler)).unwrap();
        let parameters = BTreeMap::from([("value".into(), "ok".into())]);
        assert_eq!(
            handlers.execute("echo", &parameters).await.unwrap().message,
            "ok"
        );
        assert!(handlers.execute("echo", &BTreeMap::new()).await.is_err());
        assert_eq!(handlers.schemas()[0].name, "echo");
    }

    #[test]
    fn builtin_handler_values_are_validated() {
        let handlers = builtin_handlers(&[]).unwrap();
        assert!(
            handlers
                .validate_action(
                    "open_url",
                    &BTreeMap::from([("url".into(), "file:///secret".into())]),
                )
                .is_err()
        );
        assert!(
            handlers
                .validate_action(
                    "set_volume",
                    &BTreeMap::from([("level".into(), "101".into())]),
                )
                .is_err()
        );
        assert!(
            handlers
                .validate_action(
                    "click_mouse",
                    &BTreeMap::from([
                        ("x".into(), "10".into()),
                        ("y".into(), "20".into()),
                        ("duration_ms".into(), "0".into()),
                    ]),
                )
                .is_err()
        );
    }

    #[tokio::test]
    async fn builtin_handler_rejects_parameters_outside_command_allowlist() {
        let command = CommandConfig {
            id: "site".into(),
            enabled: true,
            phrases: vec!["site".into()],
            slots: BTreeMap::new(),
            handler: "open_url".into(),
            risk: RiskLevel::Low,
            requires_confirmation: false,
            timeout_ms: 1_000,
            parameters: BTreeMap::from([("url".into(), "https://example.com".into())]),
            actions: Vec::new(),
        };
        let handlers = builtin_handlers(&[command]).unwrap();
        let parameters = BTreeMap::from([
            ("_command_id".into(), "site".into()),
            ("url".into(), "https://example.org".into()),
        ]);
        assert!(handlers.execute("open_url", &parameters).await.is_err());
    }

    #[test]
    fn builtin_allowlist_accepts_only_values_allowed_by_slot() {
        let command = CommandConfig {
            id: "volume".into(),
            enabled: true,
            phrases: vec!["громкость {level}".into()],
            slots: BTreeMap::from([(
                "level".into(),
                SlotConfig {
                    value_type: SlotType::Integer,
                    min: Some(0),
                    max: Some(100),
                    values: Vec::new(),
                },
            )]),
            handler: "set_volume".into(),
            risk: RiskLevel::Low,
            requires_confirmation: false,
            timeout_ms: 1_000,
            parameters: BTreeMap::from([("level".into(), "{level}".into())]),
            actions: Vec::new(),
        };
        let allowlist = parameter_allowlist(&[command], "set_volume");
        assert!(
            require_allowlisted(
                &allowlist,
                &BTreeMap::from([
                    ("_command_id".into(), "volume".into()),
                    ("level".into(), "35".into()),
                ]),
            )
            .is_ok()
        );
        assert!(
            require_allowlisted(
                &allowlist,
                &BTreeMap::from([
                    ("_command_id".into(), "volume".into()),
                    ("level".into(), "101".into()),
                ]),
            )
            .is_err()
        );
    }
}
