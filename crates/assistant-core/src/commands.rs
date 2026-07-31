use std::{
    collections::{BTreeSet, HashMap},
    process::Command,
    sync::Arc,
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    config::CommandConfig,
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

/// Exact-match registry built from validated command configuration.
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

    /// Finds an enabled command by exact normalized phrase.
    pub fn find(&self, text: &str) -> Option<&CommandConfig> {
        let text = normalize(text, self.normalize_yo);
        self.find_normalized(&text)
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
        let text = normalize(text, self.normalize_yo);
        self.find_normalized(&text).or_else(|| {
            wake_words.into_iter().find_map(|wake_word| {
                let wake_word = normalize(wake_word, self.normalize_yo);
                text.strip_prefix(&wake_word)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .and_then(|text| self.find_normalized(text))
            })
        })
    }

    fn find_normalized(&self, text: &str) -> Option<&CommandConfig> {
        self.commands.iter().find(|command| {
            command.enabled
                && command
                    .phrases
                    .iter()
                    .any(|phrase| normalize(phrase, self.normalize_yo) == *text)
        })
    }
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
        let handler = self
            .handlers
            .get(&command.handler)
            .ok_or_else(|| CoreError::Command(format!("unknown handler: {}", command.handler)))?;
        handler.schema().validate(&command.parameters)
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
        handler.schema().validate(parameters)?;
        handler.execute(parameters).await
    }
}

/// Legacy built-in executor for configured application allowlist entries.
pub struct LaunchAppExecutor {
    allowlist: HashMap<String, String>,
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

/// Creates the built-in handler set used by the stock daemon.
pub fn builtin_handlers(commands: &[CommandConfig]) -> Result<HandlerRegistry, CoreError> {
    let mut handlers = HandlerRegistry::new();
    handlers.register(Arc::new(LaunchAppHandler::from_commands(commands)))?;
    Ok(handlers)
}

impl LaunchAppExecutor {
    /// Builds an executable allowlist from enabled `launch_app` commands.
    pub fn from_commands(commands: &[CommandConfig]) -> Self {
        let allowlist = commands
            .iter()
            .filter(|c| c.enabled && c.handler == "launch_app")
            .filter_map(|c| {
                c.parameters
                    .get("executable")
                    .map(|path| (c.id.clone(), path.clone()))
            })
            .collect();
        Self { allowlist }
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
        let command_id = parameters
            .get("_command_id")
            .ok_or_else(|| CoreError::Command("missing command id".into()))?;
        let executable = self
            .allowlist
            .get(command_id)
            .ok_or_else(|| CoreError::Command("executable is not allowlisted".into()))?;
        Command::new(executable)
            .spawn()
            .map_err(|e| CoreError::Command(e.to_string()))?;
        Ok(CommandResult {
            message: format!("launched {command_id}"),
        })
    }
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
                handler: "launch_app".into(),
                risk: RiskLevel::Low,
                requires_confirmation: false,
                timeout_ms: 1_000,
                parameters: BTreeMap::from([("executable".into(), "notepad.exe".into())]),
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
}
