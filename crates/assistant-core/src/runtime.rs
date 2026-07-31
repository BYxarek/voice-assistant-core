use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use tokio::sync::{broadcast, watch};

use crate::{
    commands::CommandRegistry,
    config::{PolicyConfig, RiskLevel},
    domain::{
        AssistantEvent, AssistantState, CommandExecutor, CommandParameters,
        ConfirmationCancelReason, CoreError, SpeechRecognizer, Transcript, TranscriptionRequest,
    },
    metrics::CoreMetrics,
    service::RuntimeUpdate,
};

struct PendingCommand {
    id: String,
    handler: String,
    parameters: CommandParameters,
    timeout: Duration,
    expires_at: Instant,
    confirmation_id: Option<String>,
}

const MAX_SUBMITTED_TEXT_BYTES: usize = 4_096;

/// Stateful command pipeline shared by the daemon and diagnostic CLI.
pub struct Runtime {
    state: AssistantState,
    recognizer: Arc<dyn SpeechRecognizer>,
    registry: CommandRegistry,
    executor: Arc<dyn CommandExecutor>,
    events: broadcast::Sender<AssistantEvent>,
    state_changes: watch::Sender<AssistantState>,
    metrics: CoreMetrics,
    wake_word: String,
    policy: PolicyConfig,
    stt_timeout: Duration,
    cooldown: Duration,
    pending: Option<PendingCommand>,
    confirmation_sequence: u64,
}

impl Runtime {
    /// Creates a runtime with explicit wake-word, policy and shared metrics.
    #[allow(
        clippy::too_many_arguments,
        reason = "the constructor exposes independent runtime adapters and bounded timings"
    )]
    pub fn new(
        recognizer: Arc<dyn SpeechRecognizer>,
        registry: CommandRegistry,
        executor: Arc<dyn CommandExecutor>,
        wake_word: String,
        policy: PolicyConfig,
        stt_timeout: Duration,
        cooldown: Duration,
        metrics: CoreMetrics,
    ) -> Self {
        let (events, _) = broadcast::channel(128);
        let (state_changes, _) = watch::channel(AssistantState::Starting);
        Self {
            state: AssistantState::Starting,
            recognizer,
            registry,
            executor,
            events,
            state_changes,
            metrics,
            wake_word,
            policy,
            stt_timeout,
            cooldown,
            pending: None,
            confirmation_sequence: 0,
        }
    }

    /// Subscribes to state, transcript, confirmation and execution events.
    pub fn subscribe(&self) -> broadcast::Receiver<AssistantEvent> {
        self.events.subscribe()
    }

    /// Returns a sender used by IPC event subscriptions.
    pub fn event_sender(&self) -> broadcast::Sender<AssistantEvent> {
        self.events.clone()
    }

    /// Subscribes to the latest state without waiting for the runtime task.
    pub fn state_receiver(&self) -> watch::Receiver<AssistantState> {
        self.state_changes.subscribe()
    }

    /// Current state of the state machine.
    pub fn state(&self) -> AssistantState {
        self.state
    }

    /// Validates and publishes one state transition.
    pub fn transition(&mut self, next: AssistantState) -> Result<(), CoreError> {
        if !self.state.can_transition_to(next) {
            return Err(CoreError::InvalidTransition(self.state, next));
        }
        let previous = std::mem::replace(&mut self.state, next);
        self.state_changes.send_replace(next);
        let _ = self.events.send(AssistantEvent::StateChanged {
            previous,
            current: next,
        });
        Ok(())
    }

    /// Moves a new runtime into idle listening.
    pub fn start(&mut self) -> Result<(), CoreError> {
        self.transition(AssistantState::IdleListening)
    }

    /// Stops audio processing without shutting down the daemon.
    pub fn suspend(&mut self) -> Result<(), CoreError> {
        if self.state == AssistantState::Suspended {
            return Ok(());
        }
        if self.state == AssistantState::AwaitingConfirmation {
            self.cancel_pending(ConfirmationCancelReason::Suspended)?;
        }
        self.transition(AssistantState::Suspended)
    }

    /// Resumes a suspended runtime.
    pub fn resume(&mut self) -> Result<(), CoreError> {
        if self.state == AssistantState::IdleListening {
            return Ok(());
        }
        self.transition(AssistantState::IdleListening)
    }

    /// Moves any active state to shutdown.
    pub fn shutdown(&mut self) -> Result<(), CoreError> {
        if self.state == AssistantState::ShuttingDown {
            return Ok(());
        }
        self.pending = None;
        self.transition(AssistantState::ShuttingDown)
    }

    /// Publishes a recoverable component error and restores idle state.
    pub fn report_recoverable(&mut self, component: &str, message: impl Into<String>) {
        if self.state == AssistantState::ShuttingDown {
            return;
        }
        self.metrics.recoverable_error();
        let _ = self.events.send(AssistantEvent::RecoverableError {
            component: component.into(),
            message: message.into(),
        });
        if let Some(pending) = self.pending.take() {
            let _ = self.events.send(AssistantEvent::ConfirmationCancelled {
                command_id: pending.id,
                confirmation_id: pending.confirmation_id.unwrap_or_default(),
                reason: ConfirmationCancelReason::Recovery,
            });
        }
        if self.state.can_transition_to(AssistantState::Recovering) {
            let _ = self.transition(AssistantState::Recovering);
            let _ = self.transition(AssistantState::IdleListening);
        }
    }

    /// Transcribes captured command audio and applies matching and policy.
    pub async fn process_command_audio(
        &mut self,
        request: TranscriptionRequest,
    ) -> Result<(), CoreError> {
        self.start_command_capture()?;
        self.process_captured_audio(request).await
    }

    /// Publishes wake-word and capture states at detection time.
    pub fn start_command_capture(&mut self) -> Result<(), CoreError> {
        self.transition(AssistantState::SpeechDetected)?;
        self.transition(AssistantState::WakeWordDetected)?;
        self.transition(AssistantState::CapturingCommand)
    }

    /// Starts capture requested by a trusted application without a wake word.
    pub fn start_manual_capture(&mut self) -> Result<(), CoreError> {
        self.transition(AssistantState::CapturingCommand)
    }

    /// Returns an empty or too-short manual capture to idle listening.
    pub fn cancel_capture(&mut self) -> Result<(), CoreError> {
        self.transition(AssistantState::IdleListening)
    }

    /// Matches application-supplied text through the normal command policy.
    pub async fn process_text(&mut self, text: String) -> Result<(), CoreError> {
        if text.trim().is_empty() || text.len() > MAX_SUBMITTED_TEXT_BYTES {
            return Err(CoreError::Command(format!(
                "submitted text must contain 1..={MAX_SUBMITTED_TEXT_BYTES} UTF-8 bytes"
            )));
        }
        self.transition(AssistantState::MatchingCommand)?;
        self.match_text(text, false).await
    }

    /// Transcribes audio after an already published capture phase.
    pub async fn process_captured_audio(
        &mut self,
        request: TranscriptionRequest,
    ) -> Result<(), CoreError> {
        self.transition(AssistantState::Transcribing)?;

        let started = Instant::now();
        let outcome =
            tokio::time::timeout(self.stt_timeout, self.recognizer.transcribe(request)).await;
        self.metrics
            .observe_stt(started.elapsed().as_millis() as u64);
        let transcript = match outcome {
            Ok(Ok(transcript)) => transcript,
            Ok(Err(error)) => {
                if matches!(error, CoreError::Unavailable(_)) {
                    let _ = self.events.send(AssistantEvent::TranscriptUnavailable {
                        reason: crate::TranscriptUnavailableReason::ModelUnavailable,
                    });
                }
                return self.fail("stt", error);
            }
            Err(_) => return self.fail("stt", CoreError::Recognition("STT timed out".into())),
        };
        if transcript.text.trim().is_empty() {
            let _ = self.events.send(AssistantEvent::TranscriptUnavailable {
                reason: crate::TranscriptUnavailableReason::Silence,
            });
            self.transition(AssistantState::MatchingCommand)?;
            self.enter_cooldown()?;
            return Ok(());
        }
        let _ = self.events.send(AssistantEvent::TranscriptReady {
            text: transcript.text.clone(),
            confidence: transcript.confidence,
        });
        self.transition(AssistantState::MatchingCommand)?;
        self.match_text(transcript.text, true).await
    }

    async fn match_text(&mut self, text: String, allow_wake_word: bool) -> Result<(), CoreError> {
        let command = if allow_wake_word {
            self.registry.find_after_wake_word(&text, &self.wake_word)
        } else {
            self.registry.find(&text)
        };
        let Some(command) = command else {
            let _ = self.events.send(AssistantEvent::CommandNotMatched { text });
            self.enter_cooldown()?;
            return Ok(());
        };

        let id = command.id.clone();
        let mut parameters = command.parameters.clone();
        parameters.insert("_command_id".into(), id.clone());
        let pending = PendingCommand {
            id: id.clone(),
            handler: command.handler.clone(),
            parameters,
            timeout: Duration::from_millis(command.timeout_ms),
            expires_at: Instant::now() + Duration::from_millis(self.policy.confirmation_timeout_ms),
            confirmation_id: None,
        };
        let requires_confirmation = self.policy.confirmations_enabled
            && (command.requires_confirmation || command.risk == RiskLevel::High);
        let _ = self.events.send(AssistantEvent::CommandMatched {
            command_id: id.clone(),
        });

        if requires_confirmation {
            self.confirmation_sequence = self.confirmation_sequence.wrapping_add(1);
            let confirmation_id = format!("{id}-{:016x}", self.confirmation_sequence);
            let mut pending = pending;
            pending.confirmation_id = Some(confirmation_id.clone());
            self.pending = Some(pending);
            let _ = self.events.send(AssistantEvent::ConfirmationRequired {
                command_id: id,
                confirmation_id,
            });
            self.transition(AssistantState::AwaitingConfirmation)?;
            return Ok(());
        }

        self.execute(pending).await
    }

    /// Executes the matching pending command before its deadline.
    pub async fn confirm_command(&mut self, confirmation_id: &str) -> Result<(), CoreError> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| CoreError::Confirmation("no command is awaiting confirmation".into()))?;
        if pending.confirmation_id.as_deref() != Some(confirmation_id) {
            self.pending = Some(pending);
            return Err(CoreError::Confirmation(
                "confirmation command id does not match".into(),
            ));
        }
        if Instant::now() >= pending.expires_at {
            self.pending = Some(pending);
            self.cancel_pending(ConfirmationCancelReason::Expired)?;
            return Err(CoreError::Confirmation("confirmation expired".into()));
        }
        self.execute(pending).await
    }

    /// Cancels the matching pending command.
    pub fn cancel_command(&mut self, confirmation_id: &str) -> Result<(), CoreError> {
        if self
            .pending
            .as_ref()
            .and_then(|pending| pending.confirmation_id.as_deref())
            != Some(confirmation_id)
        {
            return Err(CoreError::Confirmation(
                "confirmation command id does not match".into(),
            ));
        }
        self.cancel_pending(ConfirmationCancelReason::User)
    }

    /// Cancels an expired confirmation, returning whether one expired.
    pub fn expire_confirmation(&mut self) -> Result<bool, CoreError> {
        let expired = self
            .pending
            .as_ref()
            .is_some_and(|pending| Instant::now() >= pending.expires_at);
        if expired {
            self.cancel_pending(ConfirmationCancelReason::Expired)?;
        }
        Ok(expired)
    }

    async fn execute(&mut self, command: PendingCommand) -> Result<(), CoreError> {
        self.transition(AssistantState::ExecutingCommand)?;
        let _ = self.events.send(AssistantEvent::CommandStarted {
            command_id: command.id.clone(),
        });
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            command.timeout,
            self.executor.execute(&command.handler, &command.parameters),
        )
        .await;
        self.metrics
            .observe_handler(started.elapsed().as_millis() as u64);
        let result = match outcome {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => return self.fail("command", error),
            Err(_) => return self.fail("command", CoreError::Command("command timed out".into())),
        };
        self.metrics.command_completed();
        let _ = self.events.send(AssistantEvent::CommandFinished {
            command_id: command.id,
            result,
        });
        self.enter_cooldown()
    }

    /// Returns the current one-time confirmation token.
    pub fn pending_confirmation_id(&self) -> Option<&str> {
        self.pending
            .as_ref()
            .and_then(|pending| pending.confirmation_id.as_deref())
    }

    /// Completes the configured cooldown and resumes listening.
    pub fn complete_cooldown(&mut self) -> Result<(), CoreError> {
        if self.state != AssistantState::Cooldown {
            return Ok(());
        }
        self.transition(AssistantState::IdleListening)
    }

    /// Cooldown duration used by the runtime service and audio gate.
    pub fn cooldown(&self) -> Duration {
        self.cooldown
    }

    /// Replaces the complete adapter set after validated config reload.
    #[allow(clippy::too_many_arguments)]
    pub fn reconfigure(
        &mut self,
        recognizer: Arc<dyn SpeechRecognizer>,
        registry: CommandRegistry,
        executor: Arc<dyn CommandExecutor>,
        wake_word: String,
        policy: PolicyConfig,
        stt_timeout: Duration,
        cooldown: Duration,
    ) -> Result<(), CoreError> {
        self.reconfigure_partial(RuntimeUpdate {
            recognizer: Some(recognizer),
            commands: Some((registry, executor)),
            wake_word: Some(wake_word),
            policy: Some(policy),
            stt_timeout: Some(stt_timeout),
            cooldown: Some(cooldown),
        })
    }

    /// Replaces only supplied adapters and policy after validated config reload.
    pub fn reconfigure_partial(&mut self, update: RuntimeUpdate) -> Result<(), CoreError> {
        if self.state == AssistantState::AwaitingConfirmation {
            self.cancel_pending(ConfirmationCancelReason::Reconfigured)?;
            self.complete_cooldown()?;
        }
        if !matches!(
            self.state,
            AssistantState::Starting | AssistantState::IdleListening | AssistantState::Suspended
        ) {
            return Err(CoreError::Busy("configuration reload".into()));
        }
        if let Some(recognizer) = update.recognizer {
            self.recognizer = recognizer;
        }
        if let Some((registry, executor)) = update.commands {
            self.registry = registry;
            self.executor = executor;
        }
        if let Some(wake_word) = update.wake_word {
            self.wake_word = wake_word;
        }
        if let Some(policy) = update.policy {
            self.policy = policy;
        }
        if let Some(stt_timeout) = update.stt_timeout {
            self.stt_timeout = stt_timeout;
        }
        if let Some(cooldown) = update.cooldown {
            self.cooldown = cooldown;
        }
        Ok(())
    }

    /// Publishes a lifecycle event produced by model or audio services.
    pub fn publish_event(&self, event: AssistantEvent) {
        let _ = self.events.send(event);
    }

    fn cancel_pending(&mut self, reason: ConfirmationCancelReason) -> Result<(), CoreError> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| CoreError::Confirmation("no command is awaiting confirmation".into()))?;
        let _ = self.events.send(AssistantEvent::ConfirmationCancelled {
            command_id: pending.id,
            confirmation_id: pending.confirmation_id.unwrap_or_default(),
            reason,
        });
        self.enter_cooldown()
    }

    fn enter_cooldown(&mut self) -> Result<(), CoreError> {
        self.transition(AssistantState::Cooldown)
    }

    fn fail<T>(&mut self, component: &str, error: CoreError) -> Result<T, CoreError> {
        self.report_recoverable(component, error.to_string());
        Err(error)
    }
}

/// Deterministic recognizer for application and integration tests.
pub struct MockRecognizer {
    /// Text returned for every request.
    pub text: String,
}

/// Recognizer used while an application is installing or repairing its model.
pub struct UnavailableRecognizer {
    /// Diagnostic reason returned for every request.
    pub message: String,
}

#[async_trait]
impl SpeechRecognizer for UnavailableRecognizer {
    async fn transcribe(&self, _request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        Err(CoreError::Unavailable(self.message.clone()))
    }
}

#[async_trait]
impl SpeechRecognizer for MockRecognizer {
    async fn transcribe(&self, _request: TranscriptionRequest) -> Result<Transcript, CoreError> {
        Ok(Transcript {
            text: self.text.clone(),
            confidence: Some(1.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;

    use super::*;
    use crate::{
        config::CommandConfig,
        domain::{CommandResult, SpeechRecognizer},
    };

    struct CountingExecutor(Arc<AtomicUsize>);

    #[async_trait]
    impl CommandExecutor for CountingExecutor {
        async fn execute(
            &self,
            _: &str,
            _: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(CommandResult {
                message: "ok".into(),
            })
        }
    }

    struct FailingRecognizer;

    #[async_trait]
    impl SpeechRecognizer for FailingRecognizer {
        async fn transcribe(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            Err(CoreError::Recognition("broken".into()))
        }
    }

    struct SlowRecognizer;

    #[async_trait]
    impl SpeechRecognizer for SlowRecognizer {
        async fn transcribe(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(Transcript {
                text: "late".into(),
                confidence: None,
            })
        }
    }

    struct SlowExecutor;

    #[async_trait]
    impl CommandExecutor for SlowExecutor {
        async fn execute(
            &self,
            _: &str,
            _: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(CommandResult {
                message: "late".into(),
            })
        }
    }

    fn command(risk: RiskLevel) -> CommandConfig {
        CommandConfig {
            id: "test".into(),
            enabled: true,
            phrases: vec!["открой блокнот".into()],
            handler: "launch_app".into(),
            risk,
            requires_confirmation: false,
            timeout_ms: 1_000,
            parameters: BTreeMap::from([("executable".into(), "notepad.exe".into())]),
        }
    }

    fn request() -> TranscriptionRequest {
        TranscriptionRequest {
            samples: vec![0.0],
            sample_rate: 16_000,
        }
    }

    #[tokio::test]
    async fn high_risk_command_can_run_with_confirmations_disabled() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = Runtime::new(
            Arc::new(MockRecognizer {
                text: "ассистент открой блокнот".into(),
            }),
            CommandRegistry::new(vec![command(RiskLevel::High)], true),
            Arc::new(CountingExecutor(Arc::clone(&calls))),
            "ассистент".into(),
            PolicyConfig {
                confirmations_enabled: false,
                ..PolicyConfig::default()
            },
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.process_command_audio(request()).await.unwrap();
        assert_eq!(runtime.state(), AssistantState::Cooldown);
        runtime.complete_cooldown().unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn submitted_text_uses_matching_and_command_policy_without_stt() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::clone(&calls))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.process_text("открой блокнот".into()).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.state(), AssistantState::Cooldown);
    }

    #[tokio::test]
    async fn unavailable_model_emits_transcript_reason() {
        let mut runtime = Runtime::new(
            Arc::new(UnavailableRecognizer {
                message: "model missing".into(),
            }),
            CommandRegistry::new(Vec::new(), true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "assistant".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        let mut events = runtime.subscribe();
        runtime.start().unwrap();
        assert!(runtime.process_command_audio(request()).await.is_err());
        assert!(
            std::iter::from_fn(|| events.try_recv().ok()).any(|event| matches!(
                event,
                AssistantEvent::TranscriptUnavailable {
                    reason: crate::TranscriptUnavailableReason::ModelUnavailable
                }
            ))
        );
    }

    #[tokio::test]
    async fn empty_recognizer_output_is_reported_as_silence() {
        let mut runtime = Runtime::new(
            Arc::new(MockRecognizer { text: "  ".into() }),
            CommandRegistry::new(Vec::new(), true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "assistant".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        let mut events = runtime.subscribe();
        runtime.start().unwrap();
        runtime.process_command_audio(request()).await.unwrap();
        assert!(
            std::iter::from_fn(|| events.try_recv().ok()).any(|event| matches!(
                event,
                AssistantEvent::TranscriptUnavailable {
                    reason: crate::TranscriptUnavailableReason::Silence
                }
            ))
        );
    }

    #[test]
    fn manual_capture_can_return_to_idle_without_transcription() {
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(Vec::new(), true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.start_manual_capture().unwrap();
        assert_eq!(runtime.state(), AssistantState::CapturingCommand);
        runtime.cancel_capture().unwrap();
        assert_eq!(runtime.state(), AssistantState::IdleListening);
    }

    #[tokio::test]
    async fn confirmation_waits_for_matching_command_id() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = Runtime::new(
            Arc::new(MockRecognizer {
                text: "открой блокнот".into(),
            }),
            CommandRegistry::new(vec![command(RiskLevel::High)], true),
            Arc::new(CountingExecutor(Arc::clone(&calls))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.process_command_audio(request()).await.unwrap();
        assert_eq!(runtime.state(), AssistantState::AwaitingConfirmation);
        assert!(runtime.confirm_command("wrong").await.is_err());
        let confirmation_id = runtime.pending_confirmation_id().unwrap().to_string();
        runtime.confirm_command(&confirmation_id).await.unwrap();
        assert_eq!(runtime.state(), AssistantState::Cooldown);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn recognition_failure_recovers_to_idle() {
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        assert!(runtime.process_command_audio(request()).await.is_err());
        assert_eq!(runtime.state(), AssistantState::IdleListening);
    }

    #[tokio::test]
    async fn handler_timeout_recovers_to_idle() {
        let mut timed = command(RiskLevel::Low);
        timed.timeout_ms = 1;
        let metrics = CoreMetrics::default();
        let mut runtime = Runtime::new(
            Arc::new(MockRecognizer {
                text: "открой блокнот".into(),
            }),
            CommandRegistry::new(vec![timed], true),
            Arc::new(SlowExecutor),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            metrics.clone(),
        );
        runtime.start().unwrap();
        assert!(runtime.process_command_audio(request()).await.is_err());
        assert_eq!(runtime.state(), AssistantState::IdleListening);
        assert_eq!(metrics.snapshot().recoverable_errors, 1);
    }

    #[tokio::test]
    async fn stt_timeout_recovers_to_idle() {
        let mut runtime = Runtime::new(
            Arc::new(SlowRecognizer),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_millis(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        assert!(runtime.process_command_audio(request()).await.is_err());
        assert_eq!(runtime.state(), AssistantState::IdleListening);
    }
}
