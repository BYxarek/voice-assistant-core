use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use tokio::sync::{broadcast, watch};

use crate::{
    commands::CommandRegistry,
    config::{PolicyConfig, RiskLevel},
    domain::{
        AssistantEvent, AssistantState, CommandExecutor, CommandParameters, CommandResult,
        ConfirmationCancelReason, CoreError, SpeechRecognizer, Transcript, TranscriptionRequest,
    },
    metrics::CoreMetrics,
    service::RuntimeUpdate,
};

struct PendingCommand {
    id: String,
    actions: Vec<PendingAction>,
    timeout: Duration,
    expires_at: Instant,
    confirmation_id: Option<String>,
}

struct PendingAction {
    handler: String,
    parameters: CommandParameters,
    delay: Duration,
}

#[derive(Default)]
struct HandlerGuard {
    last_started: Option<Instant>,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
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
    wake_words: Vec<String>,
    policy: PolicyConfig,
    stt_timeout: Duration,
    cooldown: Duration,
    pending: Option<PendingCommand>,
    confirmation_sequence: u64,
    streaming_stt: bool,
    confirming_by_voice: bool,
    last_partial: String,
    session_sequence: u64,
    active_session_id: Option<u64>,
    handler_guards: HashMap<String, HandlerGuard>,
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
        Self::new_with_wake_words(
            recognizer,
            registry,
            executor,
            vec![wake_word],
            policy,
            stt_timeout,
            cooldown,
            metrics,
        )
    }

    /// Creates a runtime accepting the canonical wake word and all configured aliases.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_wake_words(
        recognizer: Arc<dyn SpeechRecognizer>,
        registry: CommandRegistry,
        executor: Arc<dyn CommandExecutor>,
        wake_words: Vec<String>,
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
            wake_words,
            policy,
            stt_timeout,
            cooldown,
            pending: None,
            confirmation_sequence: 0,
            streaming_stt: false,
            confirming_by_voice: false,
            last_partial: String::new(),
            session_sequence: 0,
            active_session_id: None,
            handler_guards: HashMap::new(),
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
        if self.state == AssistantState::AwaitingConfirmation
            && self.policy.voice_confirmations_enabled
        {
            self.confirming_by_voice = true;
            self.transition(AssistantState::CapturingCommand)?;
            return self.start_speech_session();
        }
        self.transition(AssistantState::SpeechDetected)?;
        self.transition(AssistantState::WakeWordDetected)?;
        self.transition(AssistantState::CapturingCommand)?;
        self.start_speech_session()
    }

    /// Starts a VAD-delimited speech session that may or may not authorize a command.
    pub fn start_speech_capture(&mut self, wake_word_detected: bool) -> Result<(), CoreError> {
        if self.state == AssistantState::AwaitingConfirmation
            && self.policy.voice_confirmations_enabled
        {
            self.confirming_by_voice = true;
            self.transition(AssistantState::CapturingCommand)?;
            return self.start_speech_session();
        }
        self.transition(AssistantState::SpeechDetected)?;
        if wake_word_detected {
            self.transition(AssistantState::WakeWordDetected)?;
        }
        self.transition(AssistantState::CapturingCommand)?;
        self.start_speech_session()
    }

    /// Starts capture requested by a trusted application without a wake word.
    pub fn start_manual_capture(&mut self) -> Result<(), CoreError> {
        if self.state == AssistantState::AwaitingConfirmation {
            if !self.policy.voice_confirmations_enabled {
                return Err(CoreError::Busy("voice confirmation is disabled".into()));
            }
            self.confirming_by_voice = true;
        }
        self.transition(AssistantState::CapturingCommand)?;
        self.start_speech_session()
    }

    /// Returns an empty or too-short manual capture to idle listening.
    pub fn cancel_capture(&mut self) -> Result<(), CoreError> {
        self.streaming_stt = false;
        self.last_partial.clear();
        self.end_speech_session();
        if self.confirming_by_voice {
            self.confirming_by_voice = false;
            return self.transition(AssistantState::AwaitingConfirmation);
        }
        self.transition(AssistantState::IdleListening)
    }

    /// Starts incremental STT as soon as command capture begins.
    pub async fn begin_transcription_stream(&mut self, sample_rate: u32) -> Result<(), CoreError> {
        if self.active_session_id.is_none() {
            return Err(CoreError::Busy("no speech session is active".into()));
        }
        self.last_partial.clear();
        self.streaming_stt =
            match tokio::time::timeout(self.stt_timeout, self.recognizer.begin_stream(sample_rate))
                .await
            {
                Ok(result) => result?,
                Err(_) => {
                    let _ = self.recognizer.recover().await;
                    return Err(CoreError::Recognition(
                        "STT stream startup timed out".into(),
                    ));
                }
            };
        Ok(())
    }

    /// Forwards one command-audio frame to an active incremental STT session.
    pub async fn push_transcription_stream(&mut self, samples: Vec<f32>) -> Result<(), CoreError> {
        if self.streaming_stt {
            match tokio::time::timeout(
                self.stt_timeout,
                self.recognizer.push_stream_partial(samples),
            )
            .await
            {
                Ok(Ok(Some(partial)))
                    if !partial.text.trim().is_empty() && partial.text != self.last_partial =>
                {
                    self.last_partial.clone_from(&partial.text);
                    let _ = self.events.send(AssistantEvent::TranscriptPartial {
                        session_id: self.active_session_id.unwrap_or_default(),
                        text: partial.text,
                        confidence: partial.confidence,
                    });
                }
                Ok(Ok(_)) => {}
                Ok(Err(error)) => {
                    self.streaming_stt = false;
                    return Err(error);
                }
                Err(_) => {
                    self.streaming_stt = false;
                    let _ = self.recognizer.recover().await;
                    return Err(CoreError::Recognition("STT stream timed out".into()));
                }
            }
        }
        Ok(())
    }

    /// Matches application-supplied text through the normal command policy.
    pub async fn process_text(&mut self, text: String) -> Result<(), CoreError> {
        if text.trim().is_empty() || text.len() > MAX_SUBMITTED_TEXT_BYTES {
            return Err(CoreError::Command(format!(
                "submitted text must contain 1..={MAX_SUBMITTED_TEXT_BYTES} UTF-8 bytes"
            )));
        }
        if self.state == AssistantState::AwaitingConfirmation
            && self.policy.voice_confirmations_enabled
        {
            return self.process_confirmation_text(text).await;
        }
        self.transition(AssistantState::MatchingCommand)?;
        self.match_text(text, false).await
    }

    /// Transcribes audio after an already published capture phase.
    pub async fn process_captured_audio(
        &mut self,
        request: TranscriptionRequest,
    ) -> Result<(), CoreError> {
        self.process_captured_speech(request, true).await
    }

    /// Finalizes STT and only matches commands for wake-word-authorized speech.
    pub async fn process_captured_speech(
        &mut self,
        request: TranscriptionRequest,
        command_authorized: bool,
    ) -> Result<(), CoreError> {
        let session_id = self
            .active_session_id
            .take()
            .ok_or_else(|| CoreError::Busy("no speech session is active".into()))?;
        let _ = self.events.send(AssistantEvent::SpeechEnded { session_id });
        self.transition(AssistantState::Transcribing)?;

        let started = Instant::now();
        let outcome = if self.streaming_stt {
            self.streaming_stt = false;
            tokio::time::timeout(self.stt_timeout, self.recognizer.finish_stream(request)).await
        } else {
            tokio::time::timeout(self.stt_timeout, self.recognizer.transcribe(request)).await
        };
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
            Err(_) => {
                let recovery = self.recognizer.recover().await;
                let message = match recovery {
                    Ok(true) => "STT timed out; worker restarted",
                    Ok(false) => "STT timed out",
                    Err(_) => "STT timed out; worker restart budget exhausted",
                };
                return self.fail("stt", CoreError::Recognition(message.into()));
            }
        };
        self.last_partial.clear();
        if transcript.text.trim().is_empty() {
            let _ = self.events.send(AssistantEvent::TranscriptUnavailable {
                reason: crate::TranscriptUnavailableReason::Silence,
            });
            if self.confirming_by_voice {
                self.confirming_by_voice = false;
                self.transition(AssistantState::AwaitingConfirmation)?;
            } else if !command_authorized {
                self.transition(AssistantState::IdleListening)?;
            } else {
                self.transition(AssistantState::MatchingCommand)?;
                self.enter_cooldown()?;
            }
            return Ok(());
        }
        let _ = self.events.send(AssistantEvent::TranscriptFinal {
            session_id,
            text: transcript.text.clone(),
            confidence: transcript.confidence,
        });
        if self.confirming_by_voice {
            self.confirming_by_voice = false;
            return self.process_confirmation_text(transcript.text).await;
        }
        if !command_authorized {
            self.transition(AssistantState::IdleListening)?;
            return Ok(());
        }
        self.transition(AssistantState::MatchingCommand)?;
        self.match_text(transcript.text, true).await
    }

    fn start_speech_session(&mut self) -> Result<(), CoreError> {
        if self.active_session_id.is_some() {
            return Err(CoreError::Busy("speech session is already active".into()));
        }
        self.session_sequence = self.session_sequence.wrapping_add(1);
        let session_id = self.session_sequence;
        self.active_session_id = Some(session_id);
        let _ = self
            .events
            .send(AssistantEvent::SpeechStarted { session_id });
        Ok(())
    }

    fn end_speech_session(&mut self) {
        if let Some(session_id) = self.active_session_id.take() {
            let _ = self.events.send(AssistantEvent::SpeechEnded { session_id });
        }
    }

    async fn process_confirmation_text(&mut self, text: String) -> Result<(), CoreError> {
        let text = self.registry.normalized(&text);
        let matches = |phrases: &[String]| {
            phrases
                .iter()
                .any(|phrase| self.registry.normalized(phrase) == text)
        };
        let accepted = matches(&self.policy.confirmation_accept_phrases);
        let cancelled = matches(&self.policy.confirmation_cancel_phrases);
        if self.state != AssistantState::AwaitingConfirmation {
            self.transition(AssistantState::AwaitingConfirmation)?;
        }
        if accepted {
            let pending = self.pending.take().ok_or_else(|| {
                CoreError::Confirmation("no command is awaiting confirmation".into())
            })?;
            if Instant::now() >= pending.expires_at {
                self.pending = Some(pending);
                self.cancel_pending(ConfirmationCancelReason::Expired)?;
                return Err(CoreError::Confirmation("confirmation expired".into()));
            }
            return self.execute(pending).await;
        }
        if cancelled {
            return self.cancel_pending(ConfirmationCancelReason::User);
        }
        let _ = self
            .events
            .send(AssistantEvent::ConfirmationUnrecognized { text });
        Ok(())
    }

    async fn match_text(&mut self, text: String, allow_wake_word: bool) -> Result<(), CoreError> {
        let matched = if allow_wake_word {
            self.registry
                .match_after_wake_words(&text, self.wake_words.iter().map(String::as_str))
        } else {
            self.registry.match_text(&text)
        };
        let Some(matched) = matched else {
            let _ = self.events.send(AssistantEvent::CommandNotMatched { text });
            self.enter_cooldown()?;
            return Ok(());
        };

        let command = matched.command;
        let slots = matched.slots;
        let id = command.id.clone();
        let actions = command
            .resolved_actions()
            .into_iter()
            .map(|action| {
                let mut parameters = action.parameters;
                for value in parameters.values_mut() {
                    if let Some(name) = value
                        .strip_prefix('{')
                        .and_then(|value| value.strip_suffix('}'))
                        && let Some(slot) = slots.get(name)
                    {
                        value.clone_from(slot);
                    }
                }
                parameters.insert("_command_id".into(), id.clone());
                PendingAction {
                    handler: action.handler,
                    parameters,
                    delay: Duration::from_millis(action.delay_ms),
                }
            })
            .collect();
        let pending = PendingCommand {
            id: id.clone(),
            actions,
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
        let mut current_handler = None;
        let outcome = tokio::time::timeout(command.timeout, async {
            let action_count = command.actions.len();
            let mut last_result = None;
            for action in &command.actions {
                if !action.delay.is_zero() {
                    tokio::time::sleep(action.delay).await;
                }
                self.check_handler(&action.handler)?;
                current_handler = Some(action.handler.clone());
                match self
                    .executor
                    .execute(&action.handler, &action.parameters)
                    .await
                {
                    Ok(result) => {
                        self.record_handler_success(&action.handler);
                        last_result = Some(result);
                    }
                    Err(error) => {
                        self.record_handler_failure(&action.handler);
                        return Err(error);
                    }
                }
            }
            Ok::<_, CoreError>(if action_count == 1 {
                last_result.ok_or_else(|| CoreError::Command("command has no actions".into()))?
            } else {
                CommandResult {
                    message: format!("completed {action_count} actions"),
                }
            })
        })
        .await;
        self.metrics
            .observe_handler(started.elapsed().as_millis() as u64);
        let result = match outcome {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => return self.fail("command", error),
            Err(_) => {
                if let Some(handler) = current_handler {
                    self.record_handler_failure(&handler);
                }
                return self.fail("command", CoreError::Command("command timed out".into()));
            }
        };
        self.metrics.command_completed();
        let _ = self.events.send(AssistantEvent::CommandFinished {
            command_id: command.id,
            result,
        });
        self.enter_cooldown()
    }

    fn check_handler(&mut self, handler: &str) -> Result<(), CoreError> {
        let now = Instant::now();
        let guard = self.handler_guards.entry(handler.into()).or_default();
        if let Some(opened_at) = guard.opened_at {
            let reset = Duration::from_millis(self.policy.handler_circuit_breaker_ms);
            if now.duration_since(opened_at) < reset {
                return Err(CoreError::Command(format!(
                    "handler circuit is open: {handler}"
                )));
            }
            guard.opened_at = None;
        }
        let rate_limit = Duration::from_millis(self.policy.handler_rate_limit_ms);
        if !rate_limit.is_zero()
            && guard
                .last_started
                .is_some_and(|started| now.duration_since(started) < rate_limit)
        {
            return Err(CoreError::Command(format!(
                "handler rate limit exceeded: {handler}"
            )));
        }
        guard.last_started = Some(now);
        Ok(())
    }

    fn record_handler_success(&mut self, handler: &str) {
        let guard = self.handler_guards.entry(handler.into()).or_default();
        guard.consecutive_failures = 0;
        guard.opened_at = None;
    }

    fn record_handler_failure(&mut self, handler: &str) {
        if self.policy.handler_failure_threshold == 0 {
            return;
        }
        let guard = self.handler_guards.entry(handler.into()).or_default();
        guard.consecutive_failures = guard.consecutive_failures.saturating_add(1);
        if guard.consecutive_failures >= self.policy.handler_failure_threshold {
            guard.opened_at = Some(Instant::now());
        }
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
        wake_words: Vec<String>,
        policy: PolicyConfig,
        stt_timeout: Duration,
        cooldown: Duration,
    ) -> Result<(), CoreError> {
        self.reconfigure_partial(RuntimeUpdate {
            recognizer: Some(recognizer),
            commands: Some((registry, executor)),
            wake_words: Some(wake_words),
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
            self.handler_guards.clear();
        }
        if let Some(wake_words) = update.wake_words {
            self.wake_words = wake_words;
        }
        if let Some(policy) = update.policy {
            self.policy = policy;
            self.handler_guards.clear();
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
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;

    use super::*;
    use crate::{
        config::{CommandActionConfig, CommandConfig},
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

    struct RecordingExecutor(Arc<Mutex<Vec<String>>>);

    #[async_trait]
    impl CommandExecutor for RecordingExecutor {
        async fn execute(
            &self,
            handler: &str,
            _: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            self.0.lock().unwrap().push(handler.into());
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

    struct StreamingRecognizer(Arc<AtomicUsize>);

    #[async_trait]
    impl SpeechRecognizer for StreamingRecognizer {
        async fn transcribe(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            panic!("batch fallback must not run for a healthy stream")
        }

        async fn begin_stream(&self, _: u32) -> Result<bool, CoreError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        }

        async fn push_stream(&self, _: Vec<f32>) -> Result<(), CoreError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        async fn finish_stream(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Transcript {
                text: "открой блокнот".into(),
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

    struct FailingExecutor(Arc<AtomicUsize>);

    #[async_trait]
    impl CommandExecutor for FailingExecutor {
        async fn execute(
            &self,
            _: &str,
            _: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Err(CoreError::Command("failed".into()))
        }
    }

    struct ConfirmationRecognizer(AtomicUsize);

    #[async_trait]
    impl SpeechRecognizer for ConfirmationRecognizer {
        async fn transcribe(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            let text = if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                "открой блокнот"
            } else {
                "да"
            };
            Ok(Transcript {
                text: text.into(),
                confidence: None,
            })
        }
    }

    struct PartialRecognizer;

    #[async_trait]
    impl SpeechRecognizer for PartialRecognizer {
        async fn transcribe(&self, _: TranscriptionRequest) -> Result<Transcript, CoreError> {
            unreachable!()
        }

        async fn begin_stream(&self, _: u32) -> Result<bool, CoreError> {
            Ok(true)
        }

        async fn push_stream_partial(&self, _: Vec<f32>) -> Result<Option<Transcript>, CoreError> {
            Ok(Some(Transcript {
                text: "открой".into(),
                confidence: None,
            }))
        }
    }

    fn command(risk: RiskLevel) -> CommandConfig {
        CommandConfig {
            id: "test".into(),
            enabled: true,
            phrases: vec!["открой блокнот".into()],
            slots: BTreeMap::new(),
            handler: "launch_app".into(),
            risk,
            requires_confirmation: false,
            timeout_ms: 1_000,
            parameters: BTreeMap::from([("executable".into(), "notepad.exe".into())]),
            actions: Vec::new(),
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
    async fn non_wake_speech_is_transcribed_without_executing() {
        let calls = Arc::new(AtomicUsize::new(0));
        let phrase = command(RiskLevel::Low).phrases[0].clone();
        let mut runtime = Runtime::new(
            Arc::new(MockRecognizer { text: phrase }),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::clone(&calls))),
            "assistant".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        let mut events = runtime.subscribe();
        runtime.start().unwrap();
        runtime.start_speech_capture(false).unwrap();
        runtime
            .process_captured_speech(request(), false)
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.state(), AssistantState::IdleListening);
        let session_events: Vec<_> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                AssistantEvent::SpeechStarted { session_id } => Some(("start", session_id)),
                AssistantEvent::SpeechEnded { session_id } => Some(("end", session_id)),
                AssistantEvent::TranscriptFinal { session_id, .. } => Some(("final", session_id)),
                _ => None,
            })
            .collect();
        assert_eq!(session_events, [("start", 1), ("end", 1), ("final", 1)]);
    }

    #[tokio::test]
    async fn command_actions_execute_in_order_with_delays() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut sequenced = command(RiskLevel::Low);
        sequenced.handler.clear();
        sequenced.parameters.clear();
        sequenced.actions = vec![
            CommandActionConfig {
                handler: "first".into(),
                delay_ms: 0,
                parameters: BTreeMap::new(),
            },
            CommandActionConfig {
                handler: "second".into(),
                delay_ms: 20,
                parameters: BTreeMap::new(),
            },
        ];
        let phrase = sequenced.phrases[0].clone();
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(vec![sequenced], true),
            Arc::new(RecordingExecutor(Arc::clone(&calls))),
            "assistant".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        let started = Instant::now();
        runtime.process_text(phrase).await.unwrap();
        assert_eq!(*calls.lock().unwrap(), ["first", "second"]);
        assert!(started.elapsed() >= Duration::from_millis(20));
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
    async fn pending_command_can_be_confirmed_by_voice() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = Runtime::new(
            Arc::new(ConfirmationRecognizer(AtomicUsize::new(0))),
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
        runtime.start_command_capture().unwrap();
        runtime.process_captured_audio(request()).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(runtime.state(), AssistantState::Cooldown);
    }

    #[tokio::test]
    async fn changed_partial_transcript_is_published_once() {
        let mut runtime = Runtime::new(
            Arc::new(PartialRecognizer),
            CommandRegistry::new(Vec::new(), true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        let mut events = runtime.subscribe();
        runtime.start().unwrap();
        runtime.start_command_capture().unwrap();
        runtime.begin_transcription_stream(16_000).await.unwrap();
        runtime.push_transcription_stream(vec![0.1]).await.unwrap();
        runtime.push_transcription_stream(vec![0.1]).await.unwrap();
        let partials: Vec<_> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|event| match event {
                AssistantEvent::TranscriptPartial { session_id, .. } => Some(session_id),
                _ => None,
            })
            .collect();
        assert_eq!(partials, [1]);
    }

    #[tokio::test]
    async fn handler_rate_limit_rejects_repeated_start() {
        let calls = Arc::new(AtomicUsize::new(0));
        let policy = PolicyConfig {
            handler_rate_limit_ms: 60_000,
            ..PolicyConfig::default()
        };
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::clone(&calls))),
            "ассистент".into(),
            policy,
            Duration::from_secs(1),
            Duration::from_millis(1),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.process_text("открой блокнот".into()).await.unwrap();
        runtime.complete_cooldown().unwrap();
        assert!(runtime.process_text("открой блокнот".into()).await.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn handler_circuit_opens_after_consecutive_failures() {
        let calls = Arc::new(AtomicUsize::new(0));
        let policy = PolicyConfig {
            handler_failure_threshold: 2,
            ..PolicyConfig::default()
        };
        let mut runtime = Runtime::new(
            Arc::new(FailingRecognizer),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(FailingExecutor(Arc::clone(&calls))),
            "ассистент".into(),
            policy,
            Duration::from_secs(1),
            Duration::from_millis(1),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        for _ in 0..3 {
            assert!(runtime.process_text("открой блокнот".into()).await.is_err());
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
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
    async fn streaming_stt_starts_during_capture_and_finishes_without_batch_fallback() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut runtime = Runtime::new(
            Arc::new(StreamingRecognizer(Arc::clone(&calls))),
            CommandRegistry::new(vec![command(RiskLevel::Low)], true),
            Arc::new(CountingExecutor(Arc::new(AtomicUsize::new(0)))),
            "ассистент".into(),
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        runtime.start_command_capture().unwrap();
        runtime.begin_transcription_stream(16_000).await.unwrap();
        runtime
            .push_transcription_stream(vec![0.1; 320])
            .await
            .unwrap();
        runtime.process_captured_audio(request()).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3);
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
