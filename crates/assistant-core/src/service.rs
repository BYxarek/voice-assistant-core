use std::{sync::Arc, time::Duration};

use tokio::{
    sync::{broadcast, mpsc, oneshot, watch},
    task::{AbortHandle, JoinHandle},
};

use crate::{
    commands::CommandRegistry,
    config::PolicyConfig,
    domain::{
        AssistantEvent, AssistantState, CommandExecutor, CoreError, SpeechRecognizer,
        TranscriptionRequest,
    },
    runtime::Runtime,
};

/// Complete validated adapter set accepted during runtime reload.
pub struct RuntimeComponents {
    /// Replaceable STT adapter.
    pub recognizer: Arc<dyn SpeechRecognizer>,
    /// Exact command matcher.
    pub commands: CommandRegistry,
    /// Typed handler registry or compatible executor.
    pub executor: Arc<dyn CommandExecutor>,
    /// Prefix removed before exact command matching.
    pub wake_words: Vec<String>,
    /// Confirmation behavior.
    pub policy: PolicyConfig,
    /// Maximum duration of one recognizer request.
    pub stt_timeout: Duration,
    /// Delay before returning from `Cooldown` to listening.
    pub cooldown: Duration,
}

/// Differential runtime update; omitted fields keep their active component.
#[derive(Default)]
pub struct RuntimeUpdate {
    /// Replacement STT adapter.
    pub recognizer: Option<Arc<dyn SpeechRecognizer>>,
    /// Replacement command matcher and typed executor.
    pub commands: Option<(CommandRegistry, Arc<dyn CommandExecutor>)>,
    /// Replacement prefix removed before exact command matching.
    pub wake_words: Option<Vec<String>>,
    /// Replacement confirmation behavior.
    pub policy: Option<PolicyConfig>,
    /// Replacement maximum duration of one recognizer request.
    pub stt_timeout: Option<Duration>,
    /// Replacement delay before returning from `Cooldown`.
    pub cooldown: Option<Duration>,
}

enum Control {
    Start(oneshot::Sender<Result<(), CoreError>>),
    BeginCapture(oneshot::Sender<Result<(), CoreError>>),
    BeginManualCapture(oneshot::Sender<Result<(), CoreError>>),
    CancelCapture(oneshot::Sender<Result<(), CoreError>>),
    BeginStream(u32, oneshot::Sender<Result<(), CoreError>>),
    PushStream(Vec<f32>, oneshot::Sender<Result<(), CoreError>>),
    Captured(TranscriptionRequest, oneshot::Sender<Result<(), CoreError>>),
    SubmitText(String, oneshot::Sender<Result<(), CoreError>>),
    Suspend(oneshot::Sender<Result<(), CoreError>>),
    Resume(oneshot::Sender<Result<(), CoreError>>),
    Confirm(String, oneshot::Sender<Result<(), CoreError>>),
    Cancel(String, oneshot::Sender<Result<(), CoreError>>),
    Reconfigure(RuntimeUpdate, oneshot::Sender<Result<(), CoreError>>),
    Recover(String, String),
    Publish(AssistantEvent),
    Shutdown(oneshot::Sender<Result<(), CoreError>>),
}

/// Cloneable non-blocking application-facing handle to the runtime task.
#[derive(Clone)]
pub struct RuntimeHandle {
    controls: mpsc::Sender<Control>,
    state: watch::Receiver<AssistantState>,
    events: broadcast::Sender<AssistantEvent>,
    shutdown: watch::Sender<bool>,
}

impl RuntimeHandle {
    /// Completes startup after required adapters have been initialized.
    pub async fn start(&self) -> Result<(), CoreError> {
        self.call(Control::Start).await
    }

    /// Returns the latest state without locking or queueing behind inference.
    pub fn state(&self) -> AssistantState {
        *self.state.borrow()
    }

    /// Watches state changes while retaining the latest value.
    pub fn watch_state(&self) -> watch::Receiver<AssistantState> {
        self.state.clone()
    }

    /// Subscribes to the ordered event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AssistantEvent> {
        self.events.subscribe()
    }

    /// Returns the event sender required by transports that create subscriptions.
    pub fn event_sender(&self) -> broadcast::Sender<AssistantEvent> {
        self.events.clone()
    }

    /// Returns the shutdown signal used by the daemon lifecycle.
    pub fn shutdown_receiver(&self) -> watch::Receiver<bool> {
        self.shutdown.subscribe()
    }

    /// Publishes wake-word and capture states immediately.
    pub async fn begin_capture(&self) -> Result<(), CoreError> {
        self.call(Control::BeginCapture).await
    }

    /// Publishes capture state for an application-requested recording.
    pub async fn begin_manual_capture(&self) -> Result<(), CoreError> {
        self.call(Control::BeginManualCapture).await
    }

    /// Cancels a manual recording that did not produce enough audio.
    pub async fn cancel_capture(&self) -> Result<(), CoreError> {
        self.call(Control::CancelCapture).await
    }

    /// Starts optional incremental STT for an active capture.
    pub async fn begin_transcription_stream(&self, sample_rate: u32) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::BeginStream(sample_rate, response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Queues one incremental command-audio chunk.
    pub async fn push_transcription_stream(&self, samples: Vec<f32>) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::PushStream(samples, response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Queues completed command audio for STT and matching.
    pub async fn captured_audio(&self, request: TranscriptionRequest) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::Captured(request, response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Matches validated application text through command policy without STT.
    pub async fn submit_text(&self, text: impl Into<String>) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::SubmitText(text.into(), response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Suspends listening after any current bounded operation.
    pub async fn suspend(&self) -> Result<(), CoreError> {
        self.call(Control::Suspend).await
    }

    /// Resumes listening.
    pub async fn resume(&self) -> Result<(), CoreError> {
        self.call(Control::Resume).await
    }

    /// Confirms exactly one pending invocation token.
    pub async fn confirm(&self, confirmation_id: impl Into<String>) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::Confirm(confirmation_id.into(), response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Cancels exactly one pending invocation token.
    pub async fn cancel(&self, confirmation_id: impl Into<String>) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::Cancel(confirmation_id.into(), response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Atomically swaps a complete validated runtime adapter set.
    pub async fn reconfigure(&self, components: RuntimeComponents) -> Result<(), CoreError> {
        self.reconfigure_partial(RuntimeUpdate {
            recognizer: Some(components.recognizer),
            commands: Some((components.commands, components.executor)),
            wake_words: Some(components.wake_words),
            policy: Some(components.policy),
            stt_timeout: Some(components.stt_timeout),
            cooldown: Some(components.cooldown),
        })
        .await
    }

    /// Atomically swaps only the supplied validated runtime components.
    pub async fn reconfigure_partial(&self, update: RuntimeUpdate) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(Control::Reconfigure(update, response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }

    /// Reports a recoverable adapter error without blocking its producer.
    pub fn report_recoverable(&self, component: impl Into<String>, message: impl Into<String>) {
        let _ = self
            .controls
            .try_send(Control::Recover(component.into(), message.into()));
    }

    /// Publishes a model or application lifecycle event.
    pub fn publish_event(&self, event: AssistantEvent) {
        let _ = self.controls.try_send(Control::Publish(event));
    }

    /// Requests graceful shutdown and immediately wakes daemon lifecycle code.
    pub async fn shutdown(&self) -> Result<(), CoreError> {
        let result = self.call(Control::Shutdown).await;
        if result.is_ok() {
            self.shutdown.send_replace(true);
        }
        result
    }

    /// Wakes process lifecycle code when a bounded runtime operation must be aborted.
    pub fn signal_shutdown(&self) {
        self.shutdown.send_replace(true);
    }

    async fn call(
        &self,
        constructor: impl FnOnce(oneshot::Sender<Result<(), CoreError>>) -> Control,
    ) -> Result<(), CoreError> {
        let (response, receiver) = oneshot::channel();
        self.controls
            .send(constructor(response))
            .await
            .map_err(|_| stopped())?;
        receiver.await.map_err(|_| stopped())?
    }
}

/// Owned task token for deterministic shutdown and tests.
pub struct RuntimeTask {
    task: JoinHandle<()>,
}

impl RuntimeTask {
    /// Returns a cancellation handle for process shutdown paths.
    pub fn abort_handle(&self) -> AbortHandle {
        self.task.abort_handle()
    }

    /// Waits until the runtime task exits.
    pub async fn join(self) -> Result<(), tokio::task::JoinError> {
        self.task.await
    }
}

/// Starts a dedicated task that exclusively owns mutable runtime state.
pub fn spawn_runtime_service(
    mut runtime: Runtime,
    capacity: usize,
) -> Result<(RuntimeHandle, RuntimeTask), CoreError> {
    if capacity == 0 {
        return Err(CoreError::Unavailable(
            "runtime control queue capacity must be greater than zero".into(),
        ));
    }
    let (controls, mut receiver) = mpsc::channel(capacity);
    let state = runtime.state_receiver();
    let events = runtime.event_sender();
    let (shutdown, mut shutdown_receiver) = watch::channel(false);
    let handle = RuntimeHandle {
        controls,
        state,
        events,
        shutdown,
    };
    let task = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_millis(100));
        let mut cooldown_deadline = None;
        loop {
            tokio::select! {
                changed = shutdown_receiver.changed() => {
                    if changed.is_err() || *shutdown_receiver.borrow() {
                        let _ = runtime.shutdown();
                        break;
                    }
                }
                _ = tick.tick() => {
                    let _ = runtime.expire_confirmation();
                    if runtime.state() == AssistantState::Cooldown {
                        let deadline = cooldown_deadline
                            .get_or_insert_with(|| tokio::time::Instant::now() + runtime.cooldown());
                        if tokio::time::Instant::now() >= *deadline {
                            let _ = runtime.complete_cooldown();
                            cooldown_deadline = None;
                        }
                    } else {
                        cooldown_deadline = None;
                    }
                }
                control = receiver.recv() => {
                    let Some(control) = control else {
                        let _ = runtime.shutdown();
                        break;
                    };
                    match control {
                        Control::Start(response) => {
                            let _ = response.send(runtime.start());
                        }
                        Control::BeginCapture(response) => {
                            let _ = response.send(runtime.start_command_capture());
                        }
                        Control::BeginManualCapture(response) => {
                            let _ = response.send(runtime.start_manual_capture());
                        }
                        Control::CancelCapture(response) => {
                            let _ = response.send(runtime.cancel_capture());
                        }
                        Control::BeginStream(sample_rate, response) => {
                            let _ = response.send(runtime.begin_transcription_stream(sample_rate).await);
                        }
                        Control::PushStream(samples, response) => {
                            let _ = response.send(runtime.push_transcription_stream(samples).await);
                        }
                        Control::Captured(request, response) => {
                            let _ = response.send(runtime.process_captured_audio(request).await);
                        }
                        Control::SubmitText(text, response) => {
                            let _ = response.send(runtime.process_text(text).await);
                        }
                        Control::Suspend(response) => {
                            let _ = response.send(runtime.suspend());
                        }
                        Control::Resume(response) => {
                            let _ = response.send(runtime.resume());
                        }
                        Control::Confirm(id, response) => {
                            let _ = response.send(runtime.confirm_command(&id).await);
                        }
                        Control::Cancel(id, response) => {
                            let _ = response.send(runtime.cancel_command(&id));
                        }
                        Control::Reconfigure(update, response) => {
                            let _ = response.send(runtime.reconfigure_partial(update));
                        }
                        Control::Recover(component, message) => {
                            runtime.report_recoverable(&component, message);
                        }
                        Control::Publish(event) => runtime.publish_event(event),
                        Control::Shutdown(response) => {
                            let result = runtime.shutdown();
                            let successful = result.is_ok();
                            let _ = response.send(result);
                            if successful {
                                break;
                            }
                        }
                    }
                }
            }
        }
    });
    Ok((handle, RuntimeTask { task }))
}

fn stopped() -> CoreError {
    CoreError::Unavailable("runtime service stopped".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CommandRegistry, CoreMetrics, MockRecognizer,
        config::PolicyConfig,
        domain::{CommandParameters, CommandResult},
    };
    use async_trait::async_trait;

    struct Noop;

    #[async_trait]
    impl CommandExecutor for Noop {
        async fn execute(
            &self,
            _: &str,
            _: &CommandParameters,
        ) -> Result<CommandResult, CoreError> {
            Ok(CommandResult {
                message: "ok".into(),
            })
        }
    }

    #[tokio::test]
    async fn state_reads_do_not_queue_behind_the_runtime() {
        let mut runtime = Runtime::new_with_wake_words(
            Arc::new(MockRecognizer {
                text: String::new(),
            }),
            CommandRegistry::new(Vec::new(), true),
            Arc::new(Noop),
            vec!["assistant".into()],
            PolicyConfig::default(),
            Duration::from_secs(1),
            Duration::from_millis(10),
            CoreMetrics::default(),
        );
        runtime.start().unwrap();
        let (handle, task) = spawn_runtime_service(runtime, 4).unwrap();
        assert_eq!(handle.state(), AssistantState::IdleListening);
        handle.begin_capture().await.unwrap();
        assert_eq!(handle.state(), AssistantState::CapturingCommand);
        handle.shutdown().await.unwrap();
        task.join().await.unwrap();
    }
}
