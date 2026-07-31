#![cfg(windows)]

use std::{
    fs,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use assistant_core::{
    AssistantState, CORE_API_VERSION, CORE_VERSION, CoreConfig, IpcClientError, PROTOCOL_VERSION,
    config::CURRENT_CONFIG_VERSION,
    ipc::{CoreRequest, CoreResponse, IpcErrorCode, windows::CoreIpcClient},
};

struct DaemonProcess {
    child: Child,
    root: PathBuf,
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_serves_ipc_client_until_remote_shutdown() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "voice-assistant-daemon-e2e-{}-{nonce}",
        std::process::id()
    ));
    let config_path = root.join("assistant.toml");
    let models_path = root.join("models");
    let mut config = CoreConfig::bundled_example().unwrap();
    config.ipc.pipe_name = format!("voice-assistant-e2e-{}-{nonce}", std::process::id());
    config.ipc.io_timeout_ms = 2_000;
    config.save_atomic(&config_path).unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_assistant-daemon"))
        .args(["--config"])
        .arg(&config_path)
        .args(["--models"])
        .arg(&models_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut daemon = DaemonProcess { child, root };
    let client = CoreIpcClient::new(config.ipc.pipe_name, Duration::from_secs(5)).unwrap();

    let health = match client.request(CoreRequest::GetHealth).await.unwrap() {
        CoreResponse::Health { health } => health,
        response => panic!("unexpected health response: {response:?}"),
    };
    assert!(matches!(
        health.state,
        AssistantState::Starting | AssistantState::IdleListening
    ));
    assert_eq!(health.core_version, CORE_VERSION);
    assert!(!health.model_ready);
    assert_eq!(health.config_version, CURRENT_CONFIG_VERSION);
    assert_eq!(health.core_api_version, CORE_API_VERSION);
    assert_eq!(health.protocol_version, PROTOCOL_VERSION);
    assert!(
        health
            .components
            .iter()
            .any(|component| component.name == "stt")
    );

    match client.request(CoreRequest::ListHandlers).await.unwrap() {
        CoreResponse::Handlers { handlers } => {
            assert!(handlers.iter().any(|handler| handler.name == "launch_app"));
        }
        response => panic!("unexpected handlers response: {response:?}"),
    }

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match client.request(CoreRequest::GetStatus).await.unwrap() {
            CoreResponse::Status {
                state: AssistantState::IdleListening,
            } => break,
            CoreResponse::Status {
                state: AssistantState::Starting,
            } => {
                assert!(
                    Instant::now() < deadline,
                    "daemon did not complete initialization"
                );
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            response => panic!("unexpected startup response: {response:?}"),
        }
    }

    assert!(matches!(
        client
            .request(CoreRequest::SubmitText {
                text: String::new()
            })
            .await
            .unwrap_err(),
        IpcClientError::Server {
            code: IpcErrorCode::InvalidRequest,
            ..
        }
    ));
    assert!(matches!(
        client.request(CoreRequest::BeginCapture).await.unwrap_err(),
        IpcClientError::Server {
            code: IpcErrorCode::Audio,
            ..
        }
    ));

    assert!(matches!(
        client.request(CoreRequest::SuspendListening).await.unwrap(),
        CoreResponse::Accepted
    ));
    assert!(matches!(
        client.request(CoreRequest::GetStatus).await.unwrap(),
        CoreResponse::Status {
            state: AssistantState::Suspended
        }
    ));
    assert!(matches!(
        client.request(CoreRequest::ResumeListening).await.unwrap(),
        CoreResponse::Accepted
    ));
    assert!(matches!(
        client.request(CoreRequest::Shutdown).await.unwrap(),
        CoreResponse::Accepted
    ));

    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = daemon.child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "daemon ignored IPC shutdown");
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert!(status.success(), "daemon exited with {status}");
}
