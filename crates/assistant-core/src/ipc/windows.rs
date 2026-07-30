use std::{
    ffi::c_void,
    future::Future,
    io, mem,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::windows::named_pipe::{ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions},
    sync::broadcast,
    time::{Instant, sleep, timeout},
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE, LocalFree},
    Security::{
        Authorization::{ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1},
        PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES,
    },
    System::Threading::CreateMutexW,
};

use crate::domain::{AssistantEvent, PROTOCOL_VERSION};

use super::{CoreRequest, CoreResponse, Envelope, IpcErrorCode, decode, encode};

const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_PIPE_INSTANCES: usize = 16;

/// Boxed asynchronous server response.
pub type ResponseFuture = Pin<Box<dyn Future<Output = CoreResponse> + Send>>;
/// Concurrent request callback used by [`serve_named_pipe`].
pub type RequestHandler = Arc<dyn Fn(CoreRequest) -> ResponseFuture + Send + Sync>;

/// Reconnecting request client for the stable Windows IPC contract.
#[derive(Clone)]
pub struct CoreIpcClient {
    pipe_name: Arc<str>,
    io_timeout: Duration,
    sequence: Arc<AtomicU64>,
}

impl CoreIpcClient {
    /// Creates a client for a pipe suffix such as `voice-assistant-core`.
    pub fn new(pipe_name: impl Into<Arc<str>>, io_timeout: Duration) -> io::Result<Self> {
        let pipe_name = pipe_name.into();
        validate_pipe_name(&pipe_name)?;
        Ok(Self {
            pipe_name,
            io_timeout,
            sequence: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Sends one request over a fresh connection, avoiding stale pooled pipes.
    pub async fn request(&self, request: CoreRequest) -> Result<CoreResponse, IpcClientError> {
        let request_id = self.next_request_id();
        let mut pipe = self.connect().await?;
        write_envelope(
            &mut pipe,
            &Envelope::new(request_id.clone(), request),
            self.io_timeout,
        )
        .await?;
        let response: Envelope<CoreResponse> = read_envelope(&mut pipe, self.io_timeout).await?;
        validate_response(&response, &request_id)?;
        match response.payload {
            CoreResponse::Error { code, message } => Err(IpcClientError::Server { code, message }),
            response => Ok(response),
        }
    }

    /// Opens a dedicated event-only connection.
    pub async fn subscribe(&self) -> Result<EventSubscription, IpcClientError> {
        let request_id = self.next_request_id();
        let mut pipe = self.connect().await?;
        write_envelope(
            &mut pipe,
            &Envelope::new(request_id.clone(), CoreRequest::SubscribeEvents),
            self.io_timeout,
        )
        .await?;
        let response: Envelope<CoreResponse> = read_envelope(&mut pipe, self.io_timeout).await?;
        validate_response(&response, &request_id)?;
        if !matches!(response.payload, CoreResponse::Accepted) {
            return Err(IpcClientError::UnexpectedResponse);
        }
        Ok(EventSubscription {
            pipe,
            request_id,
            io_timeout: self.io_timeout,
        })
    }

    async fn connect(&self) -> Result<NamedPipeClient, IpcClientError> {
        let path = format!(r"\\.\pipe\{}", self.pipe_name);
        let deadline = Instant::now() + self.io_timeout;
        loop {
            match ClientOptions::new().open(&path) {
                Ok(pipe) => return Ok(pipe),
                Err(error) if Instant::now() < deadline => {
                    tracing::debug!(%error, "IPC pipe busy or unavailable; retrying");
                    sleep(Duration::from_millis(25)).await;
                }
                Err(error) => return Err(IpcClientError::Io(error)),
            }
        }
    }

    fn next_request_id(&self) -> String {
        format!(
            "client-{:016x}",
            self.sequence.fetch_add(1, Ordering::Relaxed)
        )
    }
}

/// Dedicated event stream returned by [`CoreIpcClient::subscribe`].
pub struct EventSubscription {
    pipe: NamedPipeClient,
    request_id: String,
    io_timeout: Duration,
}

impl EventSubscription {
    /// Reads the next event; reconnect by creating a new subscription after error.
    pub async fn next(&mut self) -> Result<AssistantEvent, IpcClientError> {
        let response: Envelope<CoreResponse> =
            read_envelope(&mut self.pipe, self.io_timeout).await?;
        validate_response(&response, &self.request_id)?;
        match response.payload {
            CoreResponse::Event { event } => Ok(event),
            CoreResponse::Error { code, message } => Err(IpcClientError::Server { code, message }),
            _ => Err(IpcClientError::UnexpectedResponse),
        }
    }
}

/// Typed client failures suitable for application retry policy.
#[derive(Debug, Error)]
pub enum IpcClientError {
    /// Pipe connection or framed I/O failed.
    #[error("IPC I/O failed: {0}")]
    Io(#[from] io::Error),
    /// Response version or request identifier does not match.
    #[error("IPC response protocol or request id does not match")]
    Protocol,
    /// Server returned a typed failure.
    #[error("IPC server returned {code:?}: {message}")]
    Server {
        /// Stable server error category.
        code: IpcErrorCode,
        /// Server diagnostic text.
        message: String,
    },
    /// Response variant is invalid for the operation.
    #[error("IPC server returned an unexpected response")]
    UnexpectedResponse,
}

/// Serves concurrent, local-only IPC clients using the current protocol.
pub async fn serve_named_pipe(
    pipe_name: &str,
    io_timeout: Duration,
    handler: RequestHandler,
    events: broadcast::Sender<AssistantEvent>,
) -> io::Result<()> {
    validate_pipe_name(pipe_name)?;
    let _instance = InstanceMutex::acquire(pipe_name)?;
    let path = format!(r"\\.\pipe\{pipe_name}");
    let security = PipeSecurity::current_owner()?;
    let mut first_instance = true;
    loop {
        let mut options = ServerOptions::new();
        options
            .reject_remote_clients(true)
            .max_instances(MAX_PIPE_INSTANCES);
        if first_instance {
            options.first_pipe_instance(true);
        }
        // SAFETY: `security` owns the descriptor for this call and outlives it.
        let server = unsafe {
            options.create_with_security_attributes_raw(
                &path,
                security.attributes() as *mut SECURITY_ATTRIBUTES as *mut c_void,
            )?
        };
        first_instance = false;
        server.connect().await?;
        let client_handler = Arc::clone(&handler);
        let client_events = events.clone();
        tokio::spawn(async move {
            if let Err(error) =
                serve_client(server, &client_handler, &client_events, io_timeout).await
            {
                tracing::warn!(%error, "IPC client disconnected");
            }
        });
    }
}

struct InstanceMutex(HANDLE);

impl InstanceMutex {
    fn acquire(pipe_name: &str) -> io::Result<Self> {
        let name: Vec<u16> = format!(r"Local\{pipe_name}-daemon")
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: the name is a valid NUL-terminated UTF-16 string.
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        // GetLastError must be read immediately after a successful CreateMutexW.
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            // SAFETY: `handle` was returned by CreateMutexW.
            unsafe { CloseHandle(handle) };
            return Err(io::Error::new(
                io::ErrorKind::AddrInUse,
                "another daemon already owns this pipe",
            ));
        }
        Ok(Self(handle))
    }
}

impl Drop for InstanceMutex {
    fn drop(&mut self) {
        // SAFETY: this handle is owned by the guard and closed exactly once.
        unsafe { CloseHandle(self.0) };
    }
}

async fn serve_client(
    mut server: NamedPipeServer,
    handler: &RequestHandler,
    events: &broadcast::Sender<AssistantEvent>,
    io_timeout: Duration,
) -> io::Result<()> {
    loop {
        let request = read_request(&mut server, io_timeout).await?;
        let request_id = request.request_id.clone();
        if request.protocol_version != PROTOCOL_VERSION {
            write_response(
                &mut server,
                Envelope::new(
                    request_id,
                    CoreResponse::Error {
                        code: IpcErrorCode::ProtocolVersion,
                        message: format!(
                            "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
                            request.protocol_version
                        ),
                    },
                ),
                io_timeout,
            )
            .await?;
            continue;
        }
        if matches!(request.payload, CoreRequest::SubscribeEvents) {
            write_response(
                &mut server,
                Envelope::new(request_id.clone(), CoreResponse::Accepted),
                io_timeout,
            )
            .await?;
            return serve_events(&mut server, events.subscribe(), request_id, io_timeout).await;
        }
        let response = handler(request.payload).await;
        write_response(&mut server, Envelope::new(request_id, response), io_timeout).await?;
    }
}

async fn serve_events(
    server: &mut NamedPipeServer,
    mut events: broadcast::Receiver<AssistantEvent>,
    request_id: String,
    io_timeout: Duration,
) -> io::Result<()> {
    loop {
        let response = match events.recv().await {
            Ok(event) => CoreResponse::Event { event },
            Err(broadcast::error::RecvError::Lagged(skipped)) => CoreResponse::Error {
                code: IpcErrorCode::Internal,
                message: format!("event subscriber missed {skipped} events"),
            },
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        };
        write_response(
            server,
            Envelope::new(request_id.clone(), response),
            io_timeout,
        )
        .await?;
    }
}

async fn read_request(
    server: &mut NamedPipeServer,
    io_timeout: Duration,
) -> io::Result<Envelope<CoreRequest>> {
    read_envelope(server, io_timeout).await
}

async fn read_envelope<T: DeserializeOwned>(
    stream: &mut (impl AsyncRead + Unpin),
    io_timeout: Duration,
) -> io::Result<Envelope<T>> {
    let size = timeout(io_timeout, stream.read_u32_le())
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "IPC read timed out"))??
        as usize;
    if size == 0 || size > MAX_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IPC message size is invalid",
        ));
    }
    let mut bytes = vec![0; size];
    timeout(io_timeout, stream.read_exact(&mut bytes))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "IPC read timed out"))??;
    decode(&bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

async fn write_response(
    server: &mut NamedPipeServer,
    response: Envelope<CoreResponse>,
    io_timeout: Duration,
) -> io::Result<()> {
    write_envelope(server, &response, io_timeout).await
}

async fn write_envelope<T: Serialize>(
    stream: &mut (impl AsyncWrite + Unpin),
    response: &Envelope<T>,
    io_timeout: Duration,
) -> io::Result<()> {
    let bytes =
        encode(response).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    timeout(io_timeout, async {
        stream.write_u32_le(bytes.len() as u32).await?;
        stream.write_all(&bytes).await
    })
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "IPC write timed out"))?
}

fn validate_response<T>(response: &Envelope<T>, request_id: &str) -> Result<(), IpcClientError> {
    if response.protocol_version != PROTOCOL_VERSION || response.request_id != request_id {
        return Err(IpcClientError::Protocol);
    }
    Ok(())
}

fn validate_pipe_name(pipe_name: &str) -> io::Result<()> {
    if !pipe_name.is_empty()
        && pipe_name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
    {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid pipe name",
    ))
}

struct PipeSecurity {
    descriptor: PSECURITY_DESCRIPTOR,
    attributes: SECURITY_ATTRIBUTES,
}

impl PipeSecurity {
    fn current_owner() -> io::Result<Self> {
        // SYSTEM, administrators and the object owner get full access.
        let sddl: Vec<u16> = "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;OW)\0"
            .encode_utf16()
            .collect();
        let mut descriptor = std::ptr::null_mut();
        // SAFETY: SDDL is NUL-terminated and both output pointers are valid.
        let converted = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                std::ptr::null_mut(),
            )
        };
        if converted == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            descriptor,
            attributes: SECURITY_ATTRIBUTES {
                nLength: mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: descriptor,
                bInheritHandle: 0,
            },
        })
    }

    fn attributes(&self) -> *const SECURITY_ATTRIBUTES {
        &self.attributes
    }
}

impl Drop for PipeSecurity {
    fn drop(&mut self) {
        // SAFETY: ConvertStringSecurityDescriptor allocated this descriptor with LocalAlloc.
        unsafe {
            let _ = LocalFree(self.descriptor);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_names_are_restricted() {
        assert!(validate_pipe_name("voice-assistant-core").is_ok());
        assert!(validate_pipe_name(r"..\other").is_err());
        PipeSecurity::current_owner().unwrap();
    }

    #[test]
    fn daemon_mutex_rejects_a_second_owner() {
        let name = format!("voice-assistant-core-test-{}", std::process::id());
        let first = InstanceMutex::acquire(&name).unwrap();
        let second = match InstanceMutex::acquire(&name) {
            Ok(_) => panic!("second mutex owner was accepted"),
            Err(error) => error,
        };
        assert_eq!(second.kind(), io::ErrorKind::AddrInUse);
        drop(first);
        InstanceMutex::acquire(&name).unwrap();
    }
}
