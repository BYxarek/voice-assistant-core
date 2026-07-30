#![doc = include_str!("../../../README.md")]
#![warn(missing_docs)]

/// Audio device access, WAV I/O and canonical resampling.
pub mod audio;
/// Command matching and typed extension handlers.
pub mod commands;
/// Versioned configuration, validation and application data paths.
pub mod config;
/// Stable domain types, state machine events and adapter traits.
pub mod domain;
/// Versioned IPC envelopes and Windows named-pipe transport.
pub mod ipc;
/// Lock-free operational metrics.
pub mod metrics;
#[cfg(feature = "model-source-huggingface")]
/// Pinned model installation and verification.
pub mod models;
/// Stateful command-processing runtime.
pub mod runtime;
/// Dedicated runtime task and application-facing handle.
pub mod service;
/// Voice activity detection and bounded command capture.
pub mod signal;
#[cfg(feature = "stt-sherpa-onnx")]
/// Persistent sherpa-onnx speech recognizer.
pub mod stt;
#[cfg(feature = "stt-sherpa-onnx")]
/// Sherpa-onnx open-vocabulary wake-word detector.
pub mod wakeword;

/// Semantic version of the running core build.
pub const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Stable Rust extension API version. Increment only for breaking contracts.
pub const CORE_API_VERSION: u16 = 1;

pub use commands::{
    CommandHandler, CommandRegistry, HandlerRegistry, HandlerSchema, LaunchAppExecutor,
    LaunchAppHandler, builtin_handlers,
};
pub use config::{AppPaths, CoreConfig};
pub use domain::*;
#[cfg(windows)]
pub use ipc::windows::{CoreIpcClient, EventSubscription, IpcClientError};
pub use ipc::{CoreRequest, CoreResponse, Envelope, IpcErrorCode};
pub use metrics::{CoreMetrics, MetricsSnapshot};
pub use runtime::{MockRecognizer, Runtime, UnavailableRecognizer};
pub use service::{RuntimeComponents, RuntimeHandle, RuntimeTask, spawn_runtime_service};
