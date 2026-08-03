# Voice Assistant Core

[Русский](README.md) | [English](README.en.md)

[![CI](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml/badge.svg)](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml)
[![Documentation](https://github.com/BYxarek/voice-assistant-core/actions/workflows/docs.yml/badge.svg)](https://byxarek.github.io/voice-assistant-core/)
[![Release](https://img.shields.io/github/v/release/BYxarek/voice-assistant-core)](https://github.com/BYxarek/voice-assistant-core/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BYxarek/voice-assistant-core/blob/main/LICENSE)

A local Rust voice-assistant core for Windows 10/11. The core captures microphone
audio, detects a wake phrase, recognizes Russian speech, and runs only registered,
typed commands.

The current stable release is **1.5.1**. It exposes Rust extension API v6 and IPC
protocol v7. Audio formats, queues, and inference are isolated from the GUI.

## Versioning

The project follows SemVer and publishes core milestones as `vMAJOR.MINOR.PATCH` tags:

- `CORE_VERSION` and `HealthSnapshot.core_version` contain the core build version;
- `CORE_API_VERSION` changes when the Rust extension API becomes incompatible;
- `PROTOCOL_VERSION` changes when the IPC wire format becomes incompatible;
- `CURRENT_CONFIG_VERSION` versions the configuration schema independently.

See [CHANGELOG.md](CHANGELOG.md) for release history. Every `v*` tag automatically
builds and verifies a Windows x64 archive containing the daemon, CLI, example
configuration, and required runtime DLLs.

## Features

- bounded audio pipeline: CPAL → mono → 16 kHz → KWS/VAD → STT;
- while `Starting` or `Suspended`, the worker drains bounded input without
  resampling, VAD, or inference;
- dedicated runtime task and supervised persistent STT worker with warm-up,
  timeout watchdog, bounded restart budget, and exponential backoff;
- IPC continuous recognition segments every utterance in the core, retains pre-roll,
  and publishes partial/final transcripts linked by `session_id`;
- ordinary speech never reaches command matching; a wake word, explicit push-to-talk/
  `submit_text`, or a one-time confirmation authorizes an action;
- IPC is available in `Starting` while a cancellable blocking worker loads STT
  and publishes progress;
- microphone recovery after callback failure or stall, with automatic switching
  when the default input changes;
- manual push-to-talk and safe text submission over IPC without bypassing command policy;
- typed command handlers without passing recognized text to a shell;
- one-time `confirmation_id`, spoken yes/no confirmation, timeout, and an option
  to disable confirmations completely;
- typed allowlisted command slots, handler rate limiting, and circuit breakers;
- startup without a model, plus runtime model installation, cancellation, and verification;
- pinned model revisions, SHA-256 manifests, and atomic activation;
- differential configuration updates without restarting unaffected components;
- local versioned Named Pipe IPC and a ready-to-use Rust client;
- multiple wake-word aliases and IPC discovery of registered command-handler schemas;
- health, model state, events, queue statistics, latency, CPU time, and RAM over IPC;
- component health for audio, STT, runtime, model, and IPC, including restart count
  and fault state;
- CLI tools for diagnostics, WAV verification, and soak testing.

## Requirements

- Windows 10/11 x64;
- Rust 1.97 or newer, only when building from source;
- a microphone for voice mode;
- network access only for the first model installation.

## Quick start

For normal use, download the Windows x64 ZIP from
[Releases](https://github.com/BYxarek/voice-assistant-core/releases), extract it,
and run the prebuilt applications:

```powershell
.\assistant-daemon.exe
.\assistant-cli.exe status
```

Rust and Cargo are not required for the packaged release. To develop the core:

```powershell
cargo run -p assistant-cli -- validate-config
cargo run -p assistant-cli -- devices
cargo run -p assistant-cli -- models list
cargo run -p assistant-cli -- models install
cargo run -p assistant-daemon
```

In a second terminal:

```powershell
cargo run -p assistant-cli -- status
cargo run -p assistant-cli -- soak --seconds 300
```

Default paths:

```text
%LOCALAPPDATA%\VoiceAssistantCore\
├── assistant.toml
├── models\
└── logs\
```

The first launch creates a valid configuration. The daemon can run without a
model and reports `model_status = missing`; install a model through IPC or the
CLI. For development, override the paths with global arguments:

```powershell
cargo run -p assistant-daemon -- --config .\config\assistant.example.toml --models .\models
cargo run -p assistant-cli -- --config .\config\assistant.example.toml --models .\models status
```

## Public Rust API v6

The stable export boundary is the `assistant_core` crate root.
`CORE_API_VERSION` is `6`. Version 6 includes:

- `CommandHandler`, `HandlerSchema`, and `HandlerRegistry` for command extensions;
- `RuntimeComponents`, `RuntimeUpdate`, `RuntimeHandle`, `RuntimeTask`, and
  `spawn_runtime_service` for embedding;
- `CoreIpcClient`, `EventSubscription`, `CoreRequest`, `CoreResponse`,
  `Envelope`, and `IpcErrorCode` for IPC;
- `CoreConfig`, `CommandConfig`, `SlotConfig`, and public domain events, states,
  and errors;
- streaming `SpeechRecognizer::push_stream_partial`;
- VAD session events and session-linked partial/final transcripts;
- `CoreMetrics` and `MetricsSnapshot`.

Breaking changes to these contracts require a new major crate API version and
an increment of `CORE_API_VERSION`. IPC changes remain compatible within
protocol v7; an incompatible wire-format change increments `PROTOCOL_VERSION`.

Minimal command extension:

```rust,no_run
use std::sync::Arc;
use async_trait::async_trait;
use assistant_core::{
    CommandHandler, CommandParameters, CommandResult, CoreError, HandlerRegistry,
    HandlerSchema,
};

struct Mute;

#[async_trait]
impl CommandHandler for Mute {
    fn schema(&self) -> HandlerSchema {
        HandlerSchema::new(
            "mute",
            std::iter::empty::<&str>(),
            std::iter::empty::<&str>(),
        )
    }

    async fn execute(&self, _parameters: &CommandParameters) -> Result<CommandResult, CoreError> {
        Ok(CommandResult {
            message: "muted".into(),
        })
    }
}

let mut handlers = HandlerRegistry::new();
handlers.register(Arc::new(Mute)).expect("unique valid handler");
```

`HandlerSchema` validates the handler name and allowlists for required and
optional parameters. Duplicate handler names are rejected. The application
passes the registry through `RuntimeComponents`; the daemon uses the built-in
`launch_app` handler.

STT and wake-word detection remain replaceable through `SpeechRecognizer` and
`WakeWordDetector`. The GUI does not embed the internal runtime; it uses IPC.

## IPC protocol v7

### Web Speech API as an external STT engine

The Web Speech API can be integrated without changing the core or IPC protocol.
Browser `SpeechRecognition` performs recognition, and the GUI sends only the
final text through the existing `submit_text` request:

```text
SpeechRecognition → native GUI bridge → Windows named pipe
                  → CoreRequest::SubmitText → core commands and policy
```

A regular web page cannot open a Windows named pipe directly. A WebView2,
Tauri, or Electron application needs a narrow native bridge that exposes only
`submitText(text)` and does not give web content access to a shell or arbitrary
IPC requests.

```javascript
const Recognition =
  window.SpeechRecognition ?? window.webkitSpeechRecognition;

if (!Recognition) {
  throw new Error("Web Speech API is unavailable in this browser");
}

const recognition = new Recognition();
recognition.lang = "ru-RU";
recognition.continuous = false;
recognition.interimResults = true;
recognition.maxAlternatives = 1;

recognition.onresult = async (event) => {
  for (let i = event.resultIndex; i < event.results.length; i += 1) {
    const result = event.results[i];
    const text = result[0].transcript.trim();

    if (result.isFinal && text) {
      await window.voiceAssistant.submitText(text);
    } else {
      renderPartial(text); // Display only; do not execute a command.
    }
  }
};

recognition.onerror = (event) => renderRecognitionError(event.error);
document.querySelector("#listen").addEventListener("click", () => {
  recognition.start();
});
```

The native bridge encodes the call as a regular IPC request:

```json
{
  "protocol_version": 7,
  "request_id": "web-speech-1",
  "payload": {
    "type": "submit_text",
    "text": "открой блокнот"
  }
}
```

Send only results where `isFinal === true`; otherwise one phrase may execute a
command more than once. `submit_text` accepts non-empty UTF-8 text up to 4096
bytes. The normal command matcher, allowlist, risk evaluation, confirmation,
and timeout then apply. Browser confidence is not a security boundary.

Web Speech API support varies by browser. Some implementations send audio to an
external cloud service, so offline operation and privacy are not guaranteed.
Request microphone access in the GUI following an explicit user action, display
recognition errors, and keep local `sherpa-onnx` as the primary option for
offline scenarios. Check current compatibility in the
[SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition)
and [`isFinal`](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognitionResult/isFinal)
documentation.

The default pipe is `\\.\pipe\voice-assistant-core`. The server allows only
local clients, is protected by a DACL, and permits one daemon per pipe. Each
JSON envelope has `protocol_version`, `request_id`, and `payload`, prefixed by a
little-endian `u32` length. The maximum payload is 1 MiB.

Requests:

- `get_status`, `get_health`, `get_config`, `get_metrics`, `list_handlers`;
- `validate_config`, `apply_config`, `list_audio_devices`;
- `get_model_status`, `install_model`, `cancel_model_install`, `verify_model`;
- `suspend`, `resume`, `set_continuous_recognition`, `begin_capture`, `end_capture`, `submit_text`;
- `confirm`, `cancel`;
- `subscribe_events`, `shutdown`.

`begin_capture` starts capture without a wake word. `end_capture` sends audio to
STT only when it is at least `audio.command_min_ms` long. `submit_text` accepts a
non-empty UTF-8 string up to 4096 bytes, matches it without requiring a wake-word
prefix, and applies the same risk, confirmation, and timeout rules.

`set_continuous_recognition { enabled: true }` enables daemon-side VAD segmentation.
It emits `speech_started { session_id }`, `speech_ended { session_id }`, and
session-linked `transcript_partial`/`transcript_final` events. Speech without a
wake word is transcribed but does not execute a command. Silence uses VAD speech
frames instead of averaging the full recording together with trailing silence.

`CoreIpcClient` validates the version, limits message size, and retries transient
connection failures. `subscribe_events` creates a separate `EventSubscription`.
Errors have stable `IpcErrorCode` values.

`list_handlers` returns sorted `HandlerSchema` values with required and optional
parameters. `HealthSnapshot.components` reports `ready`, `recovering`, `faulted`,
or `stopped`, together with the automatic restart count.

The event stream publishes changed `transcript_partial { session_id, text, confidence }`
events, a final `transcript_final { session_id, text, confidence }`, `audio_level { rms }` at most ten times per
second, and `transcript_unavailable { reason }`. Reasons are
`wake_word_not_detected`, `too_short`, `silence`, and `model_unavailable`.
`audio.device_id` cannot be empty; use `default` for the system input.

With `audio.device_id = "default"`, the worker watches the current Windows input
endpoint and automatically reopens the stream after it changes. `HealthSnapshot`
contains `active_audio_device`, and subscribers receive `audio_device_changed`;
`None` means that no working input endpoint is currently available. If an
explicitly selected endpoint disappears, the daemon temporarily opens the system
default, publishes `audio_device_fallback { requested_device_id, device }`, and
returns to the selected endpoint when it reappears.

Configuration can be applied only from `Idle` or `Suspended`. After migration
and validation, the runtime updates only affected components: commands,
confirmations, and microphone changes do not recreate STT; a new STT worker is
needed only when its threads or queue change. The file is replaced atomically.
Pipe settings require a daemon restart.

```toml
[wake_word]
model = "alphacep/vosk-model-streaming-ru"
keyword = "ассистент"
aliases = ["помощник"]

[inference]
model = "alphacep/vosk-model-streaming-ru"
threads = 0 # automatically uses available_parallelism
max_restarts = 3
restart_backoff_ms = 500
```

## Commands and security

```toml
[[commands]]
id = "open_notepad"
enabled = true
phrases = ["открой блокнот", "запусти блокнот"]
handler = "launch_app"
risk = "low"
requires_confirmation = false
timeout_ms = 10000

[commands.parameters]
executable = "notepad.exe"
```

The standard daemon also registers `open_url { url }`, `set_volume { level }`,
and `click_mouse { x, y, duration_ms? }`. URLs are limited to `http` and `https`;
volume must be an integer from `0` through `100`. `click_mouse` requires
`risk = "high"`; the cursor moves to the coordinates over 350 ms by default or
over the configured `100..10000` ms.

A command may contain up to 32 sequential `actions`. `delay_ms` sets the delay
before a step, up to 60000 ms, while `timeout_ms` limits the entire sequence:

```toml
[[commands]]
id = "open_site_and_set_volume"
enabled = true
phrases = ["открой сайт"]
risk = "low"
timeout_ms = 15000

[[commands.actions]]
handler = "open_url"

[commands.actions.parameters]
url = "https://example.com"

[[commands.actions]]
handler = "set_volume"
delay_ms = 500

[commands.actions.parameters]
level = "35"
```

`handler`/`parameters` and `actions` are mutually exclusive. All values are
fixed in configuration and checked against allowlists; recognized text cannot
become a URL, coordinate, or volume level.

Single-word typed slots use `{name}` in a phrase template. A text slot must have
an explicit `values` allowlist; an integer slot must have `min` and `max` bounds.
The handler validates the substituted value again:

```toml
[[commands]]
id = "set_volume_voice"
phrases = ["установи громкость {level}"]
handler = "set_volume"

[commands.slots.level]
type = "integer"
min = 0
max = 100

[commands.parameters]
level = "{level}"
```

Recognized text is never executed by `cmd.exe` or PowerShell. A handler receives
only validated `CommandConfig`. For commands requiring confirmation, the event
contains a one-time `confirmation_id`; pass that exact value to `confirm` or
`cancel`.

Confirmations are enabled by default. To explicitly disable every confirmation,
including those for high-risk commands:

```toml
[policy]
confirmations_enabled = false
confirmation_timeout_ms = 15000
voice_confirmations_enabled = true
confirmation_accept_phrases = ["да", "подтверждаю"]
confirmation_cancel_phrases = ["нет", "отмена"]
handler_rate_limit_ms = 500
handler_failure_threshold = 3
handler_circuit_breaker_ms = 30000
```

While a confirmation is pending, the user says the wake word again and then an
accept or cancel phrase. The rate limit applies between executions of one
handler. After the configured number of consecutive failures, the circuit
breaker temporarily rejects calls to that handler. Setting
`confirmations_enabled = false` deliberately reduces protection.

## Models

The built-in catalog contains seven pinned STT models from
[`alphacep`](https://huggingface.co/alphacep/models): Russian, Bengali, Tajik,
and Uzbek models in online and offline variants. List all models with:

```powershell
cargo run -p assistant-cli -- models list
cargo run -p assistant-cli -- models install alphacep/vosk-model-small-ru
```

Example with offline STT and a separate streaming wake-word model:

```toml
schema_version = 6

[inference]
model = "alphacep/vosk-model-small-ru"
threads = 0

[wake_word]
model = "alphacep/vosk-model-streaming-ru"
keyword = "ассистент"
aliases = ["помощник"]
```

Install both models, validate the configuration, and transcribe a WAV file:

```powershell
cargo run -p assistant-cli -- models install alphacep/vosk-model-small-ru
cargo run -p assistant-cli -- models install alphacep/vosk-model-streaming-ru
cargo run -p assistant-cli -- validate-config
cargo run -p assistant-cli -- transcribe .\command.wav
```

`inference.model` selects STT, while `wake_word.model` selects a separate streaming
KWS model with `lang/bpe.model`. `FinalOnly` models and models without SentencePiece
are available for STT but are intentionally rejected as wake-word models.
Changing a model through `ApplyConfig` requires a daemon restart; IPC
`InstallModel` installs both selected models.

Each installation uses an isolated temporary Hugging Face cache, preventing concurrent
Windows processes from corrupting snapshot pointers. A model-source failure or panic is
returned as `ModelError::Hub` instead of terminating the background task.

```text
models\<repo-name>\<revision>\
├── manifest.toml
├── am-onnx\
└── lang\
```

Using the catalog and recognizer directly from Rust:

```rust,no_run
use assistant_core::{
    CoreMetrics, SpeechRecognizer, TranscriptionRequest,
    audio::read_wav_mono,
    models::{ModelManager, model_spec},
    stt::SherpaOnnxRecognizer,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "alphacep/vosk-model-small-ru";
    let manager = ModelManager::new(r"C:\VoiceAssistantCore\models");
    let directory = manager.resolve(model_id)?;
    let (samples, sample_rate) = read_wav_mono(r"C:\audio\command.wav")?;
    let recognizer = SherpaOnnxRecognizer::new_for_model(
        directory,
        model_spec(model_id)?,
        0,
        2,
        CoreMetrics::default(),
    )?;
    let transcript = recognizer
        .transcribe(TranscriptionRequest { samples, sample_rate })
        .await?;
    println!("{}", transcript.text);
    Ok(())
}
```

Before loading, the core verifies write/rename access and at least 256 MiB of
free space. It downloads only allowlisted files. Installation uses a temporary
directory, verifies its contents, and activates it with a rename. A damaged
active revision is temporarily quarantined and restored if the new installation
does not complete.

## Audio diagnostics

```powershell
cargo run -p assistant-cli -- record --seconds 5 --output command.wav
cargo run -p assistant-cli -- transcribe command.wav
cargo run -p assistant-cli -- test-wake-word command.wav
cargo run -p assistant-cli -- evaluate command.wav --threads 4
cargo run -p assistant-cli -- soak --seconds 3600
cargo run -p assistant-cli -- handlers
```

`evaluate` prints the KWS result, full transcript, STT latency, and real-time
factor. Compare one unchanged WAV when selecting `inference.threads`:

```powershell
1, 2, 4, 8, 0 | ForEach-Object {
    cargo run --release -p assistant-cli -- evaluate command.wav --threads $_
}
```

`0` uses host parallelism. Select the lowest stable real-time factor while
confirming that the transcript remains identical. `soak` watches for stalled
callbacks, reconnects, bounded-queue losses, and process metrics.

Real WAV files are not stored in Git. Run the repeatable regression test with a
fixed external file:

```powershell
$env:VOICE_ASSISTANT_TEST_WAV = "C:\fixtures\assistant-open-notepad.wav"
$env:VOICE_ASSISTANT_TEST_MODEL = "$env:LOCALAPPDATA\VoiceAssistantCore\models\vosk-model-streaming-ru\83bbf6f40059480e96251be8aed3d32bc7c80c33"
$env:VOICE_ASSISTANT_TEST_TRANSCRIPT = "ассистент открой блокнот"
cargo test -p assistant-core --all-features --test real_audio -- --ignored
```

To verify any catalog `Streaming`/`FinalOnly` model separately, set
`VOICE_ASSISTANT_TEST_STT_WAV`, `VOICE_ASSISTANT_TEST_STT_MODEL`,
`VOICE_ASSISTANT_TEST_STT_MODEL_ID`, and `VOICE_ASSISTANT_TEST_STT_TRANSCRIPT`,
then run the ignored `fixed_real_wav_transcribes_with_selected_catalog_model` test.

The file must remain an unchanged mono WAV, and the expected transcript is fixed
through the environment variable.

To manually verify streaming events, run the daemon with a `Streaming` model,
enable `set_continuous_recognition`, subscribe an IPC client to events, and say an
ordinary phrase followed by a wake-word command. Session IDs must match across
speech and transcript events, and the ordinary phrase must not start a handler.
For a high-risk command, say the wake word and “да” after
`confirmation_required`; the handler must start exactly once.

## Documentation and checks

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
$env:RUSTDOCFLAGS = "-D warnings"
cargo doc --workspace --all-features --no-deps
```

`cargo test --workspace --all-features` also runs a process-level end-to-end test:
a real `assistant-daemon` starts with a temporary configuration, and an IPC
client verifies health, suspend/resume, and graceful remote shutdown. The MSVC
workspace uses one static CRT compatible with the Windows `sherpa-onnx` build,
so no additional `/NODEFAULTLIB` flags are required.

The complete developer reference is available as
[online documentation](https://byxarek.github.io/voice-assistant-core/) or
[docs/api.html](docs/api.html). Local Rustdoc is generated at
`target\doc\assistant_core\index.html`; the guide is [docs/index.html](docs/index.html).
Models, WAV files, logs, and secrets must not be committed to Git.
