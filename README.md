# Voice Assistant Core

[![CI](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml/badge.svg)](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml)
[![Documentation](https://github.com/BYxarek/voice-assistant-core/actions/workflows/docs.yml/badge.svg)](https://byxarek.github.io/voice-assistant-core/)
[![Release](https://img.shields.io/github/v/release/BYxarek/voice-assistant-core)](https://github.com/BYxarek/voice-assistant-core/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BYxarek/voice-assistant-core/blob/main/LICENSE)

Локальное ядро голосового ассистента для Windows 10/11 на Rust. Ядро принимает
звук с микрофона, обнаруживает ключевую фразу, распознаёт русскую речь и
выполняет только зарегистрированные типизированные команды.

Текущий стабильный релиз — **1.1.1**. Публичный Rust extension API v2 и IPC
protocol v3. Форматы аудио, очереди и
inference изолированы от GUI.

## Версионирование

Проект следует SemVer и публикует стадии ядра тегами `vMAJOR.MINOR.PATCH`:

- `CORE_VERSION` и `HealthSnapshot.core_version` содержат версию сборки ядра;
- `CORE_API_VERSION` меняется при несовместимом изменении Rust extension API;
- `PROTOCOL_VERSION` меняется при несовместимом изменении IPC wire format;
- `CURRENT_CONFIG_VERSION` независимо версионирует конфигурацию.

История релизов находится в [CHANGELOG.md](CHANGELOG.md). Каждый тег `v*`
автоматически собирает и проверяет Windows x64 архив с daemon, CLI, примером
конфигурации и необходимыми runtime-DLL.

## Возможности

- bounded audio pipeline: CPAL → mono → 16 кГц → KWS → VAD → STT;
- отдельная задача runtime и отдельный persistent STT worker;
- IPC доступен в состоянии `Starting`, пока cancellable blocking worker загружает STT и публикует прогресс;
- восстановление микрофона после ошибки или зависания callback и автоматическое
  переключение при смене default input;
- ручной push-to-talk и безопасная отправка текста через IPC без обхода command policy;
- типизированные обработчики команд без передачи текста в shell;
- одноразовый `confirmation_id`, timeout и полное отключение подтверждений;
- запуск без модели, установка/отмена/проверка модели во время работы;
- pinned model revision, SHA-256 manifest и атомарная активация;
- дифференциальное применение конфигурации без перезагрузки незатронутых компонентов;
- локальный versioned Named Pipe IPC и готовый Rust-клиент;
- health, состояние модели, события, очереди, задержки, CPU time и RAM через IPC;
- CLI для диагностики, проверки WAV и soak-теста.

## Требования

- Windows 10/11 x64;
- Rust 1.97 или новее — только для сборки из исходников;
- микрофон для голосового режима;
- сеть только для первой установки модели.

## Быстрый старт

Для обычного использования скачайте Windows x64 ZIP со страницы
[Releases](https://github.com/BYxarek/voice-assistant-core/releases), распакуйте
его и запустите готовые программы:

```powershell
.\assistant-daemon.exe
.\assistant-cli.exe status
```

Rust и Cargo для готового релиза не требуются. Для разработки самого ядра:

```powershell
cargo run -p assistant-cli -- validate-config
cargo run -p assistant-cli -- devices
cargo run -p assistant-cli -- models install
cargo run -p assistant-daemon
```

Во втором терминале:

```powershell
cargo run -p assistant-cli -- status
cargo run -p assistant-cli -- soak --seconds 300
```

По умолчанию используются:

```text
%LOCALAPPDATA%\VoiceAssistantCore\
├── assistant.toml
├── models\
└── logs\
```

При первом запуске создаётся валидная конфигурация. Daemon может работать без
модели и сообщит `model_status = missing`; модель можно установить через IPC или
CLI. Для разработки пути переопределяются глобальными аргументами:

```powershell
cargo run -p assistant-daemon -- --config .\config\assistant.example.toml --models .\models
cargo run -p assistant-cli -- --config .\config\assistant.example.toml --models .\models status
```

## Публичный Rust API v2

Стабильная граница экспорта находится в корне crate `assistant_core`.
`CORE_API_VERSION` равен `2`. В v2 входят:

- `CommandHandler`, `HandlerSchema`, `HandlerRegistry` — extension API команд;
- `RuntimeComponents`, `RuntimeUpdate`, `RuntimeHandle`, `RuntimeTask`,
  `spawn_runtime_service` — embedding API;
- `CoreIpcClient`, `EventSubscription`, `CoreRequest`, `CoreResponse`,
  `Envelope`, `IpcErrorCode` — IPC API;
- `CoreConfig`, `AppPaths`, публичные доменные события, состояния и ошибки;
- `CoreMetrics`, `MetricsSnapshot`.

Ломающие изменения этих контрактов требуют нового major crate API и увеличения
`CORE_API_VERSION`. IPC меняется только совместимо внутри protocol v3; для
несовместимого wire-формата увеличивается `PROTOCOL_VERSION`.

Минимальное расширение команд:

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

`HandlerSchema` проверяет имя и allowlist обязательных/необязательных параметров.
Одинаковые имена обработчиков отклоняются. Приложение передаёт registry в
`RuntimeComponents`; daemon использует встроенный `launch_app`.

STT и wake word остаются заменяемыми через `SpeechRecognizer` и
`WakeWordDetector`. GUI не встраивает внутренний runtime: он использует IPC.

## IPC protocol v3

Pipe по умолчанию: `\\.\pipe\voice-assistant-core`. Сервер допускает только
локальных клиентов, защищён DACL и разрешает один daemon на pipe. Каждый JSON
envelope имеет `protocol_version`, `request_id`, `payload` и little-endian
`u32`-длину; максимум 1 МиБ.

Запросы:

- `get_status`, `get_health`, `get_config`, `get_metrics`;
- `validate_config`, `apply_config`, `list_audio_devices`;
- `get_model_status`, `install_model`, `cancel_model_install`, `verify_model`;
- `suspend`, `resume`, `begin_capture`, `end_capture`, `submit_text`;
- `confirm`, `cancel`;
- `subscribe_events`, `shutdown`.

`begin_capture` запускает запись без wake word, а `end_capture` передаёт в STT
только аудио не короче `audio.command_min_ms`. `submit_text` принимает непустую
UTF-8 строку до 4096 байт, сопоставляет её с командой без обязательного wake-word
prefix и применяет те же risk, confirmation и timeout rules.

`CoreIpcClient` проверяет версию, ограничивает размер сообщения и повторяет
кратковременное подключение. `subscribe_events` создаёт отдельный
`EventSubscription`. Ошибки имеют стабильный `IpcErrorCode`.

Поток событий публикует `audio_level { rms }` не чаще 10 раз в секунду и
`transcript_unavailable { reason }`. Причины: `wake_word_not_detected`,
`too_short`, `silence`, `model_unavailable`. `audio.device_id` не может быть
пустым; для системного input используется значение `default`.

При `audio.device_id = "default"` worker проверяет текущий Windows input endpoint
и автоматически переоткрывает поток после его смены. `HealthSnapshot` содержит
`active_audio_device`, а подписчики получают `audio_device_changed`; значение
`None` означает, что рабочий input endpoint временно недоступен.
Если явно выбранный endpoint исчез, daemon временно открывает системный default,
публикует `audio_device_fallback { requested_device_id, device }` и возвращается
к выбранному endpoint после его появления.

Применение конфигурации выполняется только из `Idle`/`Suspended`. После миграции
и проверки runtime меняет только затронутые компоненты: команды, подтверждения
и микрофон не пересоздают STT; новый STT worker нужен только при изменении его
threads/queue. Файл заменяется атомарно, параметры pipe требуют перезапуска daemon.

## Команды и безопасность

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

Распознанный текст никогда не исполняется как `cmd.exe` или PowerShell.
Обработчик получает только проверенный `CommandConfig`. Для команд с
подтверждением событие содержит одноразовый `confirmation_id`; именно его надо
передать в `confirm`/`cancel`.

Подтверждения включены по умолчанию. Явное полное отключение, включая команды
высокого риска:

```toml
[policy]
confirmations_enabled = false
confirmation_timeout_ms = 15000
```

Это осознанное снижение защиты и должно включаться приложением явно.

## Модели

Production-модель:
[`alphacep/vosk-model-streaming-ru`](https://huggingface.co/alphacep/vosk-model-streaming-ru),
revision `83bbf6f40059480e96251be8aed3d32bc7c80c33`, Apache-2.0.

```text
models\stt-ru-streaming\<revision>\
├── manifest.toml
├── am-onnx\
└── lang\
```

Скачиваются только allowlisted-файлы. Установка идёт во временный каталог,
проверяется и активируется rename. Повреждённая активная ревизия временно
изолируется и восстанавливается, если новая установка не завершилась.

## Диагностика аудио

```powershell
cargo run -p assistant-cli -- record --seconds 5 --output command.wav
cargo run -p assistant-cli -- transcribe command.wav
cargo run -p assistant-cli -- test-wake-word command.wav
cargo run -p assistant-cli -- evaluate command.wav
cargo run -p assistant-cli -- soak --seconds 3600
```

`evaluate` печатает KWS и полный transcript. `soak` следит за зависанием
callback, reconnect, потерями bounded-очереди и метриками процесса.

Реальный WAV не хранится в Git. Повторяемый regression test запускается с
фиксированным внешним файлом:

```powershell
$env:VOICE_ASSISTANT_TEST_WAV = "C:\fixtures\assistant-open-notepad.wav"
$env:VOICE_ASSISTANT_TEST_MODEL = "$env:LOCALAPPDATA\VoiceAssistantCore\models\stt-ru-streaming\83bbf6f40059480e96251be8aed3d32bc7c80c33"
$env:VOICE_ASSISTANT_TEST_TRANSCRIPT = "ассистент открой блокнот"
cargo test -p assistant-core --all-features --test real_audio -- --ignored
```

Файл должен быть неизменным mono WAV; ожидаемый transcript фиксируется
переменной окружения.

## Документация и проверки

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
$env:RUSTDOCFLAGS = "-D warnings"
cargo doc --workspace --all-features --no-deps
```

`cargo test --workspace --all-features` запускает также процессный E2E-тест:
реальный `assistant-daemon` поднимается с временной конфигурацией, IPC-клиент
проверяет health, suspend/resume и корректное удалённое завершение. Для MSVC
workspace использует единый статический CRT, совместимый с Windows-сборкой
`sherpa-onnx`; поэтому дополнительных `/NODEFAULTLIB` флагов не требуется.

Полный справочник для разработчика:
[онлайн-документация](https://byxarek.github.io/voice-assistant-core/) или
[docs/api.html](docs/api.html). Локальный Rustdoc открывается из
`target\doc\assistant_core\index.html`. Руководство:
[docs/index.html](docs/index.html). Модели, WAV, логи и секреты не добавляются
в Git.
