# Voice Assistant Core

[Русский](README.md) | [English](README.en.md)

[![CI](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml/badge.svg)](https://github.com/BYxarek/voice-assistant-core/actions/workflows/ci.yml)
[![Documentation](https://github.com/BYxarek/voice-assistant-core/actions/workflows/docs.yml/badge.svg)](https://byxarek.github.io/voice-assistant-core/)
[![Release](https://img.shields.io/github/v/release/BYxarek/voice-assistant-core)](https://github.com/BYxarek/voice-assistant-core/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BYxarek/voice-assistant-core/blob/main/LICENSE)

Локальное ядро голосового ассистента для Windows 10/11 на Rust. Ядро принимает
звук с микрофона, обнаруживает ключевую фразу, распознаёт русскую речь и
выполняет только зарегистрированные типизированные команды.

Текущий стабильный релиз — **1.5.1**, предрелиз — **1.5.2-rc.1**. Публичный Rust extension API v6 и IPC
protocol v7. Форматы аудио, очереди и
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

- bounded audio pipeline: CPAL → mono → 16 кГц → KWS/VAD → STT;
- при `Starting`/`Suspended` worker осушает bounded input без resample, VAD и inference;
- отдельная задача runtime и supervised persistent STT worker с warm-up, timeout watchdog,
  bounded restart budget и exponential backoff;
- штатный continuous-режим через IPC сегментирует любую речь в ядре, сохраняет pre-roll
  и публикует связанные по `session_id` partial/final-транскрипты;
- обычная речь не попадает в command matching: действие разрешают wake word,
  ручной push-to-talk/`submit_text` или одноразовое подтверждение;
- IPC доступен в состоянии `Starting`, пока cancellable blocking worker загружает STT и публикует прогресс;
- восстановление микрофона после ошибки или зависания callback и автоматическое
  переключение при смене default input;
- ручной push-to-talk и безопасная отправка текста через IPC без обхода command policy;
- типизированные обработчики команд без передачи текста в shell;
- одноразовый `confirmation_id`, голосовое «да/нет», timeout и полное отключение подтверждений;
- типизированные allowlisted-слоты команд, rate limit и circuit breaker handlers;
- запуск без модели, установка/отмена/проверка модели во время работы;
- pinned model revision, SHA-256 manifest и атомарная активация;
- дифференциальное применение конфигурации без перезагрузки незатронутых компонентов;
- локальный versioned Named Pipe IPC и готовый Rust-клиент;
- несколько wake-word aliases и IPC-discovery схем зарегистрированных command handlers;
- health, состояние модели, события, очереди, задержки, CPU time и RAM через IPC;
- component health для audio, STT, runtime, model и IPC, включая restart count и fault state;
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

При каждом запуске daemon создаёт или дополняет `assistant-daemon.log` рядом с
`assistant-daemon.exe`. Журнал содержит предупреждения и ошибки всех компонентов,
внутренние диагностические события ядра, исходные цепочки ошибок, место в коде,
поток и backtrace паники. Аудио и полный распознанный текст в журнал не записываются.

Rust и Cargo для готового релиза не требуются. Для разработки самого ядра:

```powershell
cargo run -p assistant-cli -- validate-config
cargo run -p assistant-cli -- devices
cargo run -p assistant-cli -- models list
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

## Публичный Rust API v6

Стабильная граница экспорта находится в корне crate `assistant_core`.
`CORE_API_VERSION` равен `6`. В v6 входят:

- `CommandHandler`, `HandlerSchema`, `HandlerRegistry` — extension API команд;
- `RuntimeComponents`, `RuntimeUpdate`, `RuntimeHandle`, `RuntimeTask`,
  `spawn_runtime_service` — embedding API;
- `CoreIpcClient`, `EventSubscription`, `CoreRequest`, `CoreResponse`,
  `Envelope`, `IpcErrorCode` — IPC API;
- `CoreConfig`, `CommandConfig`, `SlotConfig`, публичные доменные события, состояния и ошибки;
- streaming `SpeechRecognizer::push_stream_partial`;
- VAD-сессии `SpeechStarted`/`SpeechEnded` и `TranscriptPartial`/`TranscriptFinal`;
- `CoreMetrics`, `MetricsSnapshot`.

Ломающие изменения этих контрактов требуют нового major crate API и увеличения
`CORE_API_VERSION`. IPC меняется только совместимо внутри protocol v7; для
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

## IPC protocol v7

### Web Speech API как внешний STT

Web Speech API можно подключить без изменения ядра и IPC-протокола. Браузерный
`SpeechRecognition` распознаёт речь, а GUI передаёт только финальный текст в уже
существующий запрос `submit_text`:

```text
SpeechRecognition → native bridge GUI → Windows named pipe
                  → CoreRequest::SubmitText → команды и policy ядра
```

Обычная веб-страница не может напрямую открыть Windows named pipe. В WebView2,
Tauri или Electron нужен узкий native bridge, который предоставляет только
метод `submitText(text)` и не даёт web-контенту доступ к shell или произвольным
IPC-запросам.

```javascript
const Recognition =
  window.SpeechRecognition ?? window.webkitSpeechRecognition;

if (!Recognition) {
  throw new Error("Web Speech API недоступен в этом браузере");
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
      renderPartial(text); // Только отображение, без выполнения команды.
    }
  }
};

recognition.onerror = (event) => renderRecognitionError(event.error);
document.querySelector("#listen").addEventListener("click", () => {
  recognition.start();
});
```

Native bridge кодирует вызов как обычный IPC-запрос:

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

Отправляйте в ядро только результаты с `isFinal === true`, иначе одна фраза
может выполнить команду несколько раз. `submit_text` принимает непустой UTF-8
текст размером до 4096 байт; дальше действуют обычные сопоставление команд,
allowlist, оценка риска, подтверждение и timeout. Значение confidence браузера
не является границей безопасности.

Поддержка Web Speech API зависит от браузера. В некоторых реализациях аудио
отправляется во внешний облачный сервис, поэтому офлайн-работа и приватность не
гарантируются. Запрашивайте доступ к микрофону из GUI по явному действию
пользователя, показывайте ошибки распознавания и оставляйте локальный
`sherpa-onnx` основным вариантом для офлайн-сценариев. Актуальную совместимость
проверяйте в документации [SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition)
и [`isFinal`](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognitionResult/isFinal).

Pipe по умолчанию: `\\.\pipe\voice-assistant-core`. Сервер допускает только
локальных клиентов, защищён DACL и разрешает один daemon на pipe. Каждый JSON
envelope имеет `protocol_version`, `request_id`, `payload` и little-endian
`u32`-длину; максимум 1 МиБ.

Запросы:

- `get_status`, `get_health`, `get_config`, `get_metrics`, `list_handlers`;
- `validate_config`, `apply_config`, `list_audio_devices`;
- `get_model_status`, `install_model`, `cancel_model_install`, `verify_model`;
- `suspend`, `resume`, `set_continuous_recognition`, `begin_capture`, `end_capture`, `submit_text`;
- `confirm`, `cancel`;
- `subscribe_events`, `shutdown`.

`begin_capture` запускает запись без wake word, а `end_capture` передаёт в STT
только аудио не короче `audio.command_min_ms`. `submit_text` принимает непустую
UTF-8 строку до 4096 байт, сопоставляет её с командой без обязательного wake-word
prefix и применяет те же risk, confirmation и timeout rules.

`set_continuous_recognition { enabled: true }` включает VAD-сегментацию в daemon:
ядро публикует `speech_started { session_id }`, `speech_ended { session_id }`,
`transcript_partial { session_id, text, confidence }` и
`transcript_final { session_id, text, confidence }`. Речь без wake word
только транскрибируется. Проверка тишины использует найденные VAD речевые
кадры, поэтому trailing silence не занижает энергию всей записи.

`CoreIpcClient` проверяет версию, ограничивает размер сообщения и повторяет
кратковременное подключение. `subscribe_events` создаёт отдельный
`EventSubscription`. Ошибки имеют стабильный `IpcErrorCode`.

`list_handlers` возвращает отсортированные `HandlerSchema` с обязательными и
необязательными параметрами. `HealthSnapshot.components` сообщает состояние
`ready/recovering/faulted/stopped` и число автоматических перезапусков.

Поток событий публикует изменившиеся `transcript_partial { session_id, text, confidence }`,
финальный `transcript_final { session_id, text, confidence }`, `audio_level { rms }` не чаще 10 раз в секунду и
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

```toml
[wake_word]
model = "alphacep/vosk-model-streaming-ru"
keyword = "ассистент"
aliases = ["помощник"]

[inference]
model = "alphacep/vosk-model-streaming-ru"
threads = 0 # автоматически по available_parallelism
max_restarts = 3
restart_backoff_ms = 500
```

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

Штатный daemon также регистрирует `open_url { url }`, `set_volume { level }` и
`click_mouse { x, y, duration_ms? }`. URL ограничен схемами `http`/`https`, уровень
громкости — целым числом `0..100`. `click_mouse` требует `risk = "high"`; курсор
перемещается к координатам плавно за 350 мс по умолчанию или за заданные
`100..10000` мс.

Одна команда может содержать до 32 последовательных `actions`. `delay_ms` задаёт
задержку перед шагом (не более 60000 мс), а `timeout_ms` ограничивает всю
последовательность:

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

`handler`/`parameters` и `actions` взаимоисключающие. Все значения фиксируются в
конфигурации и проверяются через allowlist; распознанный текст не становится
URL, координатой или уровнем громкости.

Однословные типизированные слоты задаются в шаблоне как `{name}`. Текстовый слот
обязан иметь allowlist `values`, целочисленный — границы `min`/`max`; после
подстановки значение повторно проверяется handler-ом:

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
voice_confirmations_enabled = true
confirmation_accept_phrases = ["да", "подтверждаю"]
confirmation_cancel_phrases = ["нет", "отмена"]
handler_rate_limit_ms = 500
handler_failure_threshold = 3
handler_circuit_breaker_ms = 30000
```

При ожидании подтверждения пользователь снова произносит wake word, затем одну
из accept/cancel-фраз. Rate limit действует между запусками одного handler;
после заданного числа последовательных ошибок circuit breaker временно отклоняет
его вызовы. `confirmations_enabled = false` — осознанное снижение защиты.

## Модели

Встроенный каталог содержит семь pinned STT-моделей организации
[`alphacep`](https://huggingface.co/alphacep/models): русские, бенгальскую,
таджикскую и узбекскую, в online и offline вариантах. Полный список:

```powershell
cargo run -p assistant-cli -- models list
cargo run -p assistant-cli -- models install alphacep/vosk-model-small-ru
```

Пример: offline STT и отдельная streaming-модель для wake word:

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

Установить обе модели, проверить конфигурацию и распознать WAV:

```powershell
cargo run -p assistant-cli -- models install alphacep/vosk-model-small-ru
cargo run -p assistant-cli -- models install alphacep/vosk-model-streaming-ru
cargo run -p assistant-cli -- validate-config
cargo run -p assistant-cli -- transcribe .\command.wav
```

`inference.model` выбирает STT, а `wake_word.model` — отдельную streaming-модель
KWS с `lang/bpe.model`. Модели `FinalOnly` и модели без SentencePiece доступны для
STT, но намеренно отклоняются как wake-word модель. Смена модели через
`ApplyConfig` требует перезапуска daemon; IPC `InstallModel` устанавливает обе
выбранные модели.

Загрузка использует изолированный временный Hugging Face cache для каждой установки,
поэтому параллельные Windows-процессы не повреждают snapshot pointers. Сбой или panic
источника модели возвращается как `ModelError::Hub`, не завершая background task.

```text
models\<repo-name>\<revision>\
├── manifest.toml
├── am-onnx\
└── lang\
```

Прямое использование каталога и recognizer из Rust:

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

Перед загрузкой проверяются запись/rename в каталоге и минимум 256 МиБ свободного места.
Скачиваются только allowlisted-файлы. Установка идёт во временный каталог,
проверяется и активируется rename. Повреждённая активная ревизия временно
изолируется и восстанавливается, если новая установка не завершилась.

## Диагностика аудио

```powershell
cargo run -p assistant-cli -- record --seconds 5 --output command.wav
cargo run -p assistant-cli -- transcribe command.wav
cargo run -p assistant-cli -- test-wake-word command.wav
cargo run -p assistant-cli -- evaluate command.wav --threads 4
cargo run -p assistant-cli -- soak --seconds 3600
cargo run -p assistant-cli -- handlers
```

`evaluate` печатает KWS, полный transcript, STT latency и real-time factor.
Для выбора `inference.threads` сравните один неизменный WAV:

```powershell
1, 2, 4, 8, 0 | ForEach-Object {
    cargo run --release -p assistant-cli -- evaluate command.wav --threads $_
}
```

`0` означает host parallelism; выбирайте минимальный стабильный real-time factor,
проверяя одинаковый transcript. `soak` следит за зависанием
callback, reconnect, потерями bounded-очереди и метриками процесса.

Реальный WAV не хранится в Git. Повторяемый regression test запускается с
фиксированным внешним файлом:

```powershell
$env:VOICE_ASSISTANT_TEST_WAV = "C:\fixtures\assistant-open-notepad.wav"
$env:VOICE_ASSISTANT_TEST_MODEL = "$env:LOCALAPPDATA\VoiceAssistantCore\models\vosk-model-streaming-ru\83bbf6f40059480e96251be8aed3d32bc7c80c33"
$env:VOICE_ASSISTANT_TEST_TRANSCRIPT = "ассистент открой блокнот"
cargo test -p assistant-core --all-features --test real_audio -- --ignored
```

Для проверки любой модели `Streaming`/`FinalOnly` из каталога отдельно задайте
`VOICE_ASSISTANT_TEST_STT_WAV`, `VOICE_ASSISTANT_TEST_STT_MODEL`,
`VOICE_ASSISTANT_TEST_STT_MODEL_ID` и `VOICE_ASSISTANT_TEST_STT_TRANSCRIPT`, затем
запустите ignored-тест `fixed_real_wav_transcribes_with_selected_catalog_model`.

Файл должен быть неизменным mono WAV; ожидаемый transcript фиксируется
переменной окружения.

Ручная проверка новых streaming-событий: запустите daemon с моделью `Streaming`,
включите `set_continuous_recognition`, подпишитесь IPC-клиентом на events и произнесите
обычную фразу, затем wake word с командой. Для каждой реплики должны совпадать
`session_id` в `speech_started`, `speech_ended`, `transcript_partial` и
`transcript_final`; первая реплика не запускает handler. Для high-risk команды после `confirmation_required`
повторите wake word и скажите «да»; handler должен стартовать ровно один раз.

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
