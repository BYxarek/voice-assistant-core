# Changelog

## [1.5.1] — 2026-08-03

### Добавлено

- IPC-режим непрерывного распознавания с VAD-сегментацией, pre-roll и событиями
  `speech_started`/`speech_ended`;
- общий `session_id` для `transcript_partial` и `transcript_final`.

### Изменено

- транскрипция отделена от command matching: речь без wake word публикуется, но не
  запускает handler;
- режимы моделей переименованы из `Online`/`Offline` в `Streaming`/`FinalOnly`;
- публичный Rust API увеличен до v6, IPC wire protocol — до v7.

### Исправлено

- тишина определяется по речевым VAD-кадрам без занижения RMS хвостом паузы;
- установка модели использует изолированный временный Hub cache, а panic источника
  преобразуется в `ModelError::Hub` вместо завершения background task.

## [1.5.0] — 2026-08-03

### Производительность

- audio worker пропускает resample/VAD/KWS в `Starting` и `Suspended`, переиспользует
  frame/pre-roll буферы и передаёт streaming STT 100-мс батчами;
- одна модель, выбранная одновременно для STT и KWS, проходит SHA-256 resolve один раз;
- `assistant-cli evaluate --threads` выводит STT latency и real-time factor для подбора
  числа native inference threads на фиксированном WAV.

### Добавлено

- изменившиеся partial-транскрипты streaming STT в IPC events;
- однословные типизированные command slots с bounds/allowlist и безопасной подстановкой;
- голосовое подтверждение и отмена pending-команд после повторного wake word;
- общий per-handler rate limit и circuit breaker последовательных ошибок;
- документация интеграции Web Speech API через `submit_text`;
- английская локализация README с переключателем языка.

### Изменено

- конфигурация увеличена до schema v6, публичный Rust API — до v5, IPC — до
  protocol v6.

## [1.4.0] — 2026-08-02

### Добавлено

- встроенные handlers `open_url`, `set_volume` и high-risk `click_mouse` с плавным
  перемещением курсора;
- последовательные `commands.actions` с задержкой перед каждым действием.

### Изменено

- конфигурация увеличена до schema v5, публичный Rust API — до v4, IPC — до
  protocol v5;
- открытие URL вынесено из runtime-задачи, чтобы зависший Windows shell не
  блокировал daemon.

## [1.3.0] — 2026-08-01

### Добавлено

- pinned-каталог семи STT-моделей `alphacep/vosk-model*` с языком, режимом и
  точными allowlist-путями ONNX-файлов;
- online/offline sherpa-onnx worker за общим `SpeechRecognizer`;
- команды `models list` и `models install [model-id]`;
- отдельный выбор STT и совместимой wake-word модели.

### Изменено

- `ModelManager` устанавливает и проверяет любую модель встроенного каталога;
- версия конфигурации увеличена до schema v4; schema 1–3 мигрируются
  автоматически;
- старый каталог установленной stock-модели продолжает разрешаться без
  повторной загрузки.

## [1.2.0] — 2026-07-31

### Добавлено

- несколько wake-word aliases в одном streaming KWS detector;
- IPC protocol v4 с `list_handlers` и component-level health;
- ранняя потоковая STT-сессия: декодирование начинается сразу после wake word;
- warm-up STT при загрузке и автоматический выбор inference threads при `threads = 0`;
- watchdog STT worker, exponential backoff, ограниченный restart budget и fault metrics;
- audio supervisor с exponential backoff и circuit breaker после повторных сбоев;
- preflight установки модели: проверка write/rename и свободного места.

### Изменено

- версия конфигурации увеличена до schema v3;
- публичный Rust extension API увеличен до v3, IPC — до protocol v4.

## [1.1.1] — 2026-07-31

### Добавлено

- IPC protocol v3: `audio_level`, причины `transcript_unavailable` и предупреждение
  `audio_device_fallback` при временной замене исчезнувшего выбранного микрофона.

### Изменено

- пустой `audio.device_id` отклоняется при валидации конфигурации;
- публичный Rust extension API увеличен до v2 из-за новых вариантов событий.

## [1.1.0] — 2026-07-30

### Добавлено

- IPC-запросы `begin_capture`, `end_capture` и `submit_text` для push-to-talk
  и текстового ввода через существующую command policy;
- `HealthSnapshot.active_audio_device` и событие `audio_device_changed`;
- автоматическое переоткрытие потока при смене default input device Windows.

### Изменено

- Windows x64 release ZIP включает runtime-DLL и проверяется запуском daemon/CLI
  из распакованного архива.

## [1.0.1] — 2026-07-30

### Изменено

- IPC protocol v2 сообщает `Starting` и прогресс blocking-загрузки STT;
- `ApplyConfig` дифференциально меняет только затронутые runtime-компоненты;
- загрузка и замена STT поддерживают кооперативную отмену и отзывчивый shutdown.

Все заметные изменения Voice Assistant Core документируются здесь. Проект
следует [Semantic Versioning](https://semver.org/lang/ru/).

## [1.0.0] — 2026-07-30

Первый стабильный публичный релиз.

### Добавлено

- стабильный Rust extension API v1 и versioned Windows Named Pipe IPC v1;
- независимый runtime со сменными STT, wake-word и command adapters;
- bounded audio/VAD/KWS/STT pipeline и восстановление аудиоустройства;
- типизированные command handlers, allowlist, timeout и подтверждения;
- установка закреплённой ревизии модели с SHA-256 и атомарной активацией;
- health, events, metrics, configuration и model lifecycle через IPC;
- CLI для диагностики, WAV-проверок и soak-тестов;
- автоматический процессный E2E daemon ↔ IPC client;
- CI, Rustdoc и GitHub Pages документация.

[1.0.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.0.0
[1.0.1]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.0.1
[1.1.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.1.0
[1.1.1]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.1.1
[1.2.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.2.0
[1.3.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.3.0
[1.4.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.4.0
[1.5.0]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.5.0
[1.5.1]: https://github.com/BYxarek/voice-assistant-core/releases/tag/v1.5.1
