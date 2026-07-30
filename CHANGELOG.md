# Changelog

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
