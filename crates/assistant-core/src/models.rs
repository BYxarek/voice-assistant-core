use std::{
    fs,
    io::Read,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use hf_hub::{
    Repo, RepoType,
    api::sync::{ApiBuilder, ApiError, ApiRepo},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Pinned Hugging Face repository for the stock Russian streaming model.
pub const ALPHACEP_STREAMING_RU_REPO: &str = "alphacep/vosk-model-streaming-ru";
/// Pinned immutable commit SHA for the stock model.
pub const ALPHACEP_STREAMING_RU_REVISION: &str = "83bbf6f40059480e96251be8aed3d32bc7c80c33";
/// Complete allowlist required by the stock model.
pub const ALPHACEP_STREAMING_RU_FILES: &[&str] = &[
    "am-onnx/encoder.int8.onnx",
    "am-onnx/decoder.int8.onnx",
    "am-onnx/joiner.int8.onnx",
    "lang/tokens.txt",
    "lang/bpe.model",
];

const SMALL_STREAMING_RU_FILES: &[&str] = ALPHACEP_STREAMING_RU_FILES;
const SMALL_STREAMING_BN_FILES: &[&str] = &[
    "am-onnx/encoder.onnx",
    "am-onnx/decoder.onnx",
    "am-onnx/joiner.onnx",
    "lang/tokens.txt",
    "lang/bpe.model",
];
const OFFLINE_RU_FILES: &[&str] = ALPHACEP_STREAMING_RU_FILES;
const OFFLINE_SMALL_RU_FILES: &[&str] = &[
    "am/encoder.int8.onnx",
    "am/decoder.int8.onnx",
    "am/joiner.int8.onnx",
    "lang/tokens.txt",
    "lang/bpe.model",
];
const OFFLINE_TG_UZ_FILES: &[&str] = &[
    "am-onnx/encoder.onnx",
    "am-onnx/decoder.onnx",
    "am-onnx/joiner.onnx",
    "lang/tokens.txt",
];

/// Native sherpa-onnx execution mode required by a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionMode {
    /// Incremental recognition while audio is captured.
    Streaming,
    /// Recognition after the complete utterance is available.
    FinalOnly,
}

/// Immutable allowlisted description of one supported Alphacep model.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    id: &'static str,
    revision: &'static str,
    language: &'static str,
    mode: RecognitionMode,
    files: &'static [&'static str],
}

impl ModelSpec {
    /// Hugging Face repository identifier and stable catalog key.
    pub const fn id(self) -> &'static str {
        self.id
    }

    /// Pinned immutable Hub commit SHA.
    pub const fn revision(self) -> &'static str {
        self.revision
    }

    /// BCP-47-style language code published by the model.
    pub const fn language(self) -> &'static str {
        self.language
    }

    /// Required recognizer execution mode.
    pub const fn mode(self) -> RecognitionMode {
        self.mode
    }

    /// Complete file allowlist used for installation and verification.
    pub const fn files(self) -> &'static [&'static str] {
        self.files
    }

    /// Whether the model includes the SentencePiece data required by keyword spotting.
    pub fn supports_wake_word(self) -> bool {
        self.files.contains(&"lang/bpe.model") && self.mode == RecognitionMode::Streaming
    }

    /// Returns the pinned encoder, decoder, joiner and token paths.
    pub fn inference_files(self) -> (&'static str, &'static str, &'static str, &'static str) {
        (
            self.files[0],
            self.files[1],
            self.files[2],
            "lang/tokens.txt",
        )
    }

    const fn new(
        id: &'static str,
        revision: &'static str,
        language: &'static str,
        mode: RecognitionMode,
        files: &'static [&'static str],
    ) -> Self {
        Self {
            id,
            revision,
            language,
            mode,
            files,
        }
    }
}

/// All currently supported pinned Alphacep STT models.
pub const ALPHACEP_MODELS: &[ModelSpec] = &[
    ModelSpec::new(
        ALPHACEP_STREAMING_RU_REPO,
        ALPHACEP_STREAMING_RU_REVISION,
        "ru",
        RecognitionMode::Streaming,
        ALPHACEP_STREAMING_RU_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-small-streaming-ru",
        "e18123ee13f694036a1eea82eb43f9895387cb59",
        "ru",
        RecognitionMode::Streaming,
        SMALL_STREAMING_RU_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-small-streaming-bn",
        "501097ae5257e5859d7956b50e4ba53a3f2be106",
        "bn",
        RecognitionMode::Streaming,
        SMALL_STREAMING_BN_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-ru",
        "df6a54a4d8e5d43e82675e4f5dba2d507731a0d1",
        "ru",
        RecognitionMode::FinalOnly,
        OFFLINE_RU_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-small-ru",
        "4d68c4017bcfa44e2a79581f7933339e916a35da",
        "ru",
        RecognitionMode::FinalOnly,
        OFFLINE_SMALL_RU_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-tg",
        "b4900abb39cad697d97d6091a262bbca41fab49c",
        "tg",
        RecognitionMode::FinalOnly,
        OFFLINE_TG_UZ_FILES,
    ),
    ModelSpec::new(
        "alphacep/vosk-model-small-streaming-uz",
        "e0417cabfbbac4efdc34e7aba367501d126ae79c",
        "uz",
        RecognitionMode::FinalOnly,
        OFFLINE_TG_UZ_FILES,
    ),
];

/// Resolves a user-facing catalog identifier without accessing the network.
pub fn model_spec(id: &str) -> Result<ModelSpec, ModelError> {
    ALPHACEP_MODELS
        .iter()
        .copied()
        .find(|model| model.id == id)
        .ok_or_else(|| ModelError::Manifest(format!("unsupported model: {id}")))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// One verified file in a model manifest.
pub struct ModelFile {
    /// Safe relative path below the revision directory.
    pub path: String,
    /// Expected byte length.
    pub size: u64,
    /// Lowercase SHA-256 digest.
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Offline verification metadata stored beside one model revision.
pub struct ModelManifest {
    /// Manifest format version.
    pub schema_version: u16,
    /// Core model identifier.
    pub model_id: String,
    /// Source repository.
    pub repo_id: String,
    /// Immutable source commit SHA.
    pub revision: String,
    /// Required inference runtime.
    pub runtime: String,
    /// Model language code.
    pub language: String,
    /// Required audio sample rate.
    pub sample_rate: u32,
    /// Upstream model license identifier.
    pub license: String,
    /// Verified allowlisted files.
    pub files: Vec<ModelFile>,
}

/// Bounded install progress emitted between allowlisted files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInstallProgress {
    /// Files copied and hashed so far.
    pub completed_files: usize,
    /// Total allowlisted files.
    pub total_files: usize,
    /// Current relative path, if any.
    pub file: Option<String>,
}

#[derive(Debug, Error)]
/// Model source, manifest and verification failures.
pub enum ModelError {
    /// Local filesystem operation failed.
    #[error("model I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// Pinned repository access failed.
    #[error("Hugging Face failed: {0}")]
    Hub(String),
    /// Manifest serialization or metadata failed.
    #[error("model manifest failed: {0}")]
    Manifest(String),
    /// File size, digest, path or required metadata did not verify.
    #[error("model verification failed: {0}")]
    Verification(String),
    /// Cooperative installation cancellation was observed.
    #[error("model installation cancelled")]
    Cancelled,
}

#[derive(Clone)]
/// Installs and verifies pinned catalog models below a caller-selected root.
pub struct ModelManager {
    root: PathBuf,
}

impl ModelManager {
    /// Creates a manager without accessing disk or network.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Installs or returns an already verified pinned catalog revision.
    pub fn install(&self, model_id: &str) -> Result<PathBuf, ModelError> {
        self.install_with_control(model_id, &AtomicBool::new(false), |_| {})
    }

    /// Installs a catalog model with cooperative cancellation and per-file progress.
    pub fn install_with_control(
        &self,
        model_id: &str,
        cancelled: &AtomicBool,
        mut progress: impl FnMut(ModelInstallProgress),
    ) -> Result<PathBuf, ModelError> {
        let spec = model_spec(model_id)?;
        if let Ok(directory) = self.resolve(model_id) {
            progress(ModelInstallProgress {
                completed_files: spec.files().len(),
                total_files: spec.files().len(),
                file: None,
            });
            return Ok(directory);
        }
        let model_root = self.root.join(storage_name(spec));
        let destination = model_root.join(spec.revision());
        fs::create_dir_all(&model_root)?;
        preflight_install(&model_root)?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| ModelError::Manifest(e.to_string()))?
            .as_nanos();
        let quarantine = model_root.join(format!(".invalid-{nonce}"));
        let quarantined = if destination.exists() {
            match self.verify(&destination) {
                Ok(_) => {
                    progress(ModelInstallProgress {
                        completed_files: spec.files().len(),
                        total_files: spec.files().len(),
                        file: None,
                    });
                    return Ok(destination);
                }
                Err(error) => {
                    tracing::warn!(%error, "quarantining invalid installed model");
                    fs::rename(&destination, &quarantine)?;
                    true
                }
            }
        } else {
            false
        };
        let temporary = self
            .root
            .join(format!(".model-{}-{nonce}", std::process::id()));
        fs::create_dir(&temporary)?;

        let result = (|| {
            let hub_cache = temporary.join(".hub-cache");
            let api = ApiBuilder::new()
                .with_cache_dir(hub_cache.clone())
                .with_progress(true)
                .build()
                .map_err(|e| ModelError::Hub(e.to_string()))?;
            let repo = api.repo(Repo::with_revision(
                spec.id().into(),
                RepoType::Model,
                spec.revision().into(),
            ));
            let files = copy_model_files(spec, &repo, &temporary, cancelled, &mut progress)?;
            fs::remove_dir_all(hub_cache)?;
            let manifest = ModelManifest {
                schema_version: 1,
                model_id: spec.id().into(),
                repo_id: spec.id().into(),
                revision: spec.revision().into(),
                runtime: "sherpa-onnx".into(),
                language: spec.language().into(),
                sample_rate: 16_000,
                license: "Apache-2.0".into(),
                files,
            };
            fs::write(
                temporary.join("manifest.toml"),
                toml::to_string_pretty(&manifest)
                    .map_err(|e| ModelError::Manifest(e.to_string()))?,
            )?;
            self.verify(&temporary)?;
            if cancelled.load(Ordering::Acquire) {
                return Err(ModelError::Cancelled);
            }
            fs::rename(&temporary, &destination)?;
            if quarantined {
                fs::remove_dir_all(&quarantine)?;
            }
            Ok(destination.clone())
        })();

        if result.is_err() && temporary.starts_with(&self.root) {
            let _ = fs::remove_dir_all(&temporary);
        }
        if result.is_err() && quarantined && !destination.exists() {
            let _ = fs::rename(&quarantine, &destination);
        }
        result
    }

    /// Resolves and verifies an installed catalog revision without network access.
    pub fn resolve(&self, model_id: &str) -> Result<PathBuf, ModelError> {
        let spec = model_spec(model_id)?;
        let directory = self.root.join(storage_name(spec)).join(spec.revision());
        let directory = if directory.is_dir() {
            directory
        } else if spec.id() == ALPHACEP_STREAMING_RU_REPO {
            self.root.join("stt-ru-streaming").join(spec.revision())
        } else {
            directory
        };
        self.verify(&directory)?;
        Ok(directory)
    }

    /// Verifies manifest metadata, safe paths, sizes and SHA-256 digests.
    pub fn verify(&self, directory: impl AsRef<Path>) -> Result<ModelManifest, ModelError> {
        let directory = directory.as_ref();
        let manifest: ModelManifest =
            toml::from_str(&fs::read_to_string(directory.join("manifest.toml"))?)
                .map_err(|e| ModelError::Manifest(e.to_string()))?;
        let spec = model_spec(&manifest.repo_id)?;
        if manifest.schema_version != 1
            || (manifest.model_id != spec.id()
                && !(spec.id() == ALPHACEP_STREAMING_RU_REPO
                    && manifest.model_id == "stt-ru-streaming"))
            || manifest.repo_id != spec.id()
            || manifest.revision != spec.revision()
            || manifest.runtime != "sherpa-onnx"
            || manifest.language != spec.language()
            || manifest.sample_rate != 16_000
            || manifest.license != "Apache-2.0"
        {
            return Err(ModelError::Verification(
                "unexpected model manifest metadata".into(),
            ));
        }
        let mut paths = std::collections::HashSet::new();
        if manifest.files.iter().any(|file| {
            Path::new(&file.path)
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
                || !paths.insert(file.path.as_str())
        }) {
            return Err(ModelError::Verification(
                "manifest contains an unsafe or duplicate path".into(),
            ));
        }
        if spec
            .files()
            .iter()
            .any(|required| !manifest.files.iter().any(|file| file.path == *required))
        {
            return Err(ModelError::Verification(
                "manifest does not contain every required file".into(),
            ));
        }
        for file in &manifest.files {
            let path = directory.join(&file.path);
            let metadata = fs::metadata(&path)?;
            if metadata.len() != file.size || digest(&path)? != file.sha256 {
                return Err(ModelError::Verification(file.path.clone()));
            }
        }
        Ok(manifest)
    }
}

fn storage_name(spec: ModelSpec) -> &'static str {
    spec.id()
        .split_once('/')
        .map_or(spec.id(), |(_, name)| name)
}

const MODEL_INSTALL_MIN_FREE_BYTES: u64 = 256 * 1024 * 1024;

fn preflight_install(directory: &Path) -> Result<(), ModelError> {
    let probe = directory.join(format!(".preflight-{}", std::process::id()));
    let renamed = directory.join(format!(".preflight-{}-renamed", std::process::id()));
    fs::write(&probe, b"voice-assistant-core")?;
    let result = fs::rename(&probe, &renamed).and_then(|()| fs::remove_file(&renamed));
    if result.is_err() {
        let _ = fs::remove_file(&probe);
        let _ = fs::remove_file(&renamed);
    }
    result?;
    if available_space(directory)? < MODEL_INSTALL_MIN_FREE_BYTES {
        return Err(ModelError::Verification(format!(
            "model installation requires at least {} MiB free",
            MODEL_INSTALL_MIN_FREE_BYTES / 1024 / 1024
        )));
    }
    Ok(())
}

#[cfg(windows)]
fn available_space(directory: &Path) -> Result<u64, ModelError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let wide: Vec<u16> = directory.as_os_str().encode_wide().chain([0]).collect();
    let mut available = 0_u64;
    // SAFETY: `wide` is a live NUL-terminated UTF-16 path and `available` is writable.
    if unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(ModelError::Io(std::io::Error::last_os_error()));
    }
    Ok(available)
}

#[cfg(not(windows))]
fn available_space(_directory: &Path) -> Result<u64, ModelError> {
    Ok(u64::MAX)
}

fn copy_model_files(
    spec: ModelSpec,
    repo: &ApiRepo,
    destination: &Path,
    cancelled: &AtomicBool,
    progress: &mut impl FnMut(ModelInstallProgress),
) -> Result<Vec<ModelFile>, ModelError> {
    let mut files = Vec::with_capacity(spec.files().len());
    for (index, relative) in spec.files().iter().enumerate() {
        if cancelled.load(Ordering::Acquire) {
            return Err(ModelError::Cancelled);
        }
        let source = model_source(relative, || repo.get(relative))?;
        let target = destination.join(relative);
        let parent = target
            .parent()
            .ok_or_else(|| ModelError::Manifest("model file has no parent".into()))?;
        fs::create_dir_all(parent)?;
        fs::copy(&source, &target)?;
        files.push(ModelFile {
            path: (*relative).into(),
            size: fs::metadata(&target)?.len(),
            sha256: digest(&target)?,
        });
        progress(ModelInstallProgress {
            completed_files: index + 1,
            total_files: spec.files().len(),
            file: Some((*relative).into()),
        });
    }
    Ok(files)
}

fn model_source<T>(
    relative: &str,
    operation: impl FnOnce() -> Result<T, ApiError>,
) -> Result<T, ModelError> {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result.map_err(|error| ModelError::Hub(error.to_string())),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("non-string panic");
            Err(ModelError::Hub(format!(
                "model source failed to prepare {relative}: {detail}"
            )))
        }
    }
}

fn digest(path: &Path) -> Result<String, ModelError> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_detects_corrupted_model_file() {
        let directory = std::env::temp_dir().join(format!(
            "assistant-model-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let mut files = Vec::new();
        for relative in ALPHACEP_STREAMING_RU_FILES {
            let path = directory.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, relative.as_bytes()).unwrap();
            files.push(ModelFile {
                path: (*relative).into(),
                size: fs::metadata(&path).unwrap().len(),
                sha256: digest(&path).unwrap(),
            });
        }
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: ALPHACEP_STREAMING_RU_REPO.into(),
            repo_id: ALPHACEP_STREAMING_RU_REPO.into(),
            revision: ALPHACEP_STREAMING_RU_REVISION.into(),
            runtime: "sherpa-onnx".into(),
            language: "ru".into(),
            sample_rate: 16_000,
            license: "Apache-2.0".into(),
            files,
        };
        fs::write(
            directory.join("manifest.toml"),
            toml::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let manager = ModelManager::new(directory.join("unused"));
        manager.verify(&directory).unwrap();
        fs::write(directory.join(ALPHACEP_STREAMING_RU_FILES[0]), b"corrupt").unwrap();
        assert!(matches!(
            manager.verify(&directory),
            Err(ModelError::Verification(_))
        ));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn catalog_covers_streaming_final_only_and_path_variants() {
        assert_eq!(ALPHACEP_MODELS.len(), 7);
        assert!(
            model_spec("alphacep/vosk-model-small-streaming-bn")
                .unwrap()
                .supports_wake_word()
        );
        let small_ru = model_spec("alphacep/vosk-model-small-ru").unwrap();
        assert_eq!(small_ru.mode(), RecognitionMode::FinalOnly);
        assert_eq!(small_ru.inference_files().0, "am/encoder.int8.onnx");
        assert!(
            !model_spec("alphacep/vosk-model-tg")
                .unwrap()
                .supports_wake_word()
        );
    }

    #[test]
    fn model_source_panic_becomes_a_typed_error() {
        let error =
            model_source::<()>("am/model.onnx", || panic!("upstream cache assertion")).unwrap_err();
        assert!(matches!(error, ModelError::Hub(message)
                if message.contains("am/model.onnx")
                    && message.contains("upstream cache assertion")));
    }
}
