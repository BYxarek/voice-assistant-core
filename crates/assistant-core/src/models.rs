use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use hf_hub::{
    Repo, RepoType,
    api::sync::{ApiBuilder, ApiRepo},
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
/// Installs and verifies the stock model below a caller-selected root.
pub struct ModelManager {
    root: PathBuf,
}

impl ModelManager {
    /// Creates a manager without accessing disk or network.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Installs or returns the already verified pinned revision.
    pub fn install_alphacep_streaming_ru(&self) -> Result<PathBuf, ModelError> {
        self.install_alphacep_streaming_ru_with_control(&AtomicBool::new(false), |_| {})
    }

    /// Installs with cooperative cancellation and per-file progress.
    pub fn install_alphacep_streaming_ru_with_control(
        &self,
        cancelled: &AtomicBool,
        mut progress: impl FnMut(ModelInstallProgress),
    ) -> Result<PathBuf, ModelError> {
        let model_root = self.root.join("stt-ru-streaming");
        let destination = model_root.join(ALPHACEP_STREAMING_RU_REVISION);
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
                        completed_files: ALPHACEP_STREAMING_RU_FILES.len(),
                        total_files: ALPHACEP_STREAMING_RU_FILES.len(),
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
            .join(format!(".stt-ru-streaming-{}-{nonce}", std::process::id()));
        fs::create_dir(&temporary)?;

        let result = (|| {
            let api = ApiBuilder::new()
                .with_progress(true)
                .build()
                .map_err(|e| ModelError::Hub(e.to_string()))?;
            let repo = api.repo(Repo::with_revision(
                ALPHACEP_STREAMING_RU_REPO.into(),
                RepoType::Model,
                ALPHACEP_STREAMING_RU_REVISION.into(),
            ));
            let files = copy_model_files(&repo, &temporary, cancelled, &mut progress)?;
            let manifest = ModelManifest {
                schema_version: 1,
                model_id: "stt-ru-streaming".into(),
                repo_id: ALPHACEP_STREAMING_RU_REPO.into(),
                revision: ALPHACEP_STREAMING_RU_REVISION.into(),
                runtime: "sherpa-onnx".into(),
                language: "ru".into(),
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

    /// Resolves and verifies the installed pinned revision without network access.
    pub fn resolve_alphacep_streaming_ru(&self) -> Result<PathBuf, ModelError> {
        let directory = self
            .root
            .join("stt-ru-streaming")
            .join(ALPHACEP_STREAMING_RU_REVISION);
        self.verify(&directory)?;
        Ok(directory)
    }

    /// Verifies manifest metadata, safe paths, sizes and SHA-256 digests.
    pub fn verify(&self, directory: impl AsRef<Path>) -> Result<ModelManifest, ModelError> {
        let directory = directory.as_ref();
        let manifest: ModelManifest =
            toml::from_str(&fs::read_to_string(directory.join("manifest.toml"))?)
                .map_err(|e| ModelError::Manifest(e.to_string()))?;
        if manifest.schema_version != 1
            || manifest.model_id != "stt-ru-streaming"
            || manifest.repo_id != ALPHACEP_STREAMING_RU_REPO
            || manifest.revision != ALPHACEP_STREAMING_RU_REVISION
            || manifest.runtime != "sherpa-onnx"
            || manifest.language != "ru"
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
        if ALPHACEP_STREAMING_RU_FILES
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
    repo: &ApiRepo,
    destination: &Path,
    cancelled: &AtomicBool,
    progress: &mut impl FnMut(ModelInstallProgress),
) -> Result<Vec<ModelFile>, ModelError> {
    let mut files = Vec::with_capacity(ALPHACEP_STREAMING_RU_FILES.len());
    for (index, relative) in ALPHACEP_STREAMING_RU_FILES.iter().enumerate() {
        if cancelled.load(Ordering::Acquire) {
            return Err(ModelError::Cancelled);
        }
        let source = repo
            .get(relative)
            .map_err(|e| ModelError::Hub(e.to_string()))?;
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
            total_files: ALPHACEP_STREAMING_RU_FILES.len(),
            file: Some((*relative).into()),
        });
    }
    Ok(files)
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
            model_id: "stt-ru-streaming".into(),
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
}
