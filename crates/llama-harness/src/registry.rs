//! Centralized `models.toml` registry.
//!
//! Format (TOML):
//!
//! ```toml
//! [models.llama-3.1-8b-instruct]
//! path    = "C:/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
//! alias   = "llama8b"               # optional
//! family  = "llama-3.1"             # chat template dispatch
//! context = 8192                    # optional, default 4096
//! gpu_layers = 35                   # optional, default -1
//! sha256  = "..."                   # optional integrity check
//!
//! [models.qwen2.5-7b]
//! path   = "C:/models/qwen2.5-7b-instruct-q5_k_m.gguf"
//! family = "qwen2.5"
//! context = 32768
//! ```

use crate::error::{HarnessError, Result};
use crate::model::{ModelInfo, ModelLoadSpec};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Where a model can come from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelSource {
    /// Local file on disk.
    Local {
        /// Absolute path to the .gguf.
        path: PathBuf,
    },
    /// Hugging Face Hub repo.
    HuggingFace {
        /// repo id, e.g. "TheBloke/Llama-2-7B-Chat-GGUF"
        repo: String,
        /// File within the repo.
        file: String,
        /// Optional revision (branch/tag/commit).
        #[serde(default)]
        revision: Option<String>,
    },
    /// Ollama-style model reference.
    Ollama {
        /// model name as served by ollama
        name: String,
    },
}

/// One entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Display name (registry key).
    pub name: String,
    /// Optional shorter alias usable in CLI / chat commands.
    #[serde(default)]
    pub alias: Option<String>,
    /// Family used to pick the chat template. e.g. "llama-3.1".
    pub family: String,
    /// Where to get the model.
    #[serde(flatten)]
    pub source: ModelSource,
    /// Default context size override.
    #[serde(default)]
    pub context: Option<i32>,
    /// Default GPU-layer count override.
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    /// SHA256 for integrity check.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Human-readable description / tags.
    #[serde(default)]
    pub description: Option<String>,
}

impl ModelEntry {
    /// Resolve to a load spec. For `Local`, returns immediately.
    /// For remote sources, returns an error if the file is not yet
    /// cached locally; remote download is left to a future version.
    pub fn resolve(&self) -> Result<ModelLoadSpec> {
        match &self.source {
            ModelSource::Local { path } => {
                let mut spec = ModelLoadSpec::default();
                spec.path = path.to_string_lossy().into_owned();
                if let Some(c) = self.context    { spec.context_size = c; }
                if let Some(g) = self.gpu_layers { spec.gpu_layers  = g; }
                spec.chat_template_name = self.family.clone();
                Ok(spec)
            }
            ModelSource::HuggingFace { .. } | ModelSource::Ollama { .. } => {
                Err(HarnessError::InvalidArg(format!(
                    "model '{}' is from a remote source; download it first or change the registry entry to type=local",
                    self.name
                )))
            }
        }
    }
}

/// The model registry: parses `models.toml` and resolves names.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// All entries keyed by name.
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,

    /// Default model loaded on harness startup.
    #[serde(default)]
    pub default: Option<String>,
}

impl ModelRegistry {
    /// Load from a TOML file at `path`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let reg: Self = toml::from_str(&s)?;
        Ok(reg)
    }

    /// Load from a TOML string.
    pub fn from_str(s: &str) -> Result<Self> {
        let reg: Self = toml::from_str(s)?;
        Ok(reg)
    }

    /// Look up an entry by name or alias.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        if let Some(e) = self.models.get(name) {
            return Some(e);
        }
        self.models.values().find(|e| e.alias.as_deref() == Some(name))
    }

    /// List all names.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.models.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Resolve a `ModelSpec` to a `ModelLoadSpec`.
    pub fn resolve_spec(&self, spec: crate::model::ModelSpec) -> Result<ModelLoadSpec> {
        use crate::model::ModelSpec as S;
        match spec {
            S::RegistryName(name) => self
                .get(&name)
                .ok_or_else(|| HarnessError::NoSuchModel(name.clone()))?
                .resolve(),
            S::Path(p) => Ok(ModelLoadSpec::from_path(p)),
            S::Inline(s) => Ok(s),
        }
    }

    /// Default registry path: `~/.llama-harness/models.toml`.
    pub fn default_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".llama-harness").join("models.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
        [models.llama-3.1-8b-instruct]
        type = "local"
        path = "C:/models/llama-3.1-8b.gguf"
        family = "llama-3.1"
        context = 8192

        [models.qwen2.5-7b]
        type = "local"
        path = "C:/models/qwen2.5-7b.gguf"
        family = "qwen2.5"
        alias = "q7"
        context = 32768
    "#;

    #[test]
    fn parse_registry() {
        let r = ModelRegistry::from_str(SAMPLE).unwrap();
        assert_eq!(r.names(), vec!["llama-3.1-8b-instruct", "qwen2.5-7b"]);
        let ll = r.get("llama-3.1-8b-instruct").unwrap();
        assert_eq!(ll.family, "llama-3.1");
        let q = r.get("q7").unwrap();   // alias
        assert_eq!(q.family, "qwen2.5");
    }

    #[test]
    fn resolve_load_spec() {
        let r = ModelRegistry::from_str(SAMPLE).unwrap();
        let spec = r.resolve_spec(ModelSpec::RegistryName(
            "llama-3.1-8b-instruct".into())).unwrap();
        assert_eq!(spec.path, "C:/models/llama-3.1-8b.gguf");
        assert_eq!(spec.context_size, 8192);
        assert_eq!(spec.chat_template_name, "llama-3.1");
    }

    #[test]
    fn missing_model() {
        let r = ModelRegistry::from_str(SAMPLE).unwrap();
        assert!(r.resolve_spec(ModelSpec::RegistryName("nope".into())).is_err());
    }
}
