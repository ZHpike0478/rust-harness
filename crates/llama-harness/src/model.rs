//! Model identity and load spec.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A caller-chosen identifier for a loaded model.
///
/// The harness never assumes an opaque random id; the caller picks it.
/// Convention: the registry name (e.g. `"llama-3.1-8b-instruct"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModelId {
    fn from(s: &str) -> Self { ModelId(s.to_string()) }
}
impl From<String> for ModelId {
    fn from(s: String) -> Self { ModelId(s) }
}

/// Information about a loaded model, as known to the Rust side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Caller-chosen id.
    pub id: String,
    /// Path to the .gguf file.
    pub path: String,
    /// Architecture (e.g. "llama", "qwen2", "mistral").
    pub arch: String,
    /// Family name used for chat template dispatch (e.g. "llama-3.1").
    pub family: String,
    /// Context window in tokens.
    pub context_size: i32,
    /// Approximate memory footprint in bytes.
    pub size_bytes: u64,
    /// Approximate parameter count in billions.
    pub n_params_b: i32,
    /// Whether the model supports tool calling (per detected chat
    /// template).
    pub supports_tools: bool,
}

impl From<llama_harness_ffi::ffi::ModelInfo> for ModelInfo {
    fn from(m: llama_harness_ffi::ffi::ModelInfo) -> Self {
        Self {
            id: m.id,
            path: m.path,
            arch: m.arch,
            family: m.family,
            context_size: m.context_size,
            size_bytes: m.size_bytes,
            n_params_b: m.n_params_b,
            supports_tools: m.supports_tools,
        }
    }
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            id: String::new(),
            path: String::new(),
            arch: String::new(),
            family: String::new(),
            context_size: 0,
            size_bytes: 0,
            n_params_b: 0,
            supports_tools: false,
        }
    }
}

/// Caller-facing spec for loading a model.
#[derive(Debug, Clone)]
pub struct ModelLoadSpec {
    /// Path to the .gguf file. Absolute.
    pub path: String,
    /// Context window. Default 4096.
    pub context_size: i32,
    /// Logical batch size. Default 512.
    pub batch_size: i32,
    /// CPU threads. 0 = auto.
    pub threads: i32,
    /// GPU layers to offload. -1 = all (default), 0 = CPU-only.
    pub gpu_layers: i32,
    /// Use mmap for weight loading. Default true.
    pub use_mmap: bool,
    /// Lock weights in RAM. Default false.
    pub use_mlock: bool,
    /// Override chat template detection. "" = auto.
    pub chat_template_name: String,
}

impl Default for ModelLoadSpec {
    fn default() -> Self {
        Self {
            path: String::new(),
            context_size: 4096,
            batch_size: 512,
            threads: 0,
            gpu_layers: -1,
            use_mmap: true,
            use_mlock: false,
            chat_template_name: String::new(),
        }
    }
}

impl ModelLoadSpec {
    /// Construct from a path.
    pub fn from_path(path: impl Into<String>) -> Self {
        Self { path: path.into(), ..Default::default() }
    }

    /// Convert to the FFI DTO.
    pub(crate) fn to_ffi(&self) -> llama_harness_ffi::ffi::ModelLoadSpec {
        llama_harness_ffi::ffi::ModelLoadSpec {
            path: self.path.clone(),
            context_size: self.context_size,
            batch_size: self.batch_size,
            threads: self.threads,
            gpu_layers: self.gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            chat_template_name: self.chat_template_name.clone(),
        }
    }
}

/// What to load. Either a registry name (resolved via `models.toml`)
/// or an explicit inline spec.
#[derive(Debug, Clone)]
pub enum ModelSpec {
    /// Resolve via the registry.
    RegistryName(String),
    /// Load directly from a path with default settings.
    Path(String),
    /// Load from a path with explicit settings.
    Inline(ModelLoadSpec),
}

impl From<&str> for ModelSpec {
    fn from(s: &str) -> Self { ModelSpec::RegistryName(s.to_string()) }
}
impl From<String> for ModelSpec {
    fn from(s: String) -> Self { ModelSpec::RegistryName(s) }
}
