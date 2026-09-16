//! Typed error model for the harness.
//!
//! All harness functions return `Result<T, HarnessError>`. The FFI
//! bridge converts its C++ `HarnessError` exceptions into these
//! variants by parsing the `KIND:message` encoding.

use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    /// Wrapped I/O failure (file missing, registry unreadable, etc.).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Underlying llama.cpp / harness-engine failure.
    #[error("backend: {0}")]
    Backend(String),

    /// User cancelled a stream mid-flight.
    #[error("cancelled")]
    Cancelled,

    /// The requested model is not loaded.
    #[error("no such model: {0}")]
    NoSuchModel(String),

    /// The model ID is already loaded.
    #[error("model already loaded: {0}")]
    AlreadyLoaded(String),

    /// Caller passed invalid arguments (bad schema, malformed spec).
    #[error("invalid argument: {0}")]
    InvalidArg(String),

    /// A model is referenced that has not been loaded.
    #[error("model not loaded: {0}")]
    ModelNotLoaded(String),

    /// No active model is set and the request required one.
    #[error("no active model")]
    NoActiveModel,

    /// Harness-engine (FFI) failure.
    #[error("engine: {0}")]
    Engine(String),

    /// Tool execution failed (returned Err from the user function).
    #[error("tool '{name}' failed: {message}")]
    ToolExecution {
        /// Tool name as registered.
        name: String,
        /// Error message returned from the tool.
        message: String,
    },

    /// Tool schema validation failed (missing fields, bad types).
    #[error("invalid tool schema for '{name}': {message}")]
    InvalidSchema {
        /// Tool name.
        name: String,
        /// Reason.
        message: String,
    },

    /// The model tried to call a tool that isn't registered.
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    /// The model's output failed to parse as a valid tool call or
    /// matched no recognizable pattern.
    #[error("malformed tool call: {0}")]
    MalformedToolCall(String),

    /// Tried to load too many models (over the configured cap).
    #[error("model capacity exceeded (max {max})")]
    CapacityExceeded {
        /// Configured cap.
        max: usize,
    },

    /// Reached `max_tool_iterations` without the model emitting a final answer.
    #[error("tool iteration limit reached ({limit}) without final answer")]
    IterationLimit {
        /// Configured limit.
        limit: usize,
    },

    /// Anything else.
    #[error("other: {0}")]
    Other(String),
}

impl HarnessError {
    /// Decode a `KIND:message` string emitted by the C++ side.
    pub(crate) fn from_ffi(encoded: &str) -> Self {
        let (kind_str, message) = match encoded.split_once(':') {
            Some((k, m)) => (k, m.to_string()),
            None => ("0", encoded.to_string()),
        };
        match kind_str {
            "1" => HarnessError::InvalidArg(message),
            "2" => HarnessError::InvalidArg(message),
            "3" => HarnessError::Backend(message),
            "4" => HarnessError::Cancelled,
            "5" => HarnessError::Backend(format!("model busy: {}", message)),
            "6" => HarnessError::NoSuchModel(message),
            "7" => HarnessError::Backend(format!("tokenization: {}", message)),
            "8" => HarnessError::InvalidSchema {
                name: "<unknown>".into(),
                message,
            },
            "9" => HarnessError::Io(std::io::Error::new(
                std::io::ErrorKind::Other, message)),
            _ => HarnessError::Other(message),
        }
    }
}

impl From<cxx::Exception> for HarnessError {
    fn from(e: cxx::Exception) -> Self {
        HarnessError::from_ffi(e.what())
    }
}

/// Crate-local Result alias.
pub type Result<T> = std::result::Result<T, HarnessError>;

// Manual From<serde_json::Error> -- used in registry/tool modules.
impl From<serde_json::Error> for HarnessError {
    fn from(e: serde_json::Error) -> Self {
        HarnessError::InvalidArg(format!("json: {}", e))
    }
}

impl From<toml::de::Error> for HarnessError {
    fn from(e: toml::de::Error) -> Self {
        HarnessError::InvalidArg(format!("toml: {}", e))
    }
}

impl From<anyhow::Error> for HarnessError {
    fn from(e: anyhow::Error) -> Self {
        HarnessError::Other(e.to_string())
    }
}

/// Convenience: Display via fmt::Display to keep `format!("{}", err)`
/// working in user code without importing Display trait.
pub struct DisplayErr<'a>(pub &'a HarnessError);

impl fmt::Display for DisplayErr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}
