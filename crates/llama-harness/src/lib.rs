//! # llama-harness
//!
//! A safe Rust harness around `llama.cpp` that adds:
//!
//!   * **Typed model registry** -- central `models.toml` so callers say
//!     `harness.chat("llama-3.1-8b-instruct")` instead of juggling paths.
//!   * **Multi-model runtime** -- load several models, switch between
//!     them per request, automatic LRU eviction past a configured cap.
//!   * **Tool calling** -- `#[tool]` proc-macro auto-derives JSON Schema
//!     from a Rust function signature; the harness compiles the schemas
//!     into a GBNF grammar so llama.cpp can only emit valid tool calls
//!     or plain text.
//!   * **Dual-layer streaming** -- both raw token deltas (for prose
//!     rendering) AND structured `StreamEvent`s (Token | ToolStart |
//!     ToolArgs | ToolResult | Done | Error).
//!   * **Two tool execution modes** -- `Auto` (harness executes tools
//!     in a ReAct loop) and `HostControlled` (harness emits a
//!     `ToolCallRequest` and waits for the caller to feed results back).
//!
//! ## Quick start
//!
//! ```no_run
//! use llama_harness::{Harness, ChatRequest, Message, Role, SamplingParams};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let mut h = Harness::builder()
//!     .registry_path("~/.llama-harness/models.toml")
//!     .max_loaded_models(2)
//!     .build()
//!     .await?;
//!
//! h.load("llama-3.1-8b-instruct").await?;
//!
//! let resp = h.chat(ChatRequest {
//!     messages: vec![Message::user("hello, world")],
//!     model: None,
//!     tools: Default::default(),
//!     tool_execution: Default::default(),
//!     max_tool_iterations: 4,
//!     sampling: SamplingParams::default(),
//!     stop: vec![],
//! }).await?;
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod model;
pub mod registry;
pub mod tool;
pub mod chat;
pub mod harness;
pub mod sampling;
pub mod message;

pub use error::{HarnessError, Result};
pub use model::{ModelId, ModelInfo, ModelLoadSpec, ModelSpec};
pub use registry::{ModelEntry, ModelRegistry, ModelSource};
pub use tool::{Tool, ToolContext, ToolDescriptor, ToolFilter, ToolOutput};
pub use chat::{
    ChatRequest, ChatResponse, ChatRole, ReActStep, StreamEvent, ToolExecution,
    ToolCallRequest, ToolCallResult,
};
pub use crate::harness::{Harness, HarnessBuilder};
pub use sampling::SamplingParams;
pub use message::{Message, Role};
