//! Tool registry + Tool trait + `#[tool]` proc-macro.
//!
//! Tools are user-defined async functions whose JSON-Schema is derived
//! from the function signature. The harness compiles the registered
//! tools' schemas into a GBNF grammar and feeds it to llama.cpp so the
//! model can only emit valid tool calls or plain text.
//!
//! Example (in agent_demo):
//!
//! ```ignore
//! use llama_harness_macros::tool;
//!
//! #[tool(description = "Get current weather for a city")]
//! async fn get_weather(city: String, units: TempUnits) -> Weather {
//!     ...
//! }
//! ```
//!
//! The macro generates a struct implementing `Tool`, with `parameters()`
//! derived from the function signature via `schemars`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

pub use llama_harness_macros::tool;

use crate::error::{HarnessError, Result};

/// Execution context passed to a tool. Carries the current model id,
/// the original user request id (for tracing), and arbitrary metadata
/// the host wants to make available.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// Model id used for the conversation that triggered this call.
    pub model_id: Option<String>,
    /// Optional trace id (ULID/UUID) for the entire chat request.
    pub request_id: Option<String>,
    /// Arbitrary host-supplied metadata.
    pub metadata: HashMap<String, String>,
}

/// Output returned by a tool. Either textual (rendered back into the
/// model as a `tool` message) or structured (also serialized to JSON
/// and rendered).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Tool-emitted text the model sees.
    pub text: String,
    /// Optional structured payload (logged, kept on the response, etc).
    #[serde(default)]
    pub data: Option<Value>,
    /// True if the tool considers this output "final" (e.g. an error
    /// that should terminate the ReAct loop).
    #[serde(default)]
    pub is_terminal: bool,
}

impl ToolOutput {
    /// Construct a textual output.
    pub fn text(s: impl Into<String>) -> Self {
        Self { text: s.into(), data: None, is_terminal: false }
    }

    /// Construct a structured output.
    pub fn data(data: Value) -> Self {
        Self {
            text: serde_json::to_string_pretty(&data).unwrap_or_default(),
            data: Some(data),
            is_terminal: false,
        }
    }

    /// Mark this output as terminating (the harness stops iterating).
    pub fn terminal(mut self) -> Self {
        self.is_terminal = true;
        self
    }
}

/// Public-facing descriptor for a registered tool: name, description,
/// parameters JSON Schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Unique tool name (must match across registration and model output).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema object describing accepted arguments.
    pub parameters: Value,
}

/// Trait for tools. Implemented by `#[tool]`-generated types, or by hand.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name.
    fn name(&self) -> &str;
    /// Tool description.
    fn description(&self) -> &str;
    /// JSON Schema for the parameters object.
    fn parameters(&self) -> Value;
    /// Execute with parsed arguments. Return `ToolOutput` or `Err`.
    async fn execute(&self, args: Value, ctx: ToolContext) -> Result<ToolOutput>;
}

/// Filter applied per-request to choose which tools the model sees.
#[derive(Debug, Clone, Default)]
pub enum ToolFilter {
    /// No tools visible to the model.
    None,
    /// All registered tools visible.
    #[default]
    All,
    /// Only tools in this list (by name). Unknown names are silently dropped.
    Named(Vec<String>),
}

/// Registry storing all `Tool` implementations by name.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
}

impl ToolRegistry {
    /// Construct an empty registry.
    pub fn new() -> Self { Self::default() }

    /// Register a tool. Replaces any existing tool with the same name.
    pub fn register<T: Tool + 'static>(&self, tool: T) {
        let mut g = self.inner.write();
        g.insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.inner.read().get(name).cloned()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Names of all registered tools (sorted).
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.inner.read().keys().cloned().collect();
        n.sort();
        n
    }

    /// Descriptors for tools matching the filter.
    pub(crate) fn descriptors(&self, filter: &ToolFilter) -> Vec<ToolDescriptor> {
        let g = self.inner.read();
        let mut out: Vec<ToolDescriptor> = g.values()
            .filter(|t| match filter {
                ToolFilter::None     => false,
                ToolFilter::All      => true,
                ToolFilter::Named(ns) => ns.contains(&t.name().to_string()),
            })
            .map(|t| ToolDescriptor {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect();
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Validate that a tool call's arguments match the tool's schema.
    /// Returns the parsed JSON if valid; an error otherwise.
    pub(crate) fn validate_call(&self, name: &str, args: &Value) -> Result<()> {
        let t = self.get(name).ok_or_else(|| HarnessError::UnknownTool(name.to_string()))?;
        // Light validation: ensure args is an object; full schema validation
        // could be added with a crate like jsonschema. The GBNF grammar
        // already guarantees structural validity of the model's output, so
        // this is a defense-in-depth check.
        if !args.is_object() {
            return Err(HarnessError::MalformedToolCall(format!(
                "tool '{}' arguments must be a JSON object, got {}",
                name, args
            )));
        }
        // Touch parameters() so the tool sees a "ping" (could be used for
        // future schema-driven arg coercion).
        let _ = t.parameters();
        Ok(())
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("names", &self.names())
            .finish()
    }
}

/// Helper: parse a single tool-call invocation from the model's raw
/// output text. The model's output is expected to be either plain text
/// or one or more `<tool_call>{...}</tool_call>` blocks (Hermes/Qwen style)
/// or `` blocks (Llama-3 style).
///
/// Returns `(tool_name, args_json)` for each call found, in order.
pub(crate) fn parse_tool_calls(raw: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut rest = raw;

    // Try Llama-3.1 format first:
    // {"name": "...", "arguments": {...}}
    while let Some(start) = rest.find("{\"name\"") {
        let after = &rest[start..];
        if let Some((name, args, end)) = extract_json_object_with_name(after) {
            out.push((name, args));
            rest = &rest[start + end..];
        } else {
            break;
        }
    }
    if !out.is_empty() {
        return out;
    }

    // Try Hermes/Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        if let Some(end_tag) = after.find("</tool_call>") {
            let inner = &after[..end_tag];
            if let Some((name, args, _)) = extract_json_object_with_name(inner) {
                out.push((name, args));
            }
            rest = &after[end_tag + "</tool_call>".len()..];
        } else {
            break;
        }
    }
    if !out.is_empty() {
        return out;
    }

    // Try plain JSON with "name" + "arguments" anywhere.
    if let Ok(v) = serde_json::from_str::<Value>(raw.trim()) {
        if let Some(obj) = v.as_object() {
            if let (Some(name), Some(args)) = (obj.get("name"), obj.get("arguments")) {
                if let Some(n) = name.as_str() {
                    out.push((n.to_string(), args.clone()));
                }
            }
        }
    }

    out
}

/// Extract `{"name": "...", "arguments": {...}}` from the start of `s`.
/// Returns (name, args_value, bytes_consumed).
fn extract_json_object_with_name(s: &str) -> Option<(String, Value, usize)> {
    let trimmed = s.trim_start();
    let offset = s.len() - trimmed.len();
    if !trimmed.starts_with('{') {
        return None;
    }
    // Brace-match the top-level object.
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    let mut end = None;
    for (i, c) in trimmed.char_indices() {
        if escape { escape = false; continue; }
        if in_str {
            if c == '\\' { escape = true; }
            else if c == '"' { in_str = false; }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => depth += 1,
            '}' => { depth -= 1; if depth == 0 { end = Some(i + 1); break; } }
            _ => {}
        }
    }
    let end = end?;
    let obj_text = &trimmed[..end];
    let v: Value = serde_json::from_str(obj_text).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = v.get("arguments")?.clone();
    Some((name, args, offset + end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama3_format() {
        let raw = r#"Hello, calling the weather tool now.
{"name": "get_weather", "arguments": {"city": "Boston"}}
Done."#;
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1["city"], "Boston");
    }

    #[test]
    fn parse_hermes_format() {
        let raw = r#"<tool_call>
{"name": "lookup", "arguments": {"q": "rust async"}}
</tool_call>"#;
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "lookup");
        assert_eq!(calls[0].1["q"], "rust async");
    }

    #[test]
    fn parse_plain_text() {
        let raw = "No tools here, just a regular answer.";
        assert!(parse_tool_calls(raw).is_empty());
    }
}
