//! Message types: System / User / Assistant / Tool.

use serde::{Deserialize, Serialize};

/// Role of a message in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// System prompt (set model behavior).
    System,
    /// User turn.
    User,
    /// Model turn. May contain text and/or tool calls.
    Assistant,
    /// Tool result. `tool_call_id` must be set.
    Tool,
}

impl Role {
    /// Wire-format string the chat templates expect.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role discriminator.
    pub role: Role,
    /// Text content of the message. Empty for assistant turns that
    /// contain only tool calls.
    pub content: String,
    /// For `Role::Tool`: the id of the tool call this result belongs to.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
    /// For `Role::Tool`: the name of the tool (some chat templates
    /// require it, e.g. llama-3 `<tool_result name="...">`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_name: String,
    /// For `Role::Assistant`: a JSON array of tool call objects emitted
    /// by the model. Empty for plain text turns.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_calls_json: String,
}

impl Message {
    /// Construct a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_call_id: String::new(),
            tool_name: String::new(),
            tool_calls_json: String::new(),
        }
    }

    /// Construct a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_call_id: String::new(),
            tool_name: String::new(),
            tool_calls_json: String::new(),
        }
    }

    /// Construct an assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_call_id: String::new(),
            tool_name: String::new(),
            tool_calls_json: String::new(),
        }
    }

    /// Attach raw `tool_calls_json` to an assistant message. Used by
    /// the harness when echoing tool calls back into the conversation.
    pub fn with_tool_calls_json(mut self, s: &str) -> Self {
        self.tool_calls_json = s.to_string();
        self
    }

    /// Construct a tool result message.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            tool_calls_json: String::new(),
        }
    }

    /// Convert to the FFI DTO.
    pub(crate) fn to_ffi(&self) -> llama_harness_ffi::ffi::ChatMessage {
        llama_harness_ffi::ffi::ChatMessage {
            role: self.role.as_str().to_string(),
            content: self.content.clone(),
            tool_call_id: self.tool_call_id.clone(),
            tool_name: self.tool_name.clone(),
            tool_calls_json: self.tool_calls_json.clone(),
        }
    }
}
