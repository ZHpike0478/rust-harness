//! agent_demo -- three tool example: time, calculator, file reader.
//!
//! Demonstrates the `Tool` trait and the `ToolFilter::Named(...)` config.
//! Requires the real llama.cpp backend to actually call tools; the stub
//! returns placeholder text, but the wiring and tool registration is
//! exercised.
//!
//! Run:
//!
//!     cargo run -p agent_demo
//!     cargo run -p agent_demo -- --registry C:/models/registry.toml

use llama_harness::{
    ChatRequest, Harness, Message, SamplingParams, Tool, ToolContext, ToolDescriptor,
    ToolFilter, ToolOutput,
};

use async_trait::async_trait;
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

// ----------------------- tool: get_time --------------------------------

struct GetTime;

#[async_trait]
impl Tool for GetTime {
    fn name(&self) -> &str { "get_time" }
    fn description(&self) -> &str {
        "Return the current UTC time as an RFC 3339 string."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        })
    }
    async fn execute(
        &self,
        _args: serde_json::Value,
        _ctx: ToolContext,
    ) -> llama_harness::Result<ToolOutput> {
        Ok(ToolOutput { text: Utc::now().to_rfc3339(), data: None, is_terminal: false, })
    }
}

// ----------------------- tool: calculator -------------------------------

#[derive(Deserialize, JsonSchema)]
struct CalcArgs {
    expression: String,
}

struct Calculator;

#[async_trait]
impl Tool for Calculator {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str {
        "Evaluate a basic arithmetic expression and return the result. \
         Supports +, -, *, /, parentheses, and integers."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '(2 + 3) * 4'."
                }
            },
            "required": ["expression"],
            "additionalProperties": false,
        })
    }
    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: ToolContext,
    ) -> llama_harness::Result<ToolOutput> {
        let parsed: CalcArgs = serde_json::from_value(args)
            .map_err(|e| llama_harness::HarnessError::InvalidArg(format!("calc args: {e}")))?;
        let result = eval(&parsed.expression).unwrap_or(f64::NAN);
        Ok(ToolOutput {
            text: format!("{} = {}", parsed.expression, result),
            data: Some(serde_json::json!({
                "expression": parsed.expression,
                "result": result,
            })),
            is_terminal: false,
        })
    }
}

// Tiny expression evaluator: digits, parens, + - * /, no precedence for
// operator-style shunting-yard -- recursive descent for * and /, then + and -.
fn eval(expr: &str) -> Option<f64> {
    let tokens = tokenize(expr)?;
    let mut p = Parser { tokens: &tokens, pos: 0 };
    p.parse_expr()
}

#[derive(Debug, Clone)]
enum Tok {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(s: &str) -> Option<Vec<Tok>> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => { chars.next(); }
            '0'..='9' | '.' => {
                let mut buf = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        buf.push(d);
                        chars.next();
                    } else { break; }
                }
                out.push(Tok::Num(buf.parse().ok()?));
            }
            '+' | '-' | '*' | '/' => { out.push(Tok::Op(c)); chars.next(); }
            '(' => { out.push(Tok::LParen); chars.next(); }
            ')' => { out.push(Tok::RParen); chars.next(); }
            _ => return None,
        }
    }
    Some(out)
}

struct Parser<'a> { tokens: &'a [Tok], pos: usize }

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> { self.tokens.get(self.pos) }
    fn bump(&mut self) -> Option<Tok> { let t = self.tokens.get(self.pos)?.clone(); self.pos += 1; Some(t) }

    fn parse_expr(&mut self) -> Option<f64> {
        let mut lhs = self.parse_term()?;
        while let Some(Tok::Op('+' | '-')) = self.peek() {
            let op = self.bump()?;
            let rhs = self.parse_term()?;
            lhs = match op { Tok::Op('+') => lhs + rhs, Tok::Op('-') => lhs - rhs, _ => unreachable!() };
        }
        Some(lhs)
    }

    fn parse_term(&mut self) -> Option<f64> {
        let mut lhs = self.parse_atom()?;
        while let Some(Tok::Op('*' | '/')) = self.peek() {
            let op = self.bump()?;
            let rhs = self.parse_atom()?;
            lhs = match op { Tok::Op('*') => lhs * rhs, Tok::Op('/') => lhs / rhs, _ => unreachable!() };
        }
        Some(lhs)
    }

    fn parse_atom(&mut self) -> Option<f64> {
        match self.bump()? {
            Tok::Num(n) => Some(n),
            Tok::LParen => {
                let v = self.parse_expr()?;
                match self.bump()? { Tok::RParen => Some(v), _ => None }
            }
            Tok::Op('-') => Some(-self.parse_atom()?),
            _ => None,
        }
    }
}

// ----------------------- tool: file_size -------------------------------

struct FileSize;

#[derive(Deserialize, JsonSchema)]
struct FileSizeArgs {
    path: String,
}

#[async_trait]
impl Tool for FileSize {
    fn name(&self) -> &str { "file_size" }
    fn description(&self) -> &str {
        "Return the size in bytes of a file at the given path."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute or relative file path." }
            },
            "required": ["path"],
            "additionalProperties": false,
        })
    }
    async fn execute(
        &self,
        args: serde_json::Value,
        _ctx: ToolContext,
    ) -> llama_harness::Result<ToolOutput> {
        let parsed: FileSizeArgs = serde_json::from_value(args)
            .map_err(|e| llama_harness::HarnessError::InvalidArg(format!("file_size args: {e}")))?;
        let meta = tokio::fs::metadata(&parsed.path).await.map_err(|e| {
            llama_harness::HarnessError::Io(std::io::Error::new(e.kind(), e.to_string()))
        })?;
        Ok(ToolOutput {
            text: format!("{} = {} bytes", parsed.path, meta.len()),
            data: Some(serde_json::json!({
                "path": parsed.path,
                "size_bytes": meta.len(),
            })),
            is_terminal: false,
        })
    }
}

// ----------------------- main ------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut registry_path: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--registry" {
            if let Some(p) = args.next() {
                registry_path = Some(PathBuf::from(p));
            }
        }
    }

    let mut builder = Harness::builder().max_loaded_models(2);
    if let Some(p) = registry_path {
        builder = builder.registry_path(p);
    }
    let harness = builder.build().await?;

    // Register tools.
    let tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(GetTime),
        Arc::new(Calculator),
        Arc::new(FileSize),
    ];
    let mut h = harness.clone();
    for t in &tools {
        h.register_tool(t.clone());
    }
    let tool_descs: Vec<ToolDescriptor> = tools.iter().map(|t| t.descriptor()).collect();
    println!("registered {} tools:", tool_descs.len());
    for d in &tool_descs {
        println!("  {} -- {}", d.name, d.description);
    }

    // Three demo prompts.
    let prompts = vec![
        "What is (17 * 23) + (144 / 12)?",
        "What's the current UTC time?",
        "What's the size of Cargo.toml in bytes?",
    ];

    for prompt in prompts {
        println!("\n> user: {prompt}");
        let req = ChatRequest {
            messages: vec![Message::user(prompt)],
            model: None,
            tools: ToolFilter::Named(
                tool_descs.iter().map(|d| d.name.clone()).collect(),
            ),
            tool_execution: Default::default(),
            max_tool_iterations: 4,
            sampling: SamplingParams::default(),
            stop: vec![],
        };
        let resp = h.chat(req).await?;
        println!("< assistant: {}", resp.final_text);
        println!(
            "[tokens: prompt={} completion={}]",
            resp.prompt_tokens, resp.completion_tokens
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}

// Helper accessor since Tool::descriptor doesn't exist on the trait;
// we construct a descriptor from the trait methods.
trait ToolDescriptorExt {
    fn descriptor(&self) -> ToolDescriptor;
}
impl<T: Tool + ?Sized> ToolDescriptorExt for T {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}
