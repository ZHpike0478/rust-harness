# Project_LlamaHarness

A safe-Rust harness around [llama.cpp](https://github.com/ggerganov/llama.cpp)
that adds:

- **Centralized model registry** -- one `models.toml` file describes every
  model the host cares about (path, sha256, family, default params).
  Callers say `harness.chat("llama-3.1-8b-instruct", ...)`.
- **Runtime model switching** -- `max_loaded_models` models can be live at
  once; LRU eviction kicks in automatically when a new one is loaded.
- **Tool calling** -- tools implement a `Tool` trait. Wire formats per
  model family: Llama-3, Hermes, Mistral, Qwen, FireFunction v2,
  raw-JSON fallback.
- **Streaming API** -- `chat_stream` returns `Pin<Box<dyn Stream<Item =
  StreamEvent>>>`. Events are token deltas *and* structured tool events
  on a single stream.
- **Auto vs host-controlled tool execution** -- per request:
  `ChatRequest::tool_execution = Auto | HostControlled`.

## Layout

```
crates/
  llama-harness/          # the safe Rust API you actually call
  llama-harness-ffi/      # cxx bridge + cxx-generated shim
  llama-harness-macros/   # `#[tool]` proc-macro
examples/
  chat_cli/               # interactive REPL
  agent_demo/             # 3 tools: time, calculator, file reader
registry/
  models.toml             # example registry
```

## Building

```bash
# whole workspace
cargo build --workspace

# run chat_cli with the example registry
./target/debug/chat_cli.exe --registry ./registry/models.toml
```

## What's here vs. what's stubbed

The current C++ backend (`StubEngine.cpp`) does NOT link llama.cpp. It
implements a working surface that tracks loaded models and emits
deterministic placeholder completions. This lets the entire Rust
workspace compile, run unit tests, and exercise the FFI in CI before
real GGUF weights are available.

To wire in real llama.cpp:

1. Vendor it as a submodule:
   ```bash
   git submodule add https://github.com/ggerganov/llama.cpp vendor/llama.cpp
   git -C vendor/llama.cpp checkout b1234
   ```
2. Replace `StubEngine.cpp` with a `LlamaEngine.cpp` that implements the
   same `Engine` class against llama.cpp.
3. `llama-harness-ffi/build.rs` already auto-detects `vendor/llama.cpp/`
   and adds the necessary include / link directives.

The Rust API does not change.

## Wire format coverage

| Model family         | Tool format                |
| -------------------- | -------------------------- |
| Llama-3.1/3.2/3.3    | `<function_calls>` XML-ish |
| Hermes-2/3 (Nous)    | `<tool_call>` XML            |
| Mistral / Mixtral    | `[TOOL_CALLS]` legacy      |
| Qwen 2.5             | `<tool_call>` XML-ish        |
| FireFunction v2      | `def function(...)`        |
| anything else        | raw JSON (fallback)        |

GBNF grammars are generated from the **live** `ToolRegistry` schemas
and cached by schema hash, so the same load pattern works for every
model family without per-family regex maintenance.

## Per-request tool execution semantics

```rust
ChatRequest {
    model:               Some("llama-3.1-8b-instruct".into()),
    messages:            vec![...],
    sampling:            SamplingParams::default(),
    tool_execution:      ToolExecution::Auto,        // or HostControlled
    tool_filter:         ToolFilter::All,            // or Allowlist(...) / Named(...)
    max_tool_iterations: 8,
    cancel:              None,
}
```
