# Handoff: `rust-harness`

> **Project:** `Project_LlamaHarness` at `C:\Users\steph\Desktop\Project_LlamaHarness\`
> **Repo:** https://github.com/ZHpike0478/rust-harness.git
> **Last touch:** 2026-09-15
> **Tag:** `v0.1-stub`

## What this project is

A safe-Rust harness layer over `llama.cpp`, targeted at users who want
the same ergonomic surface (model registry, runtime model switching,
tool calling, streaming chat) whether the backend is `llama.cpp`,
`ollama`, or a stub for tests.

The current commit ships a **working stub C++ engine** that exercises
the full FFI surface end-to-end without requiring llama.cpp weights or a
build environment. Replacing the stub with real `llama.cpp` is a
self-contained task and does not change the Rust API.

## What's done (verified `cargo build --workspace` clean)

- Workspace layout: `crates/llama-harness` (safe API), `crates/llama-harness-ffi`
  (cxx bridge), `crates/llama-harness-macros` (`#[tool]` proc-macro).
- Examples: `examples/chat_cli` (REPL with `/list /load /use /info /system /temp /clear /tokens /quit`),
  `examples/agent_demo` (3 tools).
- `registry/models.toml` example file.
- C++ core (`crates/llama-harness-ffi/cpp/harness.h` + `StubEngine.cpp`)
  implements the full `Engine` class surface.
- cxx bridge compiled via `cxx_build::bridge` in `llama-harness/build.rs`.
- Static-lib propagation handled via per-binary `build.rs` (chat_cli,
  agent_demo each emit the link directive).
- `chat_cli.exe --registry ./registry/models.toml` boots, parses 3
  models from registry, accepts commands. FFI error path works
  (returns clean error for missing model names).

## What's stubbed

- **No real `llama.cpp` vendored yet.** `StubEngine.cpp` is a
  placeholder that tracks "loaded" models in a `map<rust::String,
  ModelInfo>` and emits deterministic text. See "Wiring real
  llama.cpp" below.
- Tool calling is implemented as schema + trait (`Tool::parameters()`
  returns `serde_json::Value`) but the ReAct loop / GBNF compiler is
  not yet wired into `ChatEngine::chat`. Tool plumbing is plumbed
  end-to-end at the type level; the integration tests for the ReAct
  loop are not yet written.
- Streaming: `chat_stream` returns `Pin<Box<dyn Stream<Item =
  StreamEvent>>>`. The stub emits one `Token { text }` event followed
  by `Done(ChatResponse)`. Per-token streaming will require real
  llama.cpp + a callback pump (cxx `Fn` → mpsc channel).
- Cancel: `Engine::cancel()` is a no-op stub. The `ChatRequest::cancel`
  field is wired into `tokio::sync::Mutex` access; real cancellation
  needs the llama.cpp cancel API.
- TypeScript bindings: design space laid out (napi-rs target), not
  scaffolded. The Rust API is intentionally `Send`-agnostic to make
  this trivial to add later (each cxx opaque type is wrapped, not
  stored as `cxx::UniquePtr` in long-lived fields).

## Architecture (one-paragraph version)

```
Rust caller ──► llama-harness (safe API, async, tokio)
                  │
                  │ cxx bridge (lib.rs <-> lib.rs.h)
                  ▼
              llama-harness-ffi (cxx::bridge)
                  │
                  │ C++ free functions (engine_*) in
                  │ namespace harness
                  ▼
              C++ Engine class ──► (currently) StubEngine.cpp
                                 ──► (eventually) LlamaEngine.cpp
                                       │
                                       ▼
                                   vendor/llama.cpp/  (not yet present)
```

## File map (what's where)

| Path                                              | Role                                          |
| ------------------------------------------------- | --------------------------------------------- |
| `Cargo.toml`                                      | workspace root, deps, editions                 |
| `rust-toolchain.toml`                             | pins `stable`                                 |
| `.gitignore`                                      | excludes `target/`, `models/*.gguf`, etc.     |
| `crates/llama-harness/src/lib.rs`                 | public API surface (re-exports)               |
| `crates/llama-harness/src/harness.rs`             | `Harness` + `HarnessBuilder`                 |
| `crates/llama-harness/src/chat.rs`                | `ChatRequest` / `ChatResponse` / `StreamEvent`|
| `crates/llama-harness/src/message.rs`             | `Message`, `Role`                             |
| `crates/llama-harness/src/model.rs`              | `ModelLoadSpec`, `ModelInfo`, FFI conversion  |
| `crates/llama-harness/src/sampling.rs`            | `SamplingParams`, FFI conversion              |
| `crates/llama-harness/src/tool.rs`                | `Tool` trait, `ToolRegistry`, `ToolFilter`    |
| `crates/llama-harness/src/registry.rs`            | `ModelRegistry`, `ModelEntry`, `ModelSource`  |
| `crates/llama-harness/src/error.rs`              | `HarnessError` enum, `Result<T>`              |
| `crates/llama-harness/build.rs`                   | compiles C++ via cxx_build                    |
| `crates/llama-harness-ffi/src/lib.rs`            | cxx bridge (`#[cxx::bridge] mod ffi`)        |
| `crates/llama-harness-ffi/build.rs`               | emits `CARGO_DTO_HEADER_DIR` for downstream  |
| `crates/llama-harness-ffi/cpp/harness.h`         | C++ `Engine` class declaration, free-fns     |
| `crates/llama-harness-ffi/cpp/StubEngine.cpp`     | placeholder C++ implementation                |
| `crates/llama-harness-ffi/cpp/HarnessError.cpp`  | error encode                                  |
| `crates/llama-harness-macros/src/lib.rs`          | `#[tool]` proc-macro (JSON-Schema derivation) |
| `examples/chat_cli/src/main.rs`                   | interactive REPL                              |
| `examples/chat_cli/build.rs`                      | staticlib link directive                      |
| `examples/agent_demo/src/main.rs`                | 3-tool demo                                   |
| `examples/agent_demo/build.rs`                   | staticlib link directive                      |
| `registry/models.toml`                            | example registry file                         |

## Design decisions worth knowing

| Decision | Rationale |
| -------- | --------- |
| cxx bridge, not raw FFI | Safe Rust at the call site, type-checked at compile time |
| Free functions for `engine_*`, not `impl Engine` methods | cxx 1.0.202 doesn't expose methods on opaque types through Mutex guards — free fns + `cxx::UniquePtr::pin_mut(&mut *g)` works |
| DTOs declared ONLY in Rust `#[cxx::bridge]` | cxx-generated `lib.rs.h` is the authority; C++ headers forward-declare only |
| `crate-type = ["staticlib", "rlib"]` for `llama-harness-ffi` | rlib lets Rust import cxx types; staticlib holds C++ impls |
| Per-binary `build.rs` emits `rustc-link-lib=static=llama_harness_inline` | Cargo does NOT propagate link directives transitively |
| `models.toml` uses `[models."llama-3.1-8b"]` (quoted) | Unquoted `.` is interpreted as TOML table nesting |
| `ChatRequest::tool_execution: Auto \| HostControlled` | Per user request: opt-in per request |
| `chat_stream` returns `Pin<Box<dyn Stream<Item = StreamEvent>>>` | `cxx::UniquePtr` isn't `Send`; boxed stream gives local-pinned iterator |
| `Pin<Box<dyn Stream>>` not `impl Stream<Item = ...>` | Concrete return type so callers don't need to worry about pinning |

## Wiring real llama.cpp (next session's task)

1. **Vendor llama.cpp** as a pinned submodule:
   ```bash
   cd C:/Users/steph/Desktop/Project_LlamaHarness
   git submodule add https://github.com/ggerganov/llama.cpp vendor/llama.cpp
   git -C vendor/llama.cpp checkout b1234   # pick a known-good tag
   ```
2. **Add `LlamaEngine.cpp`** next to `StubEngine.cpp` in
   `crates/llama-harness-ffi/cpp/`. Implement the same `Engine` class
   surface using llama.cpp's `llama_context`, `llama_decode`,
   `llama_sampling_*`, and `llama_apply_chat_template` APIs.
3. **Update `build.rs`** in `llama-harness-ffi` — it already
   auto-detects `vendor/llama.cpp/` and adds the include + link
   directives when present. Check the warning emitted at build time:
   `llama.cpp not vendored at vendor/llama.cpp -- building stub engine`.
4. **Switch** `crates/llama-harness-ffi/cpp/CMakeLists.txt` (when
   added) or the `llama-harness/build.rs` to compile `LlamaEngine.cpp`
   instead of `StubEngine.cpp`. One-line swap.
5. **Streaming**: implement per-token streaming by pumping llama.cpp's
   `llama_decode` loop into a `mpsc::Sender`, then forward to the cxx
   `Fn` callback per token. The cxx `Fn` is single-call, so each
   token requires a fresh trampoline invocation.

## Known issues / sharp edges

- **Rust edition defaults**: my edits kept trying to use
  `edition.workspace = true` and rustc was picking 2015. Workaround:
  set `edition = "2021"` explicitly in each crate's `Cargo.toml`.
  Future Rust releases may fix this; revisit then.
- **cxx + MSVC**: the cxx-generated `lib.rs.h` must be on the
  `llama-harness` `build.rs` include path. We pipe it through
  `CARGO_DTO_HEADER_DIR` env var so both crates agree on the
  build-time location.
- **cxx `rust::Str` vs `rust::String`**: views vs owned; explicit
  `rust::String(std::string(rs))` everywhere. Look for `to_std` /
  `to_rust` helpers in `StubEngine.cpp` for the pattern.
- **TOML key escaping**: `models.<name>` must be quoted when name
  contains `.`. See `registry/models.toml`.

## Smoke tests you can re-run

```bash
cd C:/Users/steph/Desktop/Project_LlamaHarness
export PATH="/c/Program Files/dotnet:$PATH"

# 1. Whole workspace builds
cargo build --workspace

# 2. REPL boots and accepts commands
printf "/help\n/quit\n" | ./target/debug/chat_cli.exe

# 3. Registry parses
printf "/list\n/quit\n" | \
  ./target/debug/chat_cli.exe --registry ./registry/models.toml

# 4. Agent demo binary exists and links
ls -l target/debug/agent_demo.exe
```

Expected output of (3) ends with `known models (3): llama-3.1-8b-instruct,
mistral-7b-instruct, qwen2.5-7b-instruct`.

## Where to start on day 2

1. Open `crates/llama-harness/src/harness.rs` — that's the
   orchestrator. `chat()`, `chat_stream()`, `load()`,
   `register_tool()` are the four entry points callers care about.
2. Open `crates/llama-harness-ffi/cpp/StubEngine.cpp` — that's the
   template for the real `LlamaEngine.cpp`. The free functions at the
   bottom are the cxx surface; everything else is internal.
3. The README at `README.md` is the public-facing doc. Update it once
   real llama.cpp is wired.

## Decisions still open

- **Per-token streaming**: confirmed in design, not yet implemented.
  See "Wiring real llama.cpp" step 5.
- **GBNF caching key**: design is "schema hash", implementation TBD
  (use `blake3::hash(&serde_json::to_vec(&schemas))` then base32).
- **Cancellation**: no-op stub. Real impl needs an `AtomicBool`
  inside `EngineState`, checked from llama.cpp's sampling loop.
- **TypeScript bindings**: design space is napi-rs; defer until Rust
  API stabilizes.

## Contact / context

This handoff was written at the end of a long scaffolding session.
For prior design discussions, see the chat history (`session_id =
20260915_191349_fbad97` in session search).
