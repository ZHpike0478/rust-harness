# Handoff: `rust-harness`

> **Project:** `Project_LlamaHarness` at `C:\Users\steph\Desktop\Project_LlamaHarness\`
> **Repo:** https://github.com/ZHpike0478/rust-harness.git
> **Last touch:** 2026-09-16
> **Tag:** `v0.1-stub` -> real backend landed (llama.cpp b11007 vendored)

## What this project is

A safe-Rust harness layer over `llama.cpp`, targeted at users who want
the same ergonomic surface (model registry, runtime model switching,
tool calling, streaming chat) whether the backend is `llama.cpp`,
`ollama`, or a stub for tests.

A **real llama.cpp backend is now wired**: `vendor/llama.cpp` submodule
is pinned at `b11007` (2f3fd0252) and `LlamaEngine.cpp` implements the
full `Engine` class surface -- model load/unload, chat-template
rendering (GGUF template with chatml fallback), a full sampling chain,
per-token streaming, and cancellation. The stub remains the fallback
when `vendor/llama.cpp/build-static/src/libllama.a` is absent, so the
build stays hermetic without GGUF weights.

The current commit also ships a **working stub C++ engine** that
exercises the full FFI surface end-to-end without requiring llama.cpp
weights or a build environment.

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

## What's still open (post real-backend landing)

- **Tool calling** is implemented as schema + trait (`Tool::parameters()`
  returns `serde_json::Value`) but the ReAct loop / GBNF compiler is
  not yet wired into `ChatEngine::chat`. Tool plumbing is plumbed
  end-to-end at the type level; the integration tests for the ReAct
  loop are not yet written. The C++ side now passes tool specs through
  but does not constrain generation with a grammar yet.
- **Streaming is C++-side real but Rust-side coalesced.**
  `LlamaEngine::complete_stream` invokes the cxx `Fn` callback once per
  sampled token (real per-token). The Rust `chat_stream` still wraps
  `chat()` and emits one coalesced `Token` + `Done`; wiring the
  per-token cxx `Fn` into an `mpsc` channel -> `Stream` pump is the
  remaining Rust-side task.
- **Cancellation** is real on the C++ side: `LlamaHandles::cancel` is
  an `std::atomic<bool>` checked every decode iteration, and
  `Engine::cancel()` sets it. The Rust `ChatRequest::cancel` plumbing
  into the engine call still needs wiring.
- **KV cache management** is reset-per-request
  (`llama_memory_clear` at generate start). Session-prefix reuse /
  KV reuse across turns is not implemented.
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

## Real llama.cpp wiring (DONE -- how to rebuild from clean)

The submodule is pinned at `b11007` (2f3fd0252). llama.cpp is built
once with CMake into static libs; cargo links those artifacts:

```bash
# One-time (or after submodule update):
cd vendor/llama.cpp
# MSYS2 ucrt64 toolchain; Julia's bin dir MUST NOT precede msys2 on
# PATH or cc1 crashes silently (DLL hijack of libgmp-10.dll).
PATH="/ucrt64/bin:$PATH" cmake -G Ninja -B build-static \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF
PATH="/ucrt64/bin:$PATH" cmake --build build-static --target llama ggml
# MinGW ar names lack the lib prefix cargo expects; copy them:
cp build-static/ggml/src/ggml.a        build-static/ggml/src/libggml.a
cp build-static/ggml/src/ggml-base.a   build-static/ggml/src/libggml-base.a
cp build-static/ggml/src/ggml-cpu.a    build-static/ggml/src/libggml-cpu.a
```

Both `build.rs` scripts detect `vendor/llama.cpp/build-static/src/libllama.a`
and switch to `LlamaEngine.cpp` + the prebuilt libs automatically.
Without it they fall back to `StubEngine.cpp` (hermetic, no weights).

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

## Next steps in the build

Ordered by impact-per-effort. Steps 1-2 from the original plan are
DONE; everything below is incremental against a working runtime.

### Immediate (low effort, high value)

1. **~~Vendor real llama.cpp~~ DONE** — submodule at `b11007`.
2. **~~LlamaEngine.cpp~~ DONE** — full Engine surface implemented.
3. **Rust-side per-token streaming pump.** C++ streams per token into
   the cxx `Fn`; remaining work is the Rust side: spawn a thread +
   `std::sync::mpsc` (or tokio mpsc) inside `chat_stream`, pass the
   receiver-backed stream to the caller, and drive
   `engine_complete_stream` with a `fn(TokenDelta)` shim that forwards
   into the channel. The cxx `Fn` is single-call per invocation, but a
   plain `fn` pointer passed once is invoked once per token (it is the
   same function each time — verified working in LlamaEngine).

### Medium effort — fills out the design

4. **GBNF grammar compiler** — generate constrained-output grammars
   from live `ToolRegistry` schemas, cache by `blake3(schema_json)`.
   Currently the plumbing is type-level only; the actual ReAct loop
   in `ChatEngine::chat` reads `ToolRegistry` but doesn't drive
   `complete()` with grammar-constrained sampling yet.
5. **Tool ReAct loop** — auto-execute tool calls and feed results
   back until the model emits a final answer or `max_tool_iterations`
   is hit. Currently `ChatRequest::tool_execution =
   Auto | HostControlled` is plumbed through but the loop is not
   implemented.
6. **~~Cancellation~~ C++-side DONE** — `std::atomic<bool>` in
   `LlamaHandles`, checked every decode iteration. Rust-side wiring
   (`ChatRequest::cancel` -> `engine_cancel`) remains.
7. **Unit + integration tests** — registry parsing is tested; the chat
   pipeline, tool filter behavior, and the FFI error mapping are not.

### Larger / optional

8. **TypeScript bindings via napi-rs** — add
   `crates/llama-harness-node/` wrapping the safe Rust API. The cxx
   `UniquePtr` is intentionally not stored long-term, so this is
   mostly a thin export layer.
9. **KV cache / speculative decoding** — both stubbed in the
   architecture diagram, neither implemented.
10. **Per-family chat template coverage** —
    `StubEngine::detect_chat_template` returns one of three strings;
    needs full Llama-3 / Mistral / ChatML / ToolACE renderers.
11. **CI** — `cargo build --workspace` on push, then
    `cargo clippy --workspace -- -D warnings`, then
    `cargo test --workspace`. The stub makes the build hermetic so CI
    works without GPU.
12. **Docs site** — README covers public API; internal `docs/` is
    empty. Worth a `docs/ARCHITECTURE.md` (the diagram from the early
    turn) and `docs/VENDORING.md` (the llama.cpp swap procedure).

## Decisions still open

- **Per-token streaming**: C++-side real (per-token cxx `Fn` invocations
  in `LlamaEngine::complete_stream`). Rust-side pump (mpsc -> Stream)
  still to be built; see "Next steps" item 3.
- **GBNF caching key**: design is "schema hash", implementation TBD
  (use `blake3::hash(&serde_json::to_vec(&schemas))` then base32).
- **Cancellation**: C++-side real via `LlamaHandles::cancel` atomic.
  Rust-side plumbing open.
- **TypeScript bindings**: design space is napi-rs; defer until Rust
  API stabilizes.

## Contact / context

This handoff was written at the end of a long scaffolding session.
For prior design discussions, see the chat history (`session_id =
20260915_191349_fbad97` in session search).
