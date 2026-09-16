// SPDX-License-Identifier: MIT OR Apache-2.0
//
// llama-harness-ffi  --  C++ bridge between Rust (llama-harness)
// and llama.cpp (vendored under vendor/llama.cpp at build time).
//
// cxx integration notes:
//   * DTOs (SamplingParams, ModelLoadSpec, ModelInfo, ChatMessage,
//     ToolSpec, TokenDelta, CompletionResult) are declared EXCLUSIVELY
//     in the Rust bridge (`bridge.rs`). The cxx-generated header
//     `bridge.rs.h` is the authority for their C++ definitions; this
//     header only forward-declares them.
//   * Strings: `rust::cxxbridge1::String` (== std::string ABI, owned by cxx).
//   * Vectors: `rust::cxxbridge1::Vec<T>` with T also cxx-trivially-relocatable.
//   * Callbacks: `rust::Fn<void(...)>` (single-call shim around std::function).
//
// Design rules:
//   1. NEVER expose llama.h types in cxx signatures.
//   2. RAII everywhere.
//   3. Errors throw HarnessError; cxx catches as cxx::Exception.
//   4. Engine is a concrete class. Rust gets the same surface whether the
//      backend is the stub or real llama.cpp.

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rust/cxx.h>

namespace harness {

// ----- Forward declarations of cxx-shared DTOs ----------------------
//
// The full definitions live in the cxx-generated `bridge.rs.h`, which
// is included by `bridge.rs.cc` (the cxx shim). Forward-declaring here
// lets Engine methods reference these types without depending on their
// layout (Engine only stores them by value in function signatures, never
// as members).

struct SamplingParams;
struct ModelLoadSpec;
struct ModelInfo;
struct ChatMessage;
struct ToolSpec;
struct TokenDelta;
struct CompletionResult;

// ----- Error type thrown across the FFI ------------------------------

class HarnessError : public std::exception {
public:
    enum Kind {
        Unknown      = 0,
        NotFound     = 1,
        InvalidArg   = 2,
        Backend      = 3,
        Cancelled    = 4,
        ModelBusy    = 5,
        NoSuchModel  = 6,
        Tokenization = 7,
        Schema       = 8,
        Io           = 9,
    };

    HarnessError(Kind k, ::rust::String msg)
        : kind_(k), msg_(std::move(msg)) {}
    Kind kind() const noexcept { return kind_; }
    const ::rust::String& message() const noexcept { return msg_; }
    const char* what() const noexcept override { return msg_.data(); }
    std::string encode() const;

private:
    Kind             kind_;
    ::rust::String   msg_;
};

// ----- Engine state --------------------------------------------------
//
// Owned by Engine, allocated lazily. Definition lives in
// StubEngine.cpp (stub) or LlamaEngine.cpp (real llama.cpp).

struct EngineState {
    std::map<::rust::String, ModelInfo> models;
    ::rust::String active;
    std::mutex m;
};

// ----- Engine ---------------------------------------------------------

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&)                 = delete;
    Engine& operator=(Engine&&)      = delete;

    // ----- models (mutating) ----------------------------------------
    ModelInfo load_model(::rust::Str id, const ModelLoadSpec& spec);
    void      unload_model(::rust::Str id);
    void      set_active(::rust::Str id);
    void      cancel();

    // ----- models (const) ------------------------------------------
    bool                   has_model(::rust::Str id) const;
    ::rust::Vec<ModelInfo> list_models() const;
    ModelInfo              model_info(::rust::Str id) const;
    ::rust::String         active_model() const;

    // ----- prompt rendering ----------------------------------------
    ::rust::String render_prompt(::rust::Str model_id,
                                 const ::rust::Vec<ChatMessage>& msgs) const;

    // ----- completion ----------------------------------------------
    CompletionResult complete(::rust::Str model_id,
                              const ::rust::Vec<ChatMessage>& msgs,
                              const ::rust::Vec<ToolSpec>& tools,
                              const SamplingParams& params);

    void complete_stream(::rust::Str model_id,
                         const ::rust::Vec<ChatMessage>& msgs,
                         const ::rust::Vec<ToolSpec>& tools,
                         const SamplingParams& params,
                         ::rust::Fn<void(TokenDelta)> cb);

    // ----- introspection -------------------------------------------
    ::rust::String detect_chat_template(::rust::Str model_id) const;
    ::rust::String detect_tool_format(::rust::Str model_id) const;

    // ----- helpers (used by free-function trampolines) ------------
    std::string detect_arch(const std::string& gguf_path);
    std::string detect_family(const std::string& arch);
    std::string model_family(const std::string& id) const;

    std::shared_ptr<EngineState> state;
};

// ----- Free function exported to cxx --------------------------------
//
// cxx links these by name.

std::unique_ptr<Engine> harness_engine_new();

// ----- Engine operation wrappers -------------------------------------
//
// Pin<&mut Engine> at the Rust boundary maps to `Engine&` at the C++
// boundary; `&Engine` maps to `const Engine&`.

ModelInfo engine_load_model(Engine& engine, ::rust::Str id, const ModelLoadSpec& spec);
void      engine_unload_model(Engine& engine, ::rust::Str id);
void      engine_set_active(Engine& engine, ::rust::Str id);
void      engine_cancel(Engine& engine);

bool                  engine_has_model(const Engine& engine, ::rust::Str id);
::rust::Vec<ModelInfo> engine_list_models(const Engine& engine);
ModelInfo             engine_model_info(const Engine& engine, ::rust::Str id);
::rust::String        engine_active_model(const Engine& engine);

::rust::String engine_render_prompt(const Engine& engine, ::rust::Str model_id,
                                    const ::rust::Vec<ChatMessage>& msgs);

CompletionResult engine_complete(Engine& engine, ::rust::Str model_id,
                                 const ::rust::Vec<ChatMessage>& msgs,
                                 const ::rust::Vec<ToolSpec>& tools,
                                 const SamplingParams& params);

void engine_complete_stream(Engine& engine, ::rust::Str model_id,
                            const ::rust::Vec<ChatMessage>& msgs,
                            const ::rust::Vec<ToolSpec>& tools,
                            const SamplingParams& params,
                            ::rust::Fn<void(TokenDelta)> cb);

::rust::String engine_detect_chat_template(const Engine& engine, ::rust::Str id);
::rust::String engine_detect_tool_format(const Engine& engine, ::rust::Str id);

}  // namespace harness
