//! llama-harness-ffi -- cxx bridge to the C++ harness engine.
//!
//! Most users want `llama-harness` (the safe Rust layer) instead of
//! calling this crate directly.

#![allow(clippy::needless_lifetimes)]

// cxx bridge module. The `#[cxx::bridge]` attribute generates a public
// `ffi` submodule containing the shared DTOs and free functions; the
// type aliases (`ffi::Engine`, `ffi::ChatMessage`, ...) are what callers
// use directly.
#[cxx::bridge(namespace = "harness")]
pub mod ffi {
    // ----- DTOs ----------------------------------------------------
    #[derive(Debug, Clone)]
    struct SamplingParams {
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        repeat_penalty: f32,
        repeat_last_n: i32,
        seed: i32,
        max_tokens: i32,
        stop: Vec<String>,
    }

    #[derive(Debug, Clone)]
    struct ModelLoadSpec {
        path: String,
        context_size: i32,
        batch_size: i32,
        threads: i32,
        gpu_layers: i32,
        use_mmap: bool,
        use_mlock: bool,
        chat_template_name: String,
    }

    #[derive(Debug, Clone)]
    struct ModelInfo {
        id: String,
        path: String,
        arch: String,
        family: String,
        context_size: i32,
        size_bytes: u64,
        n_params_b: i32,
        supports_tools: bool,
    }

    #[derive(Debug, Clone)]
    struct ChatMessage {
        role: String,
        content: String,
        tool_call_id: String,
        tool_name: String,
        tool_calls_json: String,
    }

    #[derive(Debug, Clone)]
    struct ToolSpec {
        name: String,
        description: String,
        parameters_json: String,
    }

    #[derive(Debug, Clone)]
    struct TokenDelta {
        text: String,
        token_id: i32,
        is_final: bool,
        cancelled: bool,
        stop_reason: String,
    }

    #[derive(Debug, Clone)]
    struct CompletionResult {
        text: String,
        tool_calls_json: String,
        prompt_tokens: i32,
        completion_tokens: i32,
        stop_reason: String,
    }

    // ----- Trampolines --------------------------------------------
    // cxx 1.0.202+ requires the block to be marked `unsafe` because the
    // functions can be called from any thread without Rust's borrow checker
    // being able to reason about C++ invariants.
    //
    // Methods on opaque types like `Engine` are not callable from outside
    // this module, so we expose Engine operations as free functions that
    // take a `&Engine` or `Pin<&mut Engine>`. Const methods become
    // `unsafe extern "C++"` free fns that require the C++ side to be `const`.
    unsafe extern "C++" {
        include!("harness.h");

        type Engine;

        // Constructor (returns a UniquePtr<Engine>).
        fn harness_engine_new() -> UniquePtr<Engine>;

        // ----- models (mutating) ---------------------------------
        fn engine_load_model(
            engine: Pin<&mut Engine>,
            id: &str,
            spec: &ModelLoadSpec,
        ) -> ModelInfo;
        fn engine_unload_model(engine: Pin<&mut Engine>, id: &str);
        fn engine_set_active(engine: Pin<&mut Engine>, id: &str);
        fn engine_cancel(engine: Pin<&mut Engine>);

        // ----- models (const) ------------------------------------
        fn engine_has_model(engine: &Engine, id: &str) -> bool;
        fn engine_list_models(engine: &Engine) -> Vec<ModelInfo>;
        fn engine_model_info(engine: &Engine, id: &str) -> ModelInfo;
        fn engine_active_model(engine: &Engine) -> String;

        // ----- prompt rendering -----------------------------------
        fn engine_render_prompt(
            engine: &Engine,
            model_id: &str,
            msgs: &Vec<ChatMessage>,
        ) -> String;

        // ----- completion -----------------------------------------
        fn engine_complete(
            engine: Pin<&mut Engine>,
            model_id: &str,
            msgs: &Vec<ChatMessage>,
            tools: &Vec<ToolSpec>,
            params: &SamplingParams,
        ) -> CompletionResult;

        fn engine_complete_stream(
            engine: Pin<&mut Engine>,
            model_id: &str,
            msgs: &Vec<ChatMessage>,
            tools: &Vec<ToolSpec>,
            params: &SamplingParams,
            cb: fn(TokenDelta),
        );

        // ----- introspection --------------------------------------
        fn engine_detect_chat_template(engine: &Engine, id: &str) -> String;
        fn engine_detect_tool_format(engine: &Engine, id: &str) -> String;
    }
}
