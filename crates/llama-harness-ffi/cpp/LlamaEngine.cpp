// SPDX-License-Identifier: MIT OR Apache-2.0
//
// LlamaEngine.cpp -- real llama.cpp backend implementing the same
// `Engine` class surface as StubEngine.cpp. Compiled when
// LLAMA_HARNESS_REAL_BACKEND is defined (set by build.rs when
// vendor/llama.cpp is present and built).
//
// Threading model: one Engine per process is the normal case. The
// Rust layer serializes all calls behind tokio::sync::Mutex, so the
// llama handles map needs no separate lock beyond EngineState::m.

#include "harness.h"
#include "lib.rs.h"

#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace harness {

// ----- Utilities --------------------------------------------------------

static std::string to_std(const ::rust::String & s) {
    return std::string(s.data(), s.size());
}

static std::string to_std(::rust::Str s) {
    return std::string(s.data(), s.size());
}

static ::rust::String to_rust(const std::string & s) {
    return ::rust::String(s);
}

static std::shared_ptr<EngineState> state_of(Engine & e) {
    if (!e.state) {
        e.state = std::make_shared<EngineState>();
    }
    return e.state;
}

struct LlamaModel {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;

    std::string chat_template;   // raw template string (GGUF or builtin)
    int32_t context_size = 0;
    int32_t n_vocab = 0;
};

// Handle registry: llama.cpp pointers keyed by the same model ids the
// stub uses for its DTO map. Kept in a process-wide table keyed by
// EngineState* so the EngineState ABI shared with the stub is unchanged.
struct Engine::LlamaHandles {
    std::map<std::string, std::unique_ptr<LlamaModel>> by_id;
    std::atomic<bool> cancel{false};
};

using LlamaHandles = Engine::LlamaHandles;

static Engine::LlamaHandles & handles_of(Engine & e) {
    if (!e.state) {
        e.state = std::make_shared<EngineState>();
    }
    static std::mutex m;
    static std::map<EngineState *, std::unique_ptr<Engine::LlamaHandles>> table;
    std::lock_guard<std::mutex> g(m);
    auto it = table.find(e.state.get());
    if (it == table.end()) {
        it = table.emplace(e.state.get(),
            std::make_unique<Engine::LlamaHandles>()).first;
    }
    return *it->second;
}

static LlamaModel & get_model(Engine & e, const std::string & id) {
    auto & h = handles_of(e);
    auto it = h.by_id.find(id);
    if (it == h.by_id.end()) {
        throw HarnessError(HarnessError::Kind::NoSuchModel, to_rust(id));
    }
    return *it->second;
}

// ----- GGUF metadata probing ---------------------------------------------

struct GgufMeta {
    std::string arch;
    uint64_t size_bytes = 0;
};

static GgufMeta probe_gguf(const std::string & path) {
    GgufMeta out;
    gguf_init_params p;
    p.no_alloc = true;
    p.ctx = nullptr;
    gguf_context * gguf = gguf_init_from_file(path.c_str(), p);
    if (gguf) {
        const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
        if (arch_id >= 0) {
            const char * a = gguf_get_val_str(gguf, arch_id);
            if (a) out.arch = a;
        }
        gguf_free(gguf);
    }
    if (FILE * f = std::fopen(path.c_str(), "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fclose(f);
        if (sz > 0) out.size_bytes = static_cast<uint64_t>(sz);
    }
    return out;
}

std::string Engine::detect_arch(const std::string & gguf_path) {
    GgufMeta m = probe_gguf(gguf_path);
    return m.arch.empty() ? std::string("unknown") : m.arch;
}

std::string Engine::detect_family(const std::string & arch) {
    if (arch == "llama")  return "llama-3";
    if (arch == "mistral" || arch == "mixtral") return "mistral";
    if (arch == "qwen2" || arch == "qwen3")     return "qwen";
    if (arch == "gemma" || arch == "gemma2" || arch == "gemma3") return "gemma";
    if (arch == "phi2" || arch == "phi3")       return "phi";
    return arch;
}

// Chat template resolution: prefer the model's own GGUF template, fall
// back to a builtin name llama_chat_apply_template recognizes.
static std::string resolve_template(const LlamaModel & lm) {
    if (const char * t = llama_model_chat_template(lm.model, nullptr)) {
        return t;
    }
    const std::string arch = [&lm] {
        const char * a = nullptr;
        return std::string(a ? a : "");
    }();
    (void)arch;
    // Fallback by vocab characteristics is unreliable; chatml is the
    // most compatible builtin for models without an embedded template.
    return std::string("chatml");
}

// ----- Engine lifecycle -----------------------------------------------------

Engine::Engine()  = default;
Engine::~Engine() = default;

// ----- Model management ------------------------------------------------------

ModelInfo Engine::load_model(::rust::Str id, const ModelLoadSpec & spec) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);

    const std::string rid = std::string(id.data(), id.size());
    const std::string path = to_std(spec.path);

    if (handles_of(*this).by_id.count(rid) != 0) {
        throw HarnessError(HarnessError::Kind::InvalidArg,
                           to_rust("model already loaded: " + rid));
    }

    auto lm = std::make_unique<LlamaModel>();

    // --- load model ---
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = spec.gpu_layers;
    if (spec.chat_template_name.size() > 0) {
        // Non-empty template name forces builtin selection later.
    }
    lm->model = llama_model_load_from_file(path.c_str(), mp);
    if (!lm->model) {
        throw HarnessError(HarnessError::Kind::Backend,
                           to_rust("failed to load model from " + path));
    }
    lm->vocab = llama_model_get_vocab(lm->model);
    lm->n_vocab = llama_vocab_n_tokens(lm->vocab);

    // --- context ---
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = spec.context_size > 0
        ? static_cast<uint32_t>(spec.context_size) : 4096u;
    cp.n_batch = spec.batch_size > 0
        ? static_cast<uint32_t>(spec.batch_size) : 2048u;
    cp.n_ubatch = cp.n_batch;
    int32_t n_threads = spec.threads;
    if (n_threads <= 0) {
        n_threads = static_cast<int32_t>(
            std::max(1u, std::thread::hardware_concurrency()));
    }
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;

    lm->ctx = llama_init_from_model(lm->model, cp);
    if (!lm->ctx) {
        llama_model_free(lm->model);
        throw HarnessError(HarnessError::Kind::Backend,
                           to_rust("failed to create context for " + rid));
    }

    // --- metadata ---
    GgufMeta meta = probe_gguf(path);
    const std::string arch = meta.arch.empty() ? "unknown" : meta.arch;
    const std::string family = detect_family(arch);
    lm->context_size = static_cast<int32_t>(llama_n_ctx(lm->ctx));
    lm->chat_template = resolve_template(*lm);

    ModelInfo info;
    info.id = to_rust(rid);
    info.path = spec.path;
    info.arch = to_rust(arch);
    info.family = to_rust(family);
    info.context_size = lm->context_size;
    info.size_bytes = meta.size_bytes;
    info.n_params_b = static_cast<int32_t>(
        static_cast<int64_t>(llama_model_n_params(lm->model)) / 1000000000LL);
    info.supports_tools = true;

    handles_of(*this).by_id[rid] = std::move(lm);

    s->models[to_rust(rid)] = info;
    if (s->active.empty()) {
        s->active = info.id;
    }
    return info;
}

void Engine::unload_model(::rust::Str id) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);

    const std::string rid = std::string(id.data(), id.size());
    auto & h = handles_of(*this);
    auto it = h.by_id.find(rid);
    if (it == h.by_id.end()) {
        return;  // idempotent, matches stub behavior
    }
    if (it->second->ctx) {
        llama_free(it->second->ctx);
    }
    if (it->second->model) {
        llama_model_free(it->second->model);
    }
    h.by_id.erase(it);

    s->models.erase(::rust::String(rid));
    if (s->active == ::rust::String(rid)) {
        s->active = ::rust::String("");
        if (!s->models.empty()) {
            s->active = s->models.begin()->first;
        }
    }
}

void Engine::set_active(::rust::Str id) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);
    const std::string rid = std::string(id.data(), id.size());
    if (handles_of(*this).by_id.count(rid) == 0) {
        throw HarnessError(HarnessError::Kind::NoSuchModel, to_rust(rid));
    }
    s->active = ::rust::String(rid);
}

void Engine::cancel() {
    handles_of(*this).cancel.store(true, std::memory_order_relaxed);
}

bool Engine::has_model(::rust::Str id) const {
    if (!state) return false;
    std::lock_guard<std::mutex> g(state->m);
    return state->models.count(::rust::String(id.data(), id.size())) > 0;
}

ModelInfo Engine::model_info(::rust::Str id) const {
    ::rust::String rid(std::string(id.data(), id.size()));
    if (!state) {
        throw HarnessError(HarnessError::Kind::NoSuchModel, rid);
    }
    std::lock_guard<std::mutex> g(state->m);
    auto it = state->models.find(rid);
    if (it == state->models.end()) {
        throw HarnessError(HarnessError::Kind::NoSuchModel, rid);
    }
    return it->second;
}

::rust::Vec<ModelInfo> Engine::list_models() const {
    ::rust::Vec<ModelInfo> out;
    if (!state) return out;
    std::lock_guard<std::mutex> g(state->m);
    out.reserve(state->models.size());
    for (auto & kv : state->models) out.push_back(kv.second);
    return out;
}

::rust::String Engine::active_model() const {
    if (!state) return ::rust::String("");
    std::lock_guard<std::mutex> g(state->m);
    return state->active;
}

// ----- Helpers ---------------------------------------------------------------

std::string Engine::model_family(const std::string & id) const {
    if (!state) return "unknown";
    std::lock_guard<std::mutex> g(state->m);
    auto it = state->models.find(::rust::String(id));
    if (it == state->models.end()) return "unknown";
    return to_std(it->second.family);
}

::rust::String Engine::detect_chat_template(::rust::Str model_id) const {
    (void)model_id;
    return ::rust::String("gguf");  // resolved at load; see render_prompt
}

::rust::String Engine::detect_tool_format(::rust::Str model_id) const {
    const std::string fam =
        model_family(std::string(model_id.data(), model_id.size()));
    if (fam == "llama-3") return ::rust::String("llama3");
    if (fam == "mistral") return ::rust::String("mistral");
    if (fam == "qwen")    return ::rust::String("qwen");
    return ::rust::String("json");
}

::rust::String Engine::render_prompt(
    ::rust::Str model_id,
    const ::rust::Vec<ChatMessage> & msgs) const
{
    const std::string mid = std::string(model_id.data(), model_id.size());
    std::string tmpl = "chatml";
    {
        auto & h = handles_of(const_cast<Engine &>(*this));
        auto it = h.by_id.find(mid);
        if (it != h.by_id.end()) {
            tmpl = it->second->chat_template;
        }
    }

    std::vector<llama_chat_message> chat;
    std::vector<std::string> roles, contents;
    roles.reserve(msgs.size());
    contents.reserve(msgs.size());
    for (const auto & m : msgs) {
        roles.push_back(to_std(m.role));
        contents.push_back(to_std(m.content));
        chat.push_back({ roles.back().c_str(), contents.back().c_str() });
    }
    size_t total = 64;
    for (const auto & c : contents) total += c.size() * 2;
    std::vector<char> buf(total);
    const int32_t n = llama_chat_apply_template(
        tmpl.c_str(), chat.data(), chat.size(), /*add_ass=*/true,
        buf.data(), static_cast<int32_t>(buf.size()));
    if (n < 0) {
        throw HarnessError(HarnessError::Kind::Backend,
                           to_rust("chat template application failed"));
    }
    const size_t out_len = static_cast<size_t>(
        std::min<int32_t>(n, static_cast<int32_t>(buf.size()) - 1));
    return ::rust::String(std::string(buf.data(), out_len));
}

// ----- Generation core --------------------------------------------------------

namespace {

std::vector<llama_token> tokenize(const llama_vocab * vocab,
                                  const std::string & text) {
    const int32_t n = static_cast<int32_t>(text.size());
    std::vector<llama_token> toks(static_cast<size_t>(n) + 8);
    int32_t ntok = llama_tokenize(vocab, text.data(), n, toks.data(),
        static_cast<int32_t>(toks.size()),
        /*add_special=*/true, /*parse_special=*/false);
    if (ntok < 0) {
        toks.resize(static_cast<size_t>(-ntok));
        ntok = llama_tokenize(vocab, text.data(), n, toks.data(),
            static_cast<int32_t>(toks.size()), true, false);
        if (ntok < 0) {
            throw HarnessError(HarnessError::Kind::Tokenization,
                               to_rust("tokenize failed"));
        }
    }
    toks.resize(static_cast<size_t>(ntok));
    return toks;
}

std::string token_to_piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, t, buf,
        static_cast<int32_t>(sizeof(buf)),
        /*lstrip=*/0, /*special=*/false);
    if (n < 0) return "";
    return std::string(buf, static_cast<size_t>(n));
}

llama_sampler * build_sampler(const SamplingParams & p, int32_t n_vocab) {
    llama_sampler * chain = llama_sampler_chain_init(
        llama_sampler_chain_default_params());
    if (p.min_p > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(p.min_p, 1));
    }
    if (p.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
    }
    if (p.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, 1));
    }
    if (p.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temperature));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }
    if (p.repeat_penalty != 1.0f && p.repeat_last_n != 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
            n_vocab, p.repeat_last_n, p.repeat_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_dist(
        p.seed >= 0 ? static_cast<uint32_t>(p.seed) : LLAMA_DEFAULT_SEED));
    return chain;
}

// Decode-and-sample loop shared by complete() and complete_stream().
// on_token returns false to abort. Token accounting: llama.cpp does not
// expose a per-context "tokens sampled" counter, so we track it here.
CompletionResult generate(Engine & e,
                          LlamaModel & lm,
                          const std::string & prompt_text,
                          const SamplingParams & params,
                          const std::function<bool(const std::string &)> & on_token)
{
    auto & h = handles_of(e);
    const llama_vocab * vocab = lm.vocab;

    std::vector<llama_token> tokens = tokenize(vocab, prompt_text);
    if (tokens.empty()) {
        throw HarnessError(HarnessError::Kind::Backend,
                           to_rust("empty prompt after tokenization"));
    }

    // Fresh single-turn completion: clear KV so prior runs do not leak.
    llama_memory_clear(llama_get_memory(lm.ctx), /*data=*/false);

    llama_sampler * sampler = build_sampler(params, lm.n_vocab);
    h.cancel.store(false, std::memory_order_relaxed);

    CompletionResult result;
    result.prompt_tokens = static_cast<int32_t>(tokens.size());
    result.completion_tokens = 0;

    // Prefill.
    llama_batch batch = llama_batch_get_one(tokens.data(),
        static_cast<int32_t>(tokens.size()));
    if (llama_decode(lm.ctx, batch) != 0) {
        llama_sampler_free(sampler);
        throw HarnessError(HarnessError::Kind::Backend,
                           to_rust("prompt decode failed"));
    }

    // Generation loop.
    std::string full;
    for (int32_t i = 0; i < params.max_tokens; ++i) {
        if (h.cancel.load(std::memory_order_relaxed)) {
            result.stop_reason = to_rust("cancelled");
            break;
        }
        const llama_token next = llama_sampler_sample(sampler, lm.ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            result.stop_reason = to_rust("eos");
            break;
        }
        const std::string piece = token_to_piece(vocab, next);
        full += piece;
        ++result.completion_tokens;

        bool stop_hit = false;
        for (const auto & s : params.stop) {
            const std::string stop = to_std(s);
            if (!stop.empty() && full.size() >= stop.size() &&
                full.compare(full.size() - stop.size(), stop.size(), stop) == 0)
            {
                stop_hit = true;
                break;
            }
        }
        if (on_token && !on_token(piece)) {
            result.stop_reason = to_rust("cancelled");
            break;
        }
        if (stop_hit) {
            result.stop_reason = to_rust("stop");
            break;
        }

        llama_token next_mut = next;
        llama_batch one = llama_batch_get_one(&next_mut, 1);
        const int rc = llama_decode(lm.ctx, one);
        if (rc != 0) {
            result.stop_reason = to_rust(rc == 1 ? "context_full" : "decode_error");
            break;
        }
    }
    if (result.stop_reason.empty()) {
        result.stop_reason = to_rust("length");
    }
    result.text = to_rust(full);
    llama_sampler_free(sampler);
    return result;
}

}  // namespace

// ----- Completion ------------------------------------------------------------

CompletionResult Engine::complete(::rust::Str model_id,
                                  const ::rust::Vec<ChatMessage> & msgs,
                                  const ::rust::Vec<ToolSpec> & tools,
                                  const SamplingParams & params)
{
    (void)tools;  // grammar-constrained tool sampling lands with GBNF work
    const std::string mid = std::string(model_id.data(), model_id.size());
    LlamaModel & lm = get_model(*this, mid);
    const std::string prompt = to_std(render_prompt(model_id, msgs));
    return generate(*this, lm, prompt, params, nullptr);
}

// cxx Fn is single-call: invoke a fresh copy per token.
void Engine::complete_stream(::rust::Str model_id,
                             const ::rust::Vec<ChatMessage> & msgs,
                             const ::rust::Vec<ToolSpec> & tools,
                             const SamplingParams & params,
                             ::rust::Fn<void(TokenDelta)> cb)
{
    (void)tools;
    const std::string mid = std::string(model_id.data(), model_id.size());
    LlamaModel & lm = get_model(*this, mid);
    const std::string prompt = to_std(render_prompt(model_id, msgs));
    auto & h = handles_of(*this);

    CompletionResult r = generate(*this, lm, prompt, params,
        [&](const std::string & piece) {
            TokenDelta d;
            d.text = to_rust(piece);
            d.token_id = 0;
            d.is_final = false;
            d.cancelled = h.cancel.load(std::memory_order_relaxed);
            d.stop_reason = ::rust::String("");
            cb(d);
            return !h.cancel.load(std::memory_order_relaxed);
        });

    TokenDelta fin;
    fin.text = ::rust::String("");
    fin.token_id = -1;
    fin.is_final = true;
    fin.cancelled = r.stop_reason == ::rust::String("cancelled");
    fin.stop_reason = r.stop_reason;
    cb(fin);
}

}  // namespace harness

// ----- cxx trampolines (same surface as StubEngine.cpp) -----------------------

namespace harness {

std::unique_ptr<harness::Engine> harness_engine_new() {
    return std::make_unique<harness::Engine>();
}

ModelInfo engine_load_model(::harness::Engine & e, ::rust::Str id, const ModelLoadSpec & spec) {
    return e.load_model(id, spec);
}

void engine_unload_model(::harness::Engine & e, ::rust::Str id) { e.unload_model(id); }
void engine_set_active(::harness::Engine & e, ::rust::Str id)   { e.set_active(id); }
void engine_cancel(::harness::Engine & e)                       { e.cancel(); }

bool engine_has_model(const ::harness::Engine & e, ::rust::Str id) {
    return e.has_model(id);
}

ModelInfo engine_model_info(const ::harness::Engine & e, ::rust::Str id) {
    return e.model_info(id);
}

::rust::Vec<ModelInfo> engine_list_models(const ::harness::Engine & e) {
    return e.list_models();
}

::rust::String engine_active_model(const ::harness::Engine & e) {
    return e.active_model();
}

::rust::String engine_render_prompt(const ::harness::Engine & e, ::rust::Str mid,
                                    const ::rust::Vec<ChatMessage> & msgs) {
    return e.render_prompt(mid, msgs);
}

CompletionResult engine_complete(::harness::Engine & e, ::rust::Str mid,
                                 const ::rust::Vec<ChatMessage> & msgs,
                                 const ::rust::Vec<ToolSpec> & tools,
                                 const SamplingParams & p) {
    return e.complete(mid, msgs, tools, p);
}

void engine_complete_stream(::harness::Engine & e, ::rust::Str mid,
                            const ::rust::Vec<ChatMessage> & msgs,
                            const ::rust::Vec<ToolSpec> & tools,
                            const SamplingParams & p,
                            ::rust::Fn<void(TokenDelta)> cb) {
    e.complete_stream(mid, msgs, tools, p, std::move(cb));
}

::rust::String engine_detect_chat_template(const ::harness::Engine & e, ::rust::Str mid) {
    return e.detect_chat_template(mid);
}

::rust::String engine_detect_tool_format(const ::harness::Engine & e, ::rust::Str mid) {
    return e.detect_tool_format(mid);
}

}  // namespace harness