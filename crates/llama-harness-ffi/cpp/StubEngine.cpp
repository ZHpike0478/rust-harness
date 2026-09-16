// SPDX-License-Identifier: MIT OR Apache-2.0
//
// StubEngine.cpp -- placeholder Engine impl for builds without llama.cpp.

#include "harness.h"
#include "lib.rs.h"

#include <chrono>
#include <map>
#include <sstream>

namespace harness {

Engine::Engine() = default;
Engine::~Engine() = default;

static std::shared_ptr<EngineState> state_of(Engine& e) {
    if (!e.state) e.state = std::make_shared<EngineState>();
    return e.state;
}

static std::string to_std(const ::rust::String& s) {
    return std::string(s.data(), s.size());
}

static ::rust::String to_rust(const std::string& s) {
    return ::rust::String(s);
}

// ----- Model management -----------------------------------------------

ModelInfo Engine::load_model(::rust::Str id, const ModelLoadSpec& spec) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);
    ::rust::String rid(std::string(id.data(), id.size()));
    ModelInfo info;
    info.id = rid;
    info.path = spec.path;
    info.arch = to_rust(detect_arch(to_std(spec.path)));
    info.family = to_rust(detect_family(to_std(info.arch)));
    info.context_size = spec.context_size;
    info.size_bytes = 0;
    info.n_params_b = 0;
    info.supports_tools = true;
    s->models[rid] = info;
    if (s->active.empty()) s->active = rid;
    return info;
}

void Engine::unload_model(::rust::Str id) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);
    ::rust::String rid(std::string(id.data(), id.size()));
    s->models.erase(rid);
    if (s->active == rid) {
        s->active = ::rust::String("");
        if (!s->models.empty()) s->active = s->models.begin()->first;
    }
}

void Engine::set_active(::rust::Str id) {
    auto s = state_of(*this);
    std::lock_guard<std::mutex> g(s->m);
    ::rust::String rid(std::string(id.data(), id.size()));
    if (s->models.count(rid) == 0) {
        throw HarnessError(HarnessError::Kind::NoSuchModel, rid);
    }
    s->active = rid;
}

void Engine::cancel() {}

bool Engine::has_model(::rust::Str id) const {
    if (!state) return false;
    ::rust::String rid(std::string(id.data(), id.size()));
    std::lock_guard<std::mutex> g(state->m);
    return state->models.count(rid) > 0;
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
    for (auto& kv : state->models) out.push_back(kv.second);
    return out;
}

::rust::String Engine::active_model() const {
    if (!state) return ::rust::String("");
    std::lock_guard<std::mutex> g(state->m);
    return state->active;
}

// ----- Detection -------------------------------------------------------

static std::string path_basename(const std::string& p) {
    auto sep = p.find_last_of("/\\");
    if (sep == std::string::npos) return p;
    return p.substr(sep + 1);
}

std::string Engine::detect_arch(const std::string& gguf_path) {
    std::string base = path_basename(gguf_path);
    for (auto& c : base) c = static_cast<char>(std::tolower(c));
    if (base.find("llama") != std::string::npos) return "llama";
    if (base.find("mistral") != std::string::npos) return "mistral";
    if (base.find("mixtral") != std::string::npos) return "mixtral";
    if (base.find("qwen") != std::string::npos) return "qwen2";
    if (base.find("phi") != std::string::npos) return "phi";
    if (base.find("gemma") != std::string::npos) return "gemma";
    return "unknown";
}

std::string Engine::detect_family(const std::string& arch) {
    if (arch == "llama") return "llama-3";
    if (arch == "mistral" || arch == "mixtral") return "mistral";
    if (arch == "qwen2") return "qwen";
    return arch;
}

std::string Engine::model_family(const std::string& id) const {
    if (!state) return "unknown";
    std::lock_guard<std::mutex> g(state->m);
    ::rust::String rid(std::string(id.data(), id.size()));
    auto it = state->models.find(rid);
    if (it == state->models.end()) return "unknown";
    return to_std(it->second.family);
}

::rust::String Engine::detect_chat_template(::rust::Str model_id) const {
    std::string fam = model_family(std::string(model_id.data(), model_id.size()));
    if (fam == "llama-3") return ::rust::String("llama-3");
    if (fam == "mistral") return ::rust::String("mistral");
    if (fam == "qwen")    return ::rust::String("chatml");
    return ::rust::String("chatml");
}

::rust::String Engine::detect_tool_format(::rust::Str model_id) const {
    std::string fam = model_family(std::string(model_id.data(), model_id.size()));
    if (fam == "llama-3") return ::rust::String("llama3");
    if (fam == "mistral") return ::rust::String("mistral");
    if (fam == "qwen")    return ::rust::String("qwen");
    return ::rust::String("json");
}

::rust::String Engine::render_prompt(
    ::rust::Str model_id,
    const ::rust::Vec<ChatMessage>& /*msgs*/) const
{
    return ::rust::String(std::string("[stub prompt for ") +
                          std::string(model_id.data(), model_id.size()) +
                          std::string("]"));
}

// ----- Completion ------------------------------------------------------

CompletionResult Engine::complete(::rust::Str model_id,
                                  const ::rust::Vec<ChatMessage>& msgs,
                                  const ::rust::Vec<ToolSpec>& tools,
                                  const SamplingParams& params) {
    CompletionResult result;
    std::ostringstream out;
    out << "[stub:model=" << std::string(model_id.data(), model_id.size())
        << "] received " << static_cast<int>(msgs.size()) << " message(s) and "
        << static_cast<int>(tools.size())
        << " tool(s). temp=" << params.temperature
        << " top_p=" << params.top_p
        << " top_k=" << params.top_k << ".";
    result.text = ::rust::String(out.str());
    result.prompt_tokens = static_cast<int32_t>(msgs.size() * 16);
    result.completion_tokens = static_cast<int32_t>(result.text.size() / 4);
    return result;
}

void Engine::complete_stream(::rust::Str model_id,
                             const ::rust::Vec<ChatMessage>& msgs,
                             const ::rust::Vec<ToolSpec>& tools,
                             const SamplingParams& params,
                             ::rust::Fn<void(TokenDelta)> cb) {
    CompletionResult full = complete(model_id, msgs, tools, params);
    TokenDelta d;
    d.text = full.text;
    d.is_final = true;
    cb(d);
}

}  // namespace harness

// ----- cxx trampoline: free functions matching `extern "C++"` ---------

namespace harness {

std::unique_ptr<harness::Engine> harness_engine_new() {
    return std::make_unique<harness::Engine>();
}

ModelInfo engine_load_model(::harness::Engine& e, ::rust::Str id, const ModelLoadSpec& spec) {
    return e.load_model(id, spec);
}

void engine_unload_model(::harness::Engine& e, ::rust::Str id) { e.unload_model(id); }
void engine_set_active(::harness::Engine& e, ::rust::Str id) { e.set_active(id); }
void engine_cancel(::harness::Engine& e) { e.cancel(); }

bool engine_has_model(const ::harness::Engine& e, ::rust::Str id) {
    return e.has_model(id);
}

ModelInfo engine_model_info(const ::harness::Engine& e, ::rust::Str id) {
    return e.model_info(id);
}

::rust::Vec<ModelInfo> engine_list_models(const ::harness::Engine& e) {
    return e.list_models();
}

::rust::String engine_active_model(const ::harness::Engine& e) {
    return e.active_model();
}

::rust::String engine_render_prompt(const ::harness::Engine& e, ::rust::Str mid,
                                    const ::rust::Vec<ChatMessage>& msgs) {
    return e.render_prompt(mid, msgs);
}

CompletionResult engine_complete(::harness::Engine& e, ::rust::Str mid,
                                 const ::rust::Vec<ChatMessage>& msgs,
                                 const ::rust::Vec<ToolSpec>& tools,
                                 const SamplingParams& p) {
    return e.complete(mid, msgs, tools, p);
}

void engine_complete_stream(::harness::Engine& e, ::rust::Str mid,
                            const ::rust::Vec<ChatMessage>& msgs,
                            const ::rust::Vec<ToolSpec>& tools,
                            const SamplingParams& p,
                            ::rust::Fn<void(TokenDelta)> cb) {
    e.complete_stream(mid, msgs, tools, p, std::move(cb));
}

::rust::String engine_detect_chat_template(const ::harness::Engine& e, ::rust::Str mid) {
    return e.detect_chat_template(mid);
}

::rust::String engine_detect_tool_format(const ::harness::Engine& e, ::rust::Str mid) {
    return e.detect_tool_format(mid);
}

}  // namespace harness
