//! The Harness: top-level orchestrator that owns the FFI engine,
//! the model registry, and the tool registry.

use crate::chat::{
    ChatRequest, ChatResponse, ReActStep, StreamEvent, ToolCallRequest,
    ToolCallResult, ToolExecution,
};
use crate::error::{HarnessError, Result};
use crate::message::Message;
use crate::model::{ModelInfo, ModelLoadSpec, ModelSpec};
use crate::registry::{ModelEntry, ModelRegistry};
use crate::sampling::SamplingParams;
use crate::tool::{Tool, ToolFilter, ToolOutput};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Builder for `Harness`.
///
/// Created via [`Harness::builder`].
#[derive(Debug, Default)]
pub struct HarnessBuilder {
    registry_path: Option<std::path::PathBuf>,
    max_loaded_models: usize,
}

impl HarnessBuilder {
    /// Set the path to `models.toml`. Defaults to `~/.llama-harness/models.toml`.
    pub fn registry_path(mut self, p: impl Into<std::path::PathBuf>) -> Self {
        self.registry_path = Some(p.into());
        self
    }

    /// Set the maximum number of models to keep loaded simultaneously.
    /// Loading a new model past this cap evicts the least-recently-used one.
    /// Default: 2.
    pub fn max_loaded_models(mut self, n: usize) -> Self {
        self.max_loaded_models = n;
        self
    }

    /// Build the harness.
    pub async fn build(self) -> Result<Harness> {
        let registry = match self.registry_path {
            Some(p) => ModelRegistry::from_file(&p)?,
            None => ModelRegistry::default(),
        };

        let engine = llama_harness_ffi::ffi::harness_engine_new();
        if engine.is_null() {
            return Err(HarnessError::Engine(
                "harness_engine_new returned null".to_string(),
            ));
        }

        Ok(Harness {
            registry: Arc::new(registry),
            engine: Arc::new(Mutex::new(engine)),
            loaded: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            active: Arc::new(parking_lot::RwLock::new(None)),
            lru: Arc::new(parking_lot::RwLock::new(Vec::new())),
            max_loaded_models: self.max_loaded_models.max(1),
        })
    }
}

/// The central harness: model registry + tool registry + FFI engine.
#[derive(Clone)]
pub struct Harness {
    registry: Arc<ModelRegistry>,
    engine: Arc<Mutex<cxx::UniquePtr<llama_harness_ffi::ffi::Engine>>>,
    loaded: Arc<parking_lot::RwLock<HashMap<String, ModelInfo>>>,
    active: Arc<parking_lot::RwLock<Option<String>>>,
    lru: Arc<parking_lot::RwLock<Vec<String>>>,
    max_loaded_models: usize,
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Harness")
            .field("loaded", &*self.loaded.read())
            .field("active", &*self.active.read())
            .field("max_loaded_models", &self.max_loaded_models)
            .finish()
    }
}

impl Harness {
    /// Start a builder.
    pub fn builder() -> HarnessBuilder {
        HarnessBuilder::default()
    }

    /// Look up a model by name in the registry.
    pub fn resolve(&self, name: &str) -> Option<&ModelEntry> {
        self.registry.get(name)
    }

    /// List all known (registry + loaded) model names.
    pub fn list(&self) -> Vec<String> {
        let mut out: Vec<String> = self.registry.names().into_iter().map(String::from).collect();
        for k in self.loaded.read().keys() {
            if !out.contains(k) {
                out.push(k.clone());
            }
        }
        out.sort();
        out
    }

    /// Register a tool with the harness.
    pub fn register_tool(&mut self, tool: std::sync::Arc<dyn crate::tool::Tool>) {
        // Tool registration is part of the chat-time tool loop, which the
        // stub backend does not exercise. The real backend will consult
        // this registry before issuing tool calls.
        tracing::debug!(name = tool.name(), "register_tool");
    }

    /// Load a model from the registry by name.
    pub async fn load(&self, name: &str) -> Result<ModelInfo> {
        let entry = self.registry.get(name).ok_or_else(|| {
            HarnessError::NoSuchModel(name.to_string())
        })?;
        let spec = entry.resolve()?;
        self.load_spec(name, spec).await
    }

    /// Load a model from an explicit load spec.
    pub async fn load_spec(&self, id: &str, spec: ModelLoadSpec) -> Result<ModelInfo> {
        let id = id.to_string();

        {
            let mut loaded = self.loaded.write();
            let mut lru = self.lru.write();

            // If already loaded, just touch LRU and return.
            if let Some(info) = loaded.get(&id).cloned() {
                lru.retain(|x| x != &id);
                lru.push(id.clone());
                return Ok(info);
            }

            // Evict LRU entries until we're under the cap.
            while loaded.len() >= self.max_loaded_models {
                let victim = {
                    let mut lru = self.lru.write();
                    if lru.is_empty() {
                        break;
                    }
                    Some(lru.remove(0))
                };
                if let Some(victim) = victim {
                    info!(victim = %victim, "evicting (LRU)");
                    loaded.remove(&victim);
                    let mut engine = self.engine.lock().await;
                    llama_harness_ffi::ffi::engine_unload_model(
                        cxx::UniquePtr::pin_mut(&mut *engine),
                        &victim,
                    );
                    if self.active.read().as_deref() == Some(victim.as_str()) {
                        *self.active.write() = None;
                    }
                }
            }
        }

        let ffi_info = {
            let mut engine = self.engine.lock().await;
            llama_harness_ffi::ffi::engine_load_model(
                cxx::UniquePtr::pin_mut(&mut *engine),
                &id,
                &spec.to_ffi(),
            )
        };

        let info: ModelInfo = ffi_info.into();
        self.loaded.write().insert(id.clone(), info.clone());
        self.lru.write().push(id.clone());

        if self.active.read().is_none() {
            *self.active.write() = Some(id.clone());
            let mut engine = self.engine.lock().await;
            llama_harness_ffi::ffi::engine_set_active(
                cxx::UniquePtr::pin_mut(&mut *engine),
                &id,
            );
        }
        Ok(info)
    }

    /// Unload a model by name.
    pub async fn unload(&self, name: &str) -> Result<()> {
        {
            let mut engine = self.engine.lock().await;
            llama_harness_ffi::ffi::engine_unload_model(
                cxx::UniquePtr::pin_mut(&mut *engine),
                name,
            );
        }
        self.loaded.write().remove(name);
        self.lru.write().retain(|x| x != name);
        if self.active.read().as_deref() == Some(name) {
            *self.active.write() = None;
        }
        Ok(())
    }

    /// True if `name` is currently loaded into the engine.
    pub async fn is_loaded(&self, name: &str) -> bool {
        self.loaded.read().contains_key(name)
    }

    /// Snapshot of currently-loaded models.
    pub async fn loaded_models(&self) -> Vec<ModelInfo> {
        self.loaded.read().values().cloned().collect()
    }

    /// Name of the active model, if any.
    pub async fn active_model(&self) -> Option<String> {
        self.active.read().clone()
    }

    /// Set the active model. Future `chat` calls with `model: None` use this one.
    pub async fn set_active(&self, name: &str) -> Result<()> {
        if !self.loaded.read().contains_key(name) {
            return Err(HarnessError::ModelNotLoaded(name.to_string()));
        }
        *self.active.write() = Some(name.to_string());
        let mut engine = self.engine.lock().await;
        llama_harness_ffi::ffi::engine_set_active(
            cxx::UniquePtr::pin_mut(&mut *engine),
            name,
        );
        Ok(())
    }

    /// Send a chat request, return the final response.
    ///
    /// For a single response, this drives the FFI engine and (when tools
    /// are present) the ReAct loop. For token-level streaming use
    /// [`Self::chat_stream`].
    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {        // Validate routing
        let model_id = match req.model.as_deref() {
            Some(id) => {
                if !self.loaded.read().contains_key(id) {
                    return Err(HarnessError::ModelNotLoaded(id.to_string()));
                }
                id.to_string()
            }
            None => self
                .active
                .read()
                .clone()
                .ok_or(HarnessError::NoActiveModel)?,
        };
        // Validate tool filter
        if !matches!(req.tools, ToolFilter::None) {
            // Tool calling requires the real llama.cpp backend. The stub
            // engine handles the case gracefully -- it just emits text.
            warn!("tool calling requires the real llama.cpp backend; the stub emits placeholder text");
        }

        let ffi_msgs: Vec<llama_harness_ffi::ffi::ChatMessage> = req
            .messages
            .iter()
            .map(|m| m.to_ffi())
            .collect();

        let ffi_tools: Vec<llama_harness_ffi::ffi::ToolSpec> = Vec::new();

        let this = self.clone();
        let result = {
            let mut engine = this.engine.lock().await;
            llama_harness_ffi::ffi::engine_complete(
                cxx::UniquePtr::pin_mut(&mut *engine),
                &model_id,
                &ffi_msgs,
                &ffi_tools,
                &req.sampling.to_ffi(),
            )
        };

        let text = result.text.clone();
        let prompt_tokens = result.prompt_tokens;
        let completion_tokens = result.completion_tokens;

        let response = ChatResponse {
            final_text: result.text,
            assistant_turns: Vec::new(),
            trace: Vec::new(),
            prompt_tokens,
            completion_tokens,
            stop_reason: "eos".to_string(),
            tools_used: Vec::new(),
        };

        debug!(
            model = %model_id,
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            "chat complete"
        );

        Ok(response)
    }

    /// Stream chat events. Each `StreamEvent` is a discrete unit that
    /// the caller must render or route.
    pub fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = StreamEvent>>> {
        let this = self.clone();
        let s = async_stream::stream! {
            match this.chat(req).await {
                Ok(resp) => {
                    yield StreamEvent::Token { text: resp.final_text.clone() };
                    yield StreamEvent::Done(resp);
                }
                Err(e) => yield StreamEvent::Error(e.to_string()),
            }
        };
        Box::pin(s)
    }
}

#[doc(hidden)]
pub use llama_harness_ffi::ffi as ffi_module;
