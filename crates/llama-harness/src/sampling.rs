//! Sampling parameters: temperature, top_k, top_p, min_p, repeat penalty.
//!
//! Defaults target a balanced chat generation profile. Override per
//! request via `ChatRequest { sampling, .. }`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature. 1.0 = neutral, <1.0 = sharper, >1.0 = wilder.
    pub temperature: f32,

    /// Top-p (nucleus) sampling cutoff.
    pub top_p: f32,

    /// Top-k cutoff. 0 = disabled.
    pub top_k: i32,

    /// Min-p sampling cutoff (newer alternative to top-p).
    pub min_p: f32,

    /// Repetition penalty factor. 1.0 = off.
    pub repeat_penalty: f32,

    /// How many recent tokens the repeat penalty considers.
    pub repeat_last_n: i32,

    /// RNG seed. -1 = random.
    pub seed: i32,

    /// Hard cap on tokens to generate.
    pub max_tokens: i32,

    /// Extra stop strings beyond the model's defaults.
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.10,
            repeat_last_n: 64,
            seed: -1,
            max_tokens: 512,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    /// Convert to the FFI DTO. Trivial field-by-field copy.
    pub(crate) fn to_ffi(&self) -> llama_harness_ffi::ffi::SamplingParams {
        llama_harness_ffi::ffi::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
            max_tokens: self.max_tokens,
            stop: self.stop.iter().cloned().collect(),
        }
    }
}
