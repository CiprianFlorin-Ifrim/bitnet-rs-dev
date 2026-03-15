//! Safe Rust bindings to Microsoft's BitNet b1.58 inference engine.
//!
//! # Usage
//!
//! ```no_run
//! use bitnet::{init, suppress_warnings, Model, ModelParams, ContextParams, GenerateParams};
//!
//! init();
//! suppress_warnings();
//!
//! let model = Model::load(
//!     "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
//!     ModelParams::default(),
//! )?;
//!
//! let mut session = model.session(ContextParams::default())?;
//!
//! // Full response as a String.
//! let response = session.generate("The capital of France is", &GenerateParams::default())?;
//! println!("{response}");
//!
//! // Streaming — callback is invoked for each token piece as it is produced.
//! session.generate_streaming("The capital of France is", &GenerateParams::default(), |piece| {
//!     print!("{piece}");
//! })?;
//!
//! # Ok::<(), bitnet::Error>(())
//! ```
//!
//! # Model setup
//!
//! The GGUF file must be produced from the BF16 weights using the conversion
//! helper in the bitnet.cpp repository. The pre-packaged GGUF from Hugging Face
//! is missing pre-tokenizer metadata and produces incoherent output.
//!
//! ```sh
//! hf download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir models/bitnet-b1.58-2B-4T-bf16
//! python utils/convert-helper-bitnet.py models/bitnet-b1.58-2B-4T-bf16
//! ```
//!
//! # CPU only
//!
//! BitNet uses lookup-table kernels for its ternary weight format that only
//! run on the CPU. Offloading layers to a GPU backend produces incorrect output
//! and is not supported by this crate.
//!
//! # Chat template
//!
//! This model uses the Llama 3 chat template. When building conversational
//! prompts, structure them using the header tokens the model was trained with.
//! See the inference example for a full implementation.

mod error;
mod model;
mod params;
mod session;

pub use error::Error;
pub use model::Model;
pub use params::{ContextParams, GenerateParams, ModelParams, SamplingStrategy};
pub use session::Session;

/// Initialises the ggml backend. Must be called once before the first
/// Model::load call. Safe to call from main or behind a std::sync::Once guard.
pub fn init() {
    unsafe { bitnet_sys::llama_backend_init() };
}

/// Releases resources allocated by init. In most applications the OS will
/// handle this on exit, but calling it explicitly is good practice in
/// long-running processes that load and unload models at runtime.
pub fn deinit() {
    unsafe { bitnet_sys::llama_backend_free() };
}

/// Suppresses warning-level output from the underlying library, including
/// the pre-tokenizer and control token warnings, while keeping informational
/// messages such as memory usage and context size visible.
pub fn suppress_warnings() {
    unsafe extern "C" fn filter_log(
        level: std::ffi::c_int,
        text: *const std::ffi::c_char,
        _user_data: *mut std::ffi::c_void,
    ) {
        // llama.cpp log levels: 1=debug, 2=info, 3=warn, 4=error.
        // Info and errors are kept. Warnings (tokenizer noise) and debug output are suppressed.
        if level == 1 {
            let s = unsafe { std::ffi::CStr::from_ptr(text) };
            if let Ok(s) = s.to_str() {
                eprint!("{s}");
            }
        }
    }
    unsafe { bitnet_sys::llama_log_set(Some(filter_log), std::ptr::null_mut()) };
}