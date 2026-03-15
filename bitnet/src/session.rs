use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bitnet_sys as sys;

use crate::error::Error;
use crate::model::ModelInner;
use crate::params::{ContextParams, GenerateParams, SamplingStrategy};

struct SessionInner {
    ctx: NonNull<sys::llama_context>,
    // Holds a reference to the model so the llama_model pointer remains valid
    // for the lifetime of this context.
    _model: Arc<ModelInner>,
}

// llama_context must not be aliased across threads, but it is safe to move
// to another thread. Our ownership model ensures no aliasing is possible.
unsafe impl Send for SessionInner {}

impl Drop for SessionInner {
    fn drop(&mut self) {
        unsafe { sys::llama_free(self.ctx.as_ptr()) };
    }
}

/// An inference context tied to a loaded model.
///
/// Each session wraps a single llama_context with its own KV cache. Sessions
/// are not Clone. To run inference concurrently, create one session per thread
/// from the same Model.
///
/// The KV cache is cleared at the start of each generate call, so calls are
/// independent by default. For multi-turn conversations, accumulate the full
/// conversation history in your application and pass it as the prompt on each
/// turn.
///
/// # Example
///
/// ```no_run
/// use bitnet::{Model, ModelParams, ContextParams, GenerateParams};
///
/// let model = Model::load("model.gguf", ModelParams::default())?;
/// let mut session = model.session(ContextParams::default())?;
///
/// let response = session.generate("Hello!", &GenerateParams::default())?;
/// println!("{response}");
/// # Ok::<(), bitnet::Error>(())
/// ```
pub struct Session {
    inner: SessionInner,
}

impl Session {
    pub(crate) fn new(model: Arc<ModelInner>, params: ContextParams) -> Result<Self, Error> {
        let threads = params.n_threads.max(1) as i32;

        let mut c_params = unsafe { sys::llama_context_default_params() };
        c_params.n_ctx = params.n_ctx;
        c_params.n_batch = params.n_batch;
        c_params.n_threads = threads;
        c_params.n_threads_batch = threads;

        let raw = unsafe {
            sys::llama_new_context_with_model(model.ptr.as_ptr(), c_params)
        };

        let ctx = NonNull::new(raw).ok_or_else(|| {
            Error::ContextCreate("llama_new_context_with_model returned null".into())
        })?;

        Ok(Self {
            inner: SessionInner { ctx, _model: model },
        })
    }

    /// Runs inference on the prompt and returns the complete generated text.
    ///
    /// This is a convenience wrapper around generate_streaming that collects
    /// all token pieces into a single String before returning.
    pub fn generate(&mut self, prompt: &str, params: &GenerateParams) -> Result<String, Error> {
        let mut output = String::new();
        self.generate_streaming(prompt, params, |piece| output.push_str(piece))?;
        Ok(output)
    }

    /// Runs inference on the prompt, invoking the callback for each decoded
    /// token piece as it is produced.
    ///
    /// The callback receives a string slice which may be a single character,
    /// a partial word, or a full word depending on the tokeniser. It is called
    /// synchronously on the calling thread, so it can write to stdout, push to
    /// a channel, or accumulate into a buffer without additional synchronisation.
    ///
    /// Generation stops when the model emits an end-of-sequence token or when
    /// max_tokens tokens have been produced.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenerateParams,
        mut on_token: impl FnMut(&str),
    ) -> Result<(), Error> {
        let ctx = self.inner.ctx.as_ptr();
        let model = self.inner._model.ptr.as_ptr();

        unsafe { sys::llama_kv_cache_clear(ctx) };

        let tokens = tokenise(model, prompt, true)?;

        let n_ctx = unsafe { sys::llama_n_ctx(ctx) } as usize;
        if tokens.len() >= n_ctx {
            return Err(Error::Tokenise(format!(
                "prompt is {} tokens but context window is {n_ctx}",
                tokens.len()
            )));
        }

        let sampler = build_sampler(&params.sampling)?;

        // Feed each prompt token individually. The BitNet ARM TL1 and x86 TL2
        // lookup-table kernels only produce valid logits for single-token
        // batches. Batching multiple tokens causes NaN output regardless of
        // the n_batch setting. This matches the -b 1 flag used by run_inference.py.
        for (i, &tok) in tokens.iter().enumerate() {
            let mut single = [tok];
            let batch = unsafe {
                sys::llama_batch_get_one(single.as_mut_ptr(), 1, i as i32, 0)
            };
            match unsafe { sys::llama_decode(ctx, batch) } {
                0 => {}
                1 => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::KvCacheFull);
                }
                n => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::Decode(n));
                }
            }
        }

        let mut n_generated = 0usize;
        // Reuse a single-element array for every autoregressive step to avoid
        // repeated stack allocation in the hot loop.
        let mut next_token = [0i32; 1];

        loop {
            if n_generated >= params.max_tokens {
                break;
            }

            // Logits are always at output row 0 since every decode processes
            // exactly one token.
            let token = unsafe { sys::llama_sampler_sample(sampler, ctx, 0) };

            // Inform the sampler so stateful stages such as repetition penalty
            // track the accepted token correctly.
            unsafe { sys::llama_sampler_accept(sampler, token) };

            if unsafe { sys::llama_token_is_eog(model, token) } {
                break;
            }

            let piece = token_to_text(model, token)?;
            on_token(&piece);
            n_generated += 1;

            // Feed the new token back at the next position in the sequence.
            next_token[0] = token;
            let pos = (tokens.len() + n_generated - 1) as i32;
            let batch = unsafe {
                sys::llama_batch_get_one(next_token.as_mut_ptr(), 1, pos, 0)
            };
            match unsafe { sys::llama_decode(ctx, batch) } {
                0 => {}
                1 => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::KvCacheFull);
                }
                n => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::Decode(n));
                }
            }
        }

        unsafe { sys::llama_sampler_free(sampler) };

        Ok(())
    }
}

fn tokenise(
    model: *const sys::llama_model,
    text: &str,
    add_special: bool,
) -> Result<Vec<sys::llama_token>, Error> {
    let c_text = CString::new(text)
        .map_err(|_| Error::Tokenise("prompt contains a null byte".into()))?;

    // A negative return value indicates the buffer was too small and its
    // absolute value is the required capacity. We use this to allocate exactly
    // the right buffer on the second call instead of over-allocating.
    let n_required = unsafe {
        sys::llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            std::ptr::null_mut(),
            0,
            add_special,
            true,
        )
    };

    if n_required == 0 {
        return Err(Error::Tokenise("prompt produced zero tokens".into()));
    }

    let capacity = n_required.unsigned_abs() as usize;
    let mut buf: Vec<sys::llama_token> = vec![0; capacity];

    let n_tokens = unsafe {
        sys::llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            buf.as_mut_ptr(),
            capacity as i32,
            add_special,
            true,
        )
    };

    if n_tokens < 0 {
        return Err(Error::Tokenise(format!(
            "tokenisation failed with code {n_tokens}"
        )));
    }

    buf.truncate(n_tokens as usize);
    Ok(buf)
}

fn token_to_text(
    model: *const sys::llama_model,
    token: sys::llama_token,
) -> Result<String, Error> {
    // Start with a 32-byte buffer which covers the vast majority of tokens
    // without needing a retry.
    let mut buf = vec![0u8; 32];
    loop {
        let n = unsafe {
            sys::llama_token_to_piece(
                model,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                false,
            )
        };
        if n < 0 {
            // Buffer was too small. The absolute value is the required size.
            buf.resize((-n) as usize, 0);
        } else {
            buf.truncate(n as usize);
            return String::from_utf8(buf).map_err(|e| {
                Error::Internal(format!("token {token} produced invalid UTF-8: {e}"))
            });
        }
    }
}

fn build_sampler(strategy: &SamplingStrategy) -> Result<*mut sys::llama_sampler, Error> {
    let chain_params = unsafe { sys::llama_sampler_chain_default_params() };
    let chain = unsafe { sys::llama_sampler_chain_init(chain_params) };

    if chain.is_null() {
        return Err(Error::Internal("failed to create sampler chain".into()));
    }

    match strategy {
        SamplingStrategy::Greedy => {
            let stage = unsafe { sys::llama_sampler_init_greedy() };
            unsafe { sys::llama_sampler_chain_add(chain, stage) };
        }
        SamplingStrategy::TopP { temperature, top_p, seed } => {
            // Repetition penalty prevents the model from looping on the same
            // token or phrase, which is common without it on completion prompts.
            let penalty_stage = unsafe {
                sys::llama_sampler_init_penalties(64, 1.1, 0.0, 0.0)
            };
            unsafe { sys::llama_sampler_chain_add(chain, penalty_stage) };
            // Sampler stage order matches the chain used by run_inference.py:
            // top-k -> top-p -> min-p -> temperature -> distribution draw.
            let top_k_stage = unsafe { sys::llama_sampler_init_top_k(40) };
            unsafe { sys::llama_sampler_chain_add(chain, top_k_stage) };

            if *top_p < 1.0 {
                let top_p_stage = unsafe { sys::llama_sampler_init_top_p(*top_p, 1) };
                unsafe { sys::llama_sampler_chain_add(chain, top_p_stage) };
            }

            let min_p_stage = unsafe { sys::llama_sampler_init_min_p(0.05, 1) };
            unsafe { sys::llama_sampler_chain_add(chain, min_p_stage) };

            let temp_stage = unsafe { sys::llama_sampler_init_temp(*temperature) };
            unsafe { sys::llama_sampler_chain_add(chain, temp_stage) };

            // Derive a seed from the system clock when the caller has not
            // specified one, so repeated calls produce different outputs.
            let effective_seed = if *seed == u32::MAX {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(42)
            } else {
                *seed
            };

            let dist_stage = unsafe { sys::llama_sampler_init_dist(effective_seed) };
            unsafe { sys::llama_sampler_chain_add(chain, dist_stage) };
        }
    }

    Ok(chain)
}