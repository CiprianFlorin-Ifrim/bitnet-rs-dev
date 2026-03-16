# bitnet-llm-rs-dev

Development repository for [bitnet-llm](https://crates.io/crates/bitnet-llm) — safe Rust bindings to Microsoft's [BitNet b1.58](https://github.com/microsoft/BitNet) inference engine (`bitnet.cpp`).

This repo uses a git submodule for bitnet.cpp, making it easy to track upstream changes. For production use, add `bitnet-llm` directly from crates.io instead.

---

## Workspace layout
```
bitnet-llm-rs-dev/
├── Cargo.toml                workspace root
├── bitnet-llm-sys/           raw FFI + CMake build of bitnet.cpp
│   ├── build.rs
│   ├── bitnet.cpp/           git submodule
│   └── src/lib.rs
└── bitnet-llm/               safe, ergonomic Rust API
    └── src/
        ├── lib.rs
        ├── error.rs
        ├── params.rs
        ├── model.rs
        └── session.rs
```

---

## Prerequisites

| Tool | Notes |
|------|-------|
| Rust >= 1.73 | `rustup update stable` |
| CMake >= 3.14 | `brew install cmake` / `apt install cmake` |
| Clang | Required by bitnet.cpp; usually already present |
| Python 3 + pip | For model conversion only, not at Rust build time |
| `hf` CLI | `pip install huggingface_hub` |

**macOS (Apple Silicon) only:** Xcode Command Line Tools (`xcode-select --install`).

---

## Step 1 — Clone with submodule
```sh
git clone --recurse-submodules https://github.com/CiprianFlorin-Ifrim/bitnet-llm-rs-dev
```

Or if already cloned:
```sh
git submodule update --init --recursive
```

Two generated header files are required that are not committed to the BitNet
repository. Copy them from an existing BitNet installation:
```sh
cp /path/to/BitNet/include/bitnet-lut-kernels.h bitnet-llm-sys/bitnet.cpp/include/
cp /path/to/BitNet/include/ggml-bitnet.h        bitnet-llm-sys/bitnet.cpp/3rdparty/llama.cpp/ggml/include/
```

---

## Step 2 — Download the model

### Option 1 — Pre-packaged GGUF (simplest)
```sh
hf download microsoft/BitNet-b1.58-2B-4T-gguf \
    --local-dir models/BitNet-b1.58-2B-4T
```

The model file is `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`.

### Option 2 — Convert from BF16 weights (a must for ARM systems)
```sh
pip install huggingface_hub numpy torch

hf download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir models/bitnet-b1.58-2B-4T-bf16

pip install -r bitnet-llm-sys/bitnet.cpp/requirements.txt

python3 bitnet-llm-sys/bitnet.cpp/utils/convert-helper-bitnet.py \
    models/bitnet-b1.58-2B-4T-bf16
```

The output file is `models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf`.

---

## Step 3 — Build

### macOS (Apple Silicon)
```sh
cargo build --release
```

### Linux (x86-64)
```sh
cargo build --release
```

---

## Performance

Benchmarked on [Hetzner Cloud](https://www.hetzner.com/cloud/) using the
`BitNet-b1.58-2B-4T` model with `max_tokens = 512`, `TopP` sampling.

| Instance | CPU | Arch | Kernel | TTFT | Short tok/s | Long tok/s |
|----------|-----|------|--------|------|-------------|------------|
| CX23 | 2 vCPU Intel/AMD | x86-64 (AVX-512) | TL2 | ~2.5s | 2.3–2.5 | 7.4 |
| CAX11 | 2 vCPU Ampere | ARM64 | TL1 | ~2.6s | 2.4 | 7.5 |

TTFT includes prompt prefill. Short tok/s is averaged over responses under 30
tokens where prefill overhead dominates; long tok/s is averaged over responses
of 100+ tokens where generation dominates.

---

## Usage

### Running the examples
```sh
# Single prompt streaming inference
cargo run --release --example inference_streaming -- \
    models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    "What is the capital of France?"

# Single prompt, returns full response at once
cargo run --release --example inference_standard -- \
    models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    "What is the capital of France?"

# Interactive multi-turn chat
cargo run --release --example chat -- \
    models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    "You are a helpful assistant."
```

### Library usage
```rust
use bitnet_llm::{
    init, suppress_warnings, ContextParams, GenerateParams,
    Model, ModelParams, SamplingStrategy,
};
use std::io::Write;

fn main() -> Result<(), bitnet_llm::Error> {
    init();
    suppress_warnings();

    let model = Model::load(
        "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
        ModelParams::default(),
    )?;

    let mut session = model.session(ContextParams::default())?;
    let params = GenerateParams::default();

    // Turn 1
    session.generate_streaming(
        "<|start_header_id|>system<|end_header_id|>\n\
         You are a helpful assistant.<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\
         Hello!<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
        |piece| { print!("{piece}"); let _ = std::io::stdout().flush(); },
    )?;
    session.encode("<|eot_id|>")?;

    // Turn 2 — only new tokens encoded, history stays in KV cache
    session.generate_streaming(
        "<|start_header_id|>user<|end_header_id|>\n\
         How are you?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
        |piece| { print!("{piece}"); let _ = std::io::stdout().flush(); },
    )?;
    session.encode("<|eot_id|>")?;

    session.reset();
    bitnet_llm::deinit();
    Ok(())
}
```

---

## API overview

| Item | Description |
|------|-------------|
| `init()` | Initialise the backend. Call once before `Model::load`. |
| `deinit()` | Free backend resources. Optional at process exit. |
| `suppress_warnings()` | Suppress tokenizer noise while keeping memory and context info. |
| `Model::load(path, ModelParams)` | Load a GGUF model from disk. |
| `model.session(ContextParams)` | Create an inference context. |
| `session.generate(prompt, &params)` | Run inference, return `String`. |
| `session.generate_streaming(prompt, &params, callback)` | Run inference, call callback per token piece. |
| `session.encode(text)` | Feed text into the KV cache without generating output. |
| `session.reset()` | Clear the KV cache and start a fresh conversation. |
| `session.kv_pos()` | Current token position in the KV cache. |
| `session.tokens_remaining()` | Tokens remaining before context window is full. |

### `ModelParams`

| Field | Default | Description |
|-------|---------|-------------|
| `n_gpu_layers` | `0` | Must be 0. BitNet kernels are CPU-only. |
| `use_mmap` | `true` | Memory-mapped weight loading. |
| `use_mlock` | `false` | Lock weights in RAM. |

### `ContextParams`

| Field | Default | Description |
|-------|---------|-------------|
| `n_ctx` | `0` | Context window size. 0 uses the model maximum (4096). |
| `n_batch` | `32` | Internal batch buffer size. |
| `n_threads` | auto | CPU threads for inference. |

### `GenerateParams`

| Field | Default | Description |
|-------|---------|-------------|
| `max_tokens` | `512` | Maximum tokens to generate. |
| `sampling` | `TopP` | `Greedy` or `TopP { temperature, top_p, seed }`. |

---

## Important notes

**CPU only.** BitNet's ternary lookup-table kernels only run on the CPU. The
ARM TL1 kernel is used on Apple Silicon and the x86 TL2 kernel on Linux. GPU
offloading is not supported.

**Single-token processing.** The BitNet kernels produce valid output only when
processing one token at a time. This library handles this correctly internally.

**Chat template.** This model uses the Llama 3 chat template with
`<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` control tokens.

**Context window.** The model has a 4096 token context window. The library
warns at 80% usage and returns an error when full. Call `session.reset()` to
start a fresh conversation.

**Multi-turn correctness.** Always call `session.encode("<|eot_id|>")` after
each assistant turn. Always call `session.reset()` between separate
conversations. Do not call `session.encode()` before the first
`generate_streaming` call on a fresh session as this will prevent BOS from
being added.

**Updating bitnet.cpp.** To pull in upstream BitNet changes, run:
```sh
git submodule update --remote bitnet-llm-sys/bitnet.cpp
git add bitnet-llm-sys/bitnet.cpp
git commit -m "Update bitnet.cpp submodule"
```

**Generation.** The pre-packaged GGUF (`Option 1`) produces word-salad output on ARM.
Use `Option 2` (convert from BF16) on ARM Linux.
