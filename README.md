# bitnet-rs

Safe Rust bindings to Microsoft's [BitNet b1.58](https://github.com/microsoft/BitNet)
inference engine (`bitnet.cpp`).

The crate provides a clean, ergonomic API with both full-response and
streaming-callback inference, targeting macOS (Apple Silicon) and Linux
(x86-64).

---

## Workspace layout
```
bitnet-rs/
├── Cargo.toml           workspace root
├── bitnet-sys/          raw FFI + CMake build of bitnet.cpp
│   ├── build.rs
│   ├── bitnet.cpp/      git submodule (see step 1 below)
│   └── src/lib.rs
└── bitnet/              safe, ergonomic Rust API
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

## Step 1 — Add the bitnet.cpp submodule

The bitnet.cpp C++ source tree must live at `bitnet-sys/bitnet.cpp`.
```sh
git submodule add https://github.com/microsoft/BitNet bitnet-sys/bitnet.cpp
git submodule update --init --recursive
```

> If you cloned this repo without `--recurse-submodules`, run
> `git submodule update --init --recursive` from the repository root.

Two generated header files are required that are not committed to the BitNet
repository. They are produced by running `setup_env.py` in an existing BitNet
installation and must be copied into the submodule before building:
```sh
# Copy the generated headers from your BitNet installation
cp /path/to/BitNet/include/bitnet-lut-kernels.h bitnet-sys/bitnet.cpp/include/
cp /path/to/BitNet/include/ggml-bitnet.h        bitnet-sys/bitnet.cpp/3rdparty/llama.cpp/ggml/include/
```

If you do not have an existing BitNet installation, run `setup_env.py` from the
submodule itself first to generate them:
```sh
cd bitnet-sys/bitnet.cpp
pip install -r requirements.txt
python setup_env.py -md /tmp/placeholder -q i2_s || true
cd ../..
```

---

## Step 2 - Download and convert the model

The pre-packaged GGUF on Hugging Face is missing pre-tokenizer metadata and
produces incoherent output. You must convert from the BF16 master weights.
```sh
# Create a conda environment (recommended)
conda create -n bitnet python=3.11
conda activate bitnet

# Install conversion dependencies
pip install -r bitnet-sys/bitnet.cpp/requirements.txt

# Download the BF16 master weights (~5 GB)
hf download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir models/bitnet-b1.58-2B-4T-bf16

# Convert to GGUF format
python bitnet-sys/bitnet.cpp/utils/convert-helper-bitnet.py \
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

The build script auto-detects AVX-512 and enables the `TL2` BitNet kernel if
available, falling back to the AVX2 `TL1` kernel automatically.

---

## Usage

### Interactive chat
```sh
cargo run --release --example inference -- \
    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
    "You are a helpful assistant."
```

### Library usage
```rust
use bitnet::{
    init, suppress_warnings, ContextParams, GenerateParams,
    Model, ModelParams, SamplingStrategy,
};
use std::io::Write;

fn main() -> Result<(), bitnet::Error> {
    init();
    suppress_warnings();

    let model = Model::load(
        "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
        ModelParams::default(),
    )?;

    let mut session = model.session(ContextParams::default())?;

    // Build a prompt using the Llama 3 chat template
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\
                  What is the capital of France?<|eot_id|>\
                  <|start_header_id|>assistant<|end_header_id|>\n";

    // Full response at once
    let response = session.generate(prompt, &GenerateParams::default())?;
    println!("{response}");

    // Streaming — callback fires for each token piece as it is produced
    session.generate_streaming(prompt, &GenerateParams::default(), |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;

    // Custom sampling
    let params = GenerateParams {
        max_tokens: 256,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: 42,
        },
    };
    let response = session.generate(prompt, &params)?;
    println!("{response}");

    bitnet::deinit();
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
ARM TL1 kernel is used on Apple Silicon and the x86 TL2 kernel on Linux. GPU is
not supported.

**Single-token decode loop.** The BitNet kernels produce valid output only when
decoding one token at a time. This library handles this correctly internally.
Feeding multi-token batches to `llama_decode` directly will produce garbage.

**Chat template.** This model uses the Llama 3 chat template with
`<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` control tokens.
See the inference example for a complete multi-turn chat implementation.

**Build time.** The first build compiles bitnet.cpp from source via CMake,
which takes a few minutes. Subsequent builds use the cached output.

**Multi-turn chat.** Each `generate` call clears the KV cache. For
conversations, accumulate the full history in your application and pass it as
the prompt on each turn. The inference example demonstrates this pattern.