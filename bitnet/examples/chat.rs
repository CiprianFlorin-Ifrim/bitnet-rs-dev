// Interactive chat example for BitNet b1.58.
//
// Maintains a conversation history and formats each turn using the Llama 3
// chat template that this model was trained with.
//
// Usage:
//   cargo run --release --example inference -- <path-to-gguf>
//   cargo run --release --example inference -- <path-to-gguf> "optional system prompt"

use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use bitnet::{ContextParams, GenerateParams, Model, ModelParams, SamplingStrategy};

fn main() -> Result<(), bitnet::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: inference <path-to-gguf> [system-prompt]");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --release --example inference -- \\");
        eprintln!("    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \\");
        eprintln!("    \"You are a helpful assistant.\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let system_prompt = args.get(2).map(|s| s.as_str()).unwrap_or("You are a helpful assistant.");

    bitnet::init();
    bitnet::suppress_warnings();

    println!("Loading model from {model_path}...");
    let load_start = Instant::now();
    let model = Model::load(model_path, ModelParams::default())?;
    println!("Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f32());
    println!("System: {system_prompt}");
    println!("Type your message and press Enter. Type 'quit' to exit.\n");

    let params = GenerateParams {
        max_tokens: 512,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: u32::MAX,
        },
    };

    // Conversation history stored as alternating user/assistant turn strings.
    // The full history is re-encoded on every turn so the model has context.
    let mut history: Vec<(String, String)> = Vec::new();

    let stdin = io::stdin();
    let mut session = model.session(ContextParams {
        n_ctx: 0,
        n_batch: 32,
        n_threads: 10,
    })?;

    loop {
        print!("You: ");
        let _ = io::stdout().flush();

        let mut user_input = String::new();
        if stdin.lock().read_line(&mut user_input).is_err() {
            break;
        }
        let user_input = user_input.trim().to_string();

        if user_input.is_empty() {
            continue;
        }
        if user_input.eq_ignore_ascii_case("quit") || user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Build the full prompt from history plus the new user turn using the
        // Llama 3 chat template this model was trained with.
        let prompt = build_prompt(system_prompt, &history, &user_input);

        print!("Assistant: ");
        let _ = io::stdout().flush();

        let mut assistant_response = String::new();
        let gen_start = Instant::now();
        let mut token_count = 0usize;

        session.generate_streaming(&prompt, &params, |piece| {
            print!("{piece}");
            let _ = io::stdout().flush();
            assistant_response.push_str(piece);
            token_count += 1;
        })?;

        let elapsed = gen_start.elapsed().as_secs_f32();
        println!("\n[{token_count} tokens, {elapsed:.2}s, {:.1} tok/s]\n",
            token_count as f32 / elapsed);

        // Store the completed turn so the next prompt includes it as context.
        history.push((user_input, assistant_response));
    }

    bitnet::deinit();
    Ok(())
}

/// Formats the full conversation as a Llama 3 chat template prompt.
///
/// The model was trained with this specific structure. Deviating from it
/// produces incoherent output because the model expects these control tokens
/// to understand turn boundaries.
fn build_prompt(system: &str, history: &[(String, String)], user_input: &str) -> String {
    let mut prompt = String::new();

    // System turn
    prompt.push_str("<|start_header_id|>system<|end_header_id|>\n");
    prompt.push_str(system);
    prompt.push_str("<|eot_id|>");

    // Previous conversation turns
    for (user, assistant) in history {
        prompt.push_str("<|start_header_id|>user<|end_header_id|>\n");
        prompt.push_str(user);
        prompt.push_str("<|eot_id|>");
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n");
        prompt.push_str(assistant);
        prompt.push_str("<|eot_id|>");
    }

    // Current user turn — no closing eot_id so the model generates the response
    prompt.push_str("<|start_header_id|>user<|end_header_id|>\n");
    prompt.push_str(user_input);
    prompt.push_str("<|eot_id|>");
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n");

    prompt
}