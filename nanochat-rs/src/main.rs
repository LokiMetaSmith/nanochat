use anyhow::Result;
use clap::Parser;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::fs;

mod config;
mod model;

use config::Config;
use model::GPT;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model checkpoint (safetensors or folder)
    #[arg(long)]
    model_path: String,

    /// Prompt to test
    #[arg(long, default_value = "Hello world")]
    prompt: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Load Config (assume config.json exists or hardcoded for now)
    // In real app, read from model_path/config.json or parse from GGUF metadata
    let config = Config {
        n_layer: 12,
        n_head: 12,
        n_kv_head: 12,
        n_embd: 768,
        vocab_size: 50304,
        sequence_len: 1024,
        use_vision: false,
        vision_image_size: 224,
        vision_patch_size: 14,
        vision_width: 768,
        vision_layers: 12,
        vision_heads: 12,
        vision_mlp_ratio: 4.0,
        use_robotics: false,
        robotics_sensor_dim: 0,
        robotics_surface_dim: 0,
        robotics_sensor_tokens: 1,
        robotics_surface_tokens: 4,
        robotics_use_diffusion: false,
        robotics_diffusion_steps: 100,
    };

    println!("Loading model from {}...", args.model_path);
    let device = Device::Cpu; // or Cuda(0)

    // 2. Load Weights
    // If folder, look for model.safetensors
    let safetensors_path = if std::path::Path::new(&args.model_path).is_dir() {
        std::path::Path::new(&args.model_path).join("model.safetensors")
    } else {
        std::path::Path::new(&args.model_path).to_path_buf()
    };

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device)? };

    // 3. Build Model
    let model = GPT::new(&config, vb)?;
    println!("Model loaded.");

    // 4. Dummy Inference
    let dummy_input = Tensor::zeros((1, 10), DType::U32, &device)?; // Batch 1, Seq 10
    let logits = model.forward(&dummy_input)?;

    println!("Forward pass successful. Logits shape: {:?}", logits.shape());

    Ok(())
}
