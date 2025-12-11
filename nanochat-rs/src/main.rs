use anyhow::Result;
use clap::Parser;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::fs;

mod config;
mod model;
mod diffusion;

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
        n_layer: 2, // Small for testing
        n_head: 2,
        n_kv_head: 2,
        n_embd: 64,
        vocab_size: 100,
        sequence_len: 32,
        use_vision: false,
        vision_image_size: 224,
        vision_patch_size: 14,
        vision_width: 768,
        vision_layers: 12,
        vision_heads: 12,
        vision_mlp_ratio: 4.0,
        use_robotics: true, // Enable robotics for testing action head
        robotics_sensor_dim: 16,
        robotics_surface_dim: 32,
        robotics_sensor_tokens: 1,
        robotics_surface_tokens: 4,
        robotics_use_diffusion: true, // Test Diffusion!
        robotics_diffusion_steps: 5, // Small steps for fast test
    };

    println!("Loading model from {}...", args.model_path);
    let device = Device::Cpu;

    // 2. Load Weights
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

    // Dummy Sensors (B, dim)
    let dummy_sensors = Tensor::randn(0.0f32, 1.0f32, (1, 16), &device)?;

    println!("Running forward pass with Diffusion sampling...");
    let (logits, action) = model.forward(&dummy_input, Some(&dummy_sensors), None)?;

    println!("Logits shape: {:?}", logits.shape());
    if let Some(act) = action {
        println!("Action predicted (Diffusion)! Shape: {:?}", act.shape());
        println!("Sample action vector (first 5 elements): {:?}", act.flatten_all()?.to_vec1::<f32>()?[0..5].to_vec());
    } else {
        println!("No action predicted.");
    }

    Ok(())
}
