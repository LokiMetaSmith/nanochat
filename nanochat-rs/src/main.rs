use anyhow::Result;
use clap::Parser;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use std::fs;

mod config;
mod model;
mod diffusion;
mod tiny_math;
mod tiny_infovore;

use config::Config;
use model::GPT;
use tiny_infovore::TinyInfovore;
use tiny_math::{Vector, Lcg};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model checkpoint (safetensors or folder)
    #[arg(long, default_value = "model.safetensors")]
    model_path: String,

    /// Prompt to test
    #[arg(long, default_value = "Hello world")]
    prompt: String,

    /// Run the Tiny Infovore simulation demo (learning loop)
    #[arg(long)]
    sim_infovore: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.sim_infovore {
        run_infovore_demo();
        return Ok(());
    }

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

fn run_infovore_demo() {
    println!("=== TinyInfovore Learning Demo ===");
    println!("Simulating a dynamic system: Target = Rotated Input (Rotation Matrix)");

    // Config
    let dim = 4;
    let steps = 200;
    let mut learner = TinyInfovore::new(dim, dim, 0.1, 0.9, 42);
    let mut rng = Lcg::new(123);

    // Define a "Physics" system: A slow rotation + noise
    // We want to see if the learner can predict state_{t+1} from state_t
    // For simplicity, let's just say state_{t+1} = state_t * 0.5 + const (Linear decay dynamics)
    // Actually, let's do a simple mapping: x' = M * x
    // M is an identity matrix scaled by 0.9 (damping)
    let damping = 0.9f32;

    println!("System Dynamics: x(t+1) = {:.1} * x(t) + noise", damping);

    let mut current_state = Vector::random(dim, &mut rng);

    for step in 0..steps {
        // 1. Simulate "Real World" Next Step
        // next = current * damping + noise
        let deterministic_next = current_state.scale(damping);
        let noise = Vector::random(dim, &mut rng).scale(0.1); // Small noise
        let actual_next_state = deterministic_next.add(&noise);

        // 2. Learner Predicts and Updates
        let (loss, relation) = learner.update(&current_state, &actual_next_state);

        // 3. Log
        if step % 20 == 0 {
            println!("Step {:03} | Loss: {:.6} | Relation: {:.4}", step, loss, relation);
        }

        // 4. Trust Signal Check
        // If loss is low and relation is high, we could theoretically skip the next "heavy" computation
        if loss < 0.01 && relation > 0.9 {
             // System is stable and familiar
        }

        // Advance Time
        current_state = actual_next_state;
    }

    println!("=== Demo Complete ===");
    println!("Final Weights (Should be close to diag({:.1})):", damping);
    // Print first row of weights to check if diagonal is ~0.9 and rest ~0
    // But rows/cols mapping depends on implementation.
    // Row 0 should be [0.9, 0, 0, 0] roughly.
    let row0 = &learner.weights.data[0..dim];
    print!("Row 0: [");
    for v in row0 { print!("{:.3} ", v); }
    println!("]");

    // Save/Load Test
    let save_path = "infovore_demo.json";
    println!("Saving model to {}...", save_path);
    if let Err(e) = learner.save_to_json(save_path) {
        eprintln!("Failed to save model: {}", e);
    } else {
        println!("Model saved successfully.");
        match TinyInfovore::load_from_json(save_path) {
            Ok(loaded_learner) => {
                println!("Model loaded back successfully.");
                // Simple verify
                if learner.weights.data == loaded_learner.weights.data {
                     println!("Verification Success: Loaded weights match saved weights.");
                } else {
                     eprintln!("Verification Failed: Weights do not match!");
                }
            },
            Err(e) => eprintln!("Failed to load model back: {}", e),
        }
    }
}
