use crate::tiny_math::{Vector, Matrix, Lcg};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use anyhow::Result;

#[derive(Serialize, Deserialize)]
pub struct TinyInfovore {
    pub weights: Matrix,
    pub bias: Vector,
    pub manifold_centroid: Vector,
    pub momentum: f32, // For manifold EMA
    pub learning_rate: f32,
    pub initialized: bool,
}

impl TinyInfovore {
    pub fn new(input_dim: usize, output_dim: usize, learning_rate: f32, momentum: f32, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        Self {
            weights: Matrix::random(output_dim, input_dim, &mut rng),
            bias: Vector::zeros(output_dim), // Initialize bias to zeros
            manifold_centroid: Vector::zeros(input_dim), // Manifold tracks Input state
            momentum,
            learning_rate,
            initialized: false,
        }
    }

    // Predict next state based on current state
    pub fn predict(&self, state: &Vector) -> Vector {
        let linear = self.weights.matmul(state);
        linear.add(&self.bias)
    }

    // Update weights and manifold based on actual observation
    // Returns (Loss, Relation Score, Updated Prediction)
    pub fn update(&mut self, current_state: &Vector, actual_next_state: &Vector) -> (f32, f32) {
        // 1. Prediction (Forward)
        let predicted_next = self.predict(current_state);

        // 2. Compute Novelty (Surprisal) = MSE Loss
        let loss = predicted_next.mse(actual_next_state);

        // 3. Compute Relation = Cosine Similarity(current_state, manifold)
        // Note: Relation is about how "familiar" the input context is.
        let relation = if self.initialized {
            let sim = current_state.cosine_similarity(&self.manifold_centroid);
            // Clamp negative relation to 0 (we don't learn from opposites in this version, or maybe we do?)
            // Infovore.py uses F.relu, so we clamp.
            if sim < 0.0 { 0.0 } else { sim }
        } else {
            1.0 // Trust first sample fully to initialize
        };

        // 4. Update Manifold (EMA)
        if !self.initialized {
            self.manifold_centroid = current_state.clone();
            self.initialized = true;
        } else {
            // centroid = momentum * centroid + (1 - momentum) * current
            let old_part = self.manifold_centroid.scale(self.momentum);
            let new_part = current_state.scale(1.0 - self.momentum);
            self.manifold_centroid = old_part.add(&new_part);
        }

        // 5. Update Weights (Backward)
        // Loss = MSE = 1/N * sum((pred - actual)^2)
        // Gradient w.r.t Output = 2/N * (pred - actual)
        // Gradient w.r.t Weights = Gradient_Out * Input^T
        // Gradient w.r.t Bias = Gradient_Out

        // Effective Learning Rate = base_lr * relation
        // We want to learn MORE from "Related" (high relation) and "Novel" (high loss) events.
        // But if Relation is 0 (completely alien context), we don't update weights to avoid catastrophic forgetting.
        let effective_lr = self.learning_rate * relation;

        if effective_lr > 1e-7 {
            let n = actual_next_state.len() as f32;
            let grad_scale = 2.0 / n;

            // grad_out = (pred - actual) * grad_scale
            let diff = predicted_next.sub(actual_next_state);
            let grad_out = diff.scale(grad_scale);

            // Update Bias: b = b - lr * grad_out
            let bias_step = grad_out.scale(effective_lr);
            self.bias = self.bias.sub(&bias_step);

            // Update Weights: W = W - lr * (grad_out * input^T)
            // Weight Gradient is Outer Product of Grad_Out and Input
            let weight_grad = Matrix::outer(&grad_out, current_state);

            // Apply update: W -= lr * weight_grad
            // tiny_math supports add_scaled, so we add (-lr) * grad
            self.weights.add_scaled(&weight_grad, -effective_lr);
        }

        (loss, relation)
    }

    pub fn save_to_json(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load_from_json(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let learner = serde_json::from_reader(reader)?;
        Ok(learner)
    }
}
