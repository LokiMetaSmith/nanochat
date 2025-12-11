use candle_core::{Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, Sequential, VarBuilder, Activation};
use crate::config::Config;

pub struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Module for SinusoidalPosEmb {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let half_dim = self.dim / 2;
        let emb = (10000.0f64).ln() / ((half_dim - 1) as f64);
        let emb = (Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(candle_core::DType::F32)? * -emb)?.exp()?;
        let x = x.unsqueeze(1)?; // (B, 1)
        let emb = emb.unsqueeze(0)?; // (1, half_dim)
        let emb = x.broadcast_mul(&emb)?; // (B, half_dim)
        Tensor::cat(&[&emb.sin()?, &emb.cos()?], 1)
    }
}

pub struct ConditionalDenoiser {
    time_mlp: Sequential,
    cond_proj: Linear,
    net: Sequential,
}

impl ConditionalDenoiser {
    pub fn new(action_dim: usize, cond_dim: usize, hidden_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let time_dim = hidden_dim;

        let time_mlp = candle_nn::seq()
            .add(SinusoidalPosEmb::new(time_dim))
            .add(linear(time_dim, time_dim, vb.pp("time_mlp.1"))?)
            .add(Activation::Gelu) // Using Gelu instead of Mish for simplicity/availability
            .add(linear(time_dim, time_dim, vb.pp("time_mlp.3"))?);

        let cond_proj = linear(cond_dim, hidden_dim, vb.pp("cond_proj"))?;

        let input_dim = action_dim + time_dim + hidden_dim;
        let mut layers = candle_nn::seq();
        let mut in_d = input_dim;

        for i in 0..num_layers {
            layers = layers
                .add(linear(in_d, hidden_dim, vb.pp(format!("net.{}", i * 2)))?)
                .add(Activation::Gelu);
            in_d = hidden_dim;
        }
        layers = layers.add(linear(in_d, action_dim, vb.pp(format!("net.{}", num_layers * 2)))?);

        Ok(Self { time_mlp, cond_proj, net: layers })
    }

    pub fn forward(&self, x: &Tensor, t: &Tensor, cond: &Tensor) -> Result<Tensor> {
        // t must be float for embedding
        let t_float = t.to_dtype(candle_core::DType::F32)?;
        let t_emb = self.time_mlp.forward(&t_float)?;
        let c_emb = self.cond_proj.forward(cond)?;

        // Concatenate: x, t_emb, c_emb
        let h = Tensor::cat(&[x, &t_emb, &c_emb], 1)?;
        self.net.forward(&h)
    }
}

pub struct DiffusionHead {
    denoiser: ConditionalDenoiser,
    timesteps: usize,
    betas: Tensor,
    alphas: Tensor,
    alphas_cumprod: Tensor,
    action_dim: usize,
}

impl DiffusionHead {
    pub fn new(input_dim: usize, output_dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let timesteps = cfg.robotics_diffusion_steps;
        let hidden_dim = 256; // Fixed for now or add to config

        let denoiser = ConditionalDenoiser::new(output_dim, input_dim, hidden_dim, 3, vb.pp("denoiser"))?;

        // Precompute schedule (linear)
        let beta_start = 0.0001f64;
        let beta_end = 0.02f64;
        let betas_vec: Vec<f32> = (0..timesteps).map(|i| {
            let pct = i as f64 / (timesteps - 1) as f64;
            (beta_start + pct * (beta_end - beta_start)) as f32
        }).collect();

        // Create tensors on CPU first (Device::Cpu is safe default, VB device handled later if needed)
        // Ideally we use the device from VB, but VB doesn't expose it directly easily without a tensor.
        // We'll lazy load or just assume we move them during sampling.
        let device = Device::Cpu;
        let betas = Tensor::new(betas_vec.as_slice(), &device)?;
        let alphas = (1.0 - &betas)?;

        // Cumprod
        let mut alphas_cumprod_vec = Vec::with_capacity(timesteps);
        let mut curr = 1.0f32;
        for &a in alphas.to_vec1::<f32>()?.iter() {
            curr *= a;
            alphas_cumprod_vec.push(curr);
        }
        let alphas_cumprod = Tensor::new(alphas_cumprod_vec.as_slice(), &device)?;

        Ok(Self {
            denoiser,
            timesteps,
            betas,
            alphas,
            alphas_cumprod,
            action_dim: output_dim
        })
    }

    pub fn sample(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (b_sz, _) = hidden_state.dims2()?;
        let device = hidden_state.device();

        // Ensure schedule tensors are on correct device
        let alphas = self.alphas.to_device(device)?;
        let alphas_cumprod = self.alphas_cumprod.to_device(device)?;
        let betas = self.betas.to_device(device)?;

        // Start from noise
        let mut x = Tensor::randn(0.0f32, 1.0f32, (b_sz, self.action_dim), device)?;

        for i in (0..self.timesteps).rev() {
            let t = Tensor::full(i as u32, (b_sz,), device)?;

            // Predict noise
            let predicted_noise = self.denoiser.forward(&x, &t, hidden_state)?;

            // Params
            let alpha = alphas.get(i)?.to_scalar::<f32>()?;
            let alpha_hat = alphas_cumprod.get(i)?.to_scalar::<f32>()?;
            let beta = betas.get(i)?.to_scalar::<f32>()?;

            let noise = if i > 0 {
                Tensor::randn(0.0f32, 1.0f32, x.shape(), device)?
            } else {
                Tensor::zeros_like(&x)?
            };

            // DDPM Update: x_{t-1} = 1/sqrt(alpha) * (x_t - (1-alpha)/sqrt(1-alpha_hat) * eps) + sigma * z
            let coeff1 = 1.0 / alpha.sqrt();
            let coeff2 = (1.0 - alpha) / (1.0 - alpha_hat).sqrt();
            let sigma = beta.sqrt();

            let term1 = (x.sub(&(predicted_noise * coeff2 as f64)?)? * (coeff1 as f64))?;
            x = (term1 + (noise * (sigma as f64))?)?;
        }

        Ok(x)
    }
}
