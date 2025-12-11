use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder};
use crate::config::Config;
use crate::diffusion::DiffusionHead;

// --- GPT Blocks ---

pub struct CausalSelfAttention {
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let c_q = linear_no_bias(cfg.n_embd, cfg.n_head * head_dim, vb.pp("c_q"))?;
        let c_k = linear_no_bias(cfg.n_embd, cfg.n_kv_head * head_dim, vb.pp("c_k"))?;
        let c_v = linear_no_bias(cfg.n_embd, cfg.n_kv_head * head_dim, vb.pp("c_v"))?;
        let c_proj = linear_no_bias(cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_q,
            c_k,
            c_v,
            c_proj,
            n_head: cfg.n_head,
            n_kv_head: cfg.n_kv_head,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, _rotary_emb: &Tensor) -> Result<Tensor> {
        let (b_sz, t, c) = x.dims3()?;

        let q = self.c_q.forward(x)?;
        let k = self.c_k.forward(x)?;
        let v = self.c_v.forward(x)?;

        let q = q.reshape((b_sz, t, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, t, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, t, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        // Naive RoPE placeholder (simplified for now)
        // TODO: Implement actual RoPE application in Rust

        // Attention (Naive scaled dot product)
        // TODO: Use Flash Attention if available or implement SDPA

        // Placeholder attention mechanism to ensure compilation
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_probs = candle_nn::ops::softmax(&attn_weights, 3)?;
        let y = attn_probs.matmul(&v)?;

        let y = y.transpose(1, 2)?.reshape((b_sz, t, c))?;
        let y = self.c_proj.forward(&y)?;
        Ok(y)
    }
}

pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear_no_bias(cfg.n_embd, 4 * cfg.n_embd, vb.pp("c_fc"))?;
        let c_proj = linear_no_bias(4 * cfg.n_embd, cfg.n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?; // relu^2
        self.c_proj.forward(&x)
    }
}

pub struct Block {
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl Block {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self { attn, mlp })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Manual RMSNorm (no params)
        let x_norm = rms_norm(x)?;
        let x = (x + self.attn.forward(&x_norm, x)?)?; // Dummy rotary passing x for now
        let x_norm = rms_norm(&x)?;
        let x = (x + self.mlp.forward(&x_norm)?)?;
        Ok(x)
    }
}

fn rms_norm(x: &Tensor) -> Result<Tensor> {
    let _dim = x.dim(x.rank() - 1)?;
    let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;
    let x_normed = x.broadcast_div(&variance.sqrt()?)?;
    Ok(x_normed)
}

// --- Vision ---

pub struct PatchEmbed {
    proj: candle_nn::Conv2d,
    _img_size: usize,
    _patch_size: usize,
}

impl PatchEmbed {
    pub fn new(img_size: usize, patch_size: usize, in_chans: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = candle_nn::conv2d(in_chans, embed_dim, patch_size,
            candle_nn::Conv2dConfig { stride: patch_size, ..Default::default() },
            vb.pp("proj")
        )?;
        Ok(Self { proj, _img_size: img_size, _patch_size: patch_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, H, W)
        let x = self.proj.forward(x)?;
        // flatten and transpose: (B, embed_dim, grid, grid) -> (B, num_patches, embed_dim)
        let (b, c, _h, _w) = x.dims4()?;
        let x = x.flatten_from(2)?.transpose(1, 2)?;
        // Ensure result is (B, num_patches, c) - dims checked above
        Ok(x)
    }
}

pub struct VisionTransformer {
    // Simplified ViT port stub
    // In full implementation, we'd add pos_embed, blocks, norm
    patch_embed: PatchEmbed,
}

impl VisionTransformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // Just PatchEmbed for now to verify integration
        let patch_embed = PatchEmbed::new(
            cfg.vision_image_size,
            cfg.vision_patch_size,
            3,
            cfg.vision_width,
            vb.pp("patch_embed")
        )?;
        Ok(Self { patch_embed })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.patch_embed.forward(x)
    }
}

// --- Robotics ---

pub struct Projector {
    net: candle_nn::Sequential,
    n_tokens: usize,
    output_dim: usize,
}

impl Projector {
    pub fn new(in_dim: usize, out_dim: usize, n_tokens: usize, vb: VarBuilder) -> Result<Self> {
        let total_out = n_tokens * out_dim;
        let fc1 = candle_nn::linear(in_dim, total_out, vb.pp("net.0"))?;
        let fc2 = candle_nn::linear(total_out, total_out, vb.pp("net.2"))?;

        Ok(Self {
            net: candle_nn::seq().add(fc1).add(candle_nn::Activation::Gelu).add(fc2),
            n_tokens,
            output_dim: out_dim
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b_sz = x.dims()[0];
        let x = self.net.forward(x)?;
        x.reshape((b_sz, self.n_tokens, self.output_dim))
    }
}

pub struct ActionHead {
    net: candle_nn::Sequential,
}

impl ActionHead {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(in_dim, in_dim, vb.pp("net.0"))?;
        let fc2 = candle_nn::linear(in_dim, out_dim, vb.pp("net.2"))?;
        Ok(Self {
            net: candle_nn::seq().add(fc1).add(candle_nn::Activation::Gelu).add(fc2)
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.net.forward(x)
    }
}

pub struct RoboticsInterface {
    sensor_proj: Option<Projector>,
    surface_proj: Option<Projector>,
    action_head: Option<ActionHead>,
    diffusion_head: Option<DiffusionHead>,
}

impl RoboticsInterface {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let sensor_proj = if cfg.robotics_sensor_dim > 0 {
            Some(Projector::new(
                cfg.robotics_sensor_dim,
                cfg.n_embd,
                cfg.robotics_sensor_tokens,
                vb.pp("sensor_proj")
            )?)
        } else {
            None
        };

        let surface_proj = if cfg.robotics_surface_dim > 0 {
            Some(Projector::new(
                cfg.robotics_surface_dim,
                cfg.n_embd,
                cfg.robotics_surface_tokens,
                vb.pp("surface_proj")
            )?)
        } else {
            None
        };

        // Action Head (Diffusion or Regression)
        let (action_head, diffusion_head) = if cfg.robotics_surface_dim > 0 {
            if cfg.robotics_use_diffusion {
                (
                    None,
                    Some(DiffusionHead::new(
                        cfg.n_embd, // input dim (hidden state)
                        cfg.robotics_surface_dim, // output dim (action)
                        cfg,
                        vb.pp("diffusion_head")
                    )?)
                )
            } else {
                (
                    Some(ActionHead::new(
                        cfg.n_embd,
                        cfg.robotics_surface_dim,
                        vb.pp("action_head")
                    )?),
                    None
                )
            }
        } else {
            (None, None)
        };

        Ok(Self { sensor_proj, surface_proj, action_head, diffusion_head })
    }

    pub fn forward(&self, sensors: Option<&Tensor>, surface: Option<&Tensor>) -> Result<Option<Tensor>> {
        let mut embeddings = Vec::new();

        if let (Some(proj), Some(input)) = (&self.sensor_proj, sensors) {
            embeddings.push(proj.forward(input)?);
        }

        if let (Some(proj), Some(input)) = (&self.surface_proj, surface) {
            embeddings.push(proj.forward(input)?);
        }

        if embeddings.is_empty() {
            return Ok(None);
        }

        // Concatenate along time dimension (dim 1)
        Tensor::cat(&embeddings, 1).map(Some)
    }

    pub fn predict_action(&self, hidden_state: &Tensor) -> Result<Option<Tensor>> {
        if let Some(diff) = &self.diffusion_head {
            return diff.sample(hidden_state).map(Some);
        }
        if let Some(act) = &self.action_head {
            return act.forward(hidden_state).map(Some);
        }
        Ok(None)
    }
}

// --- Main Model ---

pub struct GPT {
    wte: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    robotics: Option<RoboticsInterface>,
}

impl GPT {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.n_embd, vb.pp("transformer.wte"))?;
        let blocks: Vec<Block> = (0..cfg.n_layer)
            .map(|i| Block::new(cfg, vb.pp(format!("transformer.h.{}", i))))
            .collect::<Result<Vec<_>>>()?;
        let lm_head = linear_no_bias(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head"))?;

        let robotics = if cfg.use_robotics {
            Some(RoboticsInterface::new(cfg, vb.pp("robotics_interface"))?)
        } else {
            None
        };

        Ok(Self { wte, blocks, lm_head, robotics })
    }

    // Returns (Logits, Optional Action Prediction)
    pub fn forward(&self, x: &Tensor, sensors: Option<&Tensor>, surface: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let mut x = self.wte.forward(x)?;

        // Prepend Robotics Tokens
        if let Some(robotics) = &self.robotics {
            if let Some(robot_emb) = robotics.forward(sensors, surface)? {
                x = Tensor::cat(&[&robot_emb, &x], 1)?;
            }
        }

        // Apply Blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final Norm
        x = rms_norm(&x)?;

        // Logits
        let logits = self.lm_head.forward(&x)?;

        // Action Prediction (from last token)
        let mut action_pred = None;
        if let Some(robotics) = &self.robotics {
            // Get last hidden state: x[:, -1, :]
            let (_b, t, _c) = x.dims3()?;
            let last_hidden = x.narrow(1, t-1, 1)?.squeeze(1)?;
            action_pred = robotics.predict_action(&last_hidden)?;
        }

        Ok((logits, action_pred))
    }
}
