use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub vocab_size: usize,
    pub sequence_len: usize,

    // Vision
    #[serde(default)]
    pub use_vision: bool,
    #[serde(default = "default_vision_image_size")]
    pub vision_image_size: usize,
    #[serde(default = "default_vision_patch_size")]
    pub vision_patch_size: usize,
    #[serde(default = "default_vision_width")]
    pub vision_width: usize,
    #[serde(default = "default_vision_layers")]
    pub vision_layers: usize,
    #[serde(default = "default_vision_heads")]
    pub vision_heads: usize,
    #[serde(default = "default_vision_mlp_ratio")]
    pub vision_mlp_ratio: f64,

    // Robotics
    #[serde(default)]
    pub use_robotics: bool,
    #[serde(default)]
    pub robotics_sensor_dim: usize,
    #[serde(default)]
    pub robotics_surface_dim: usize,
    #[serde(default = "default_one")]
    pub robotics_sensor_tokens: usize,
    #[serde(default = "default_four")]
    pub robotics_surface_tokens: usize,

    // Diffusion
    #[serde(default)]
    pub robotics_use_diffusion: bool,
    #[serde(default = "default_diffusion_steps")]
    pub robotics_diffusion_steps: usize,
}

fn default_vision_image_size() -> usize { 224 }
fn default_vision_patch_size() -> usize { 14 }
fn default_vision_width() -> usize { 768 }
fn default_vision_layers() -> usize { 12 }
fn default_vision_heads() -> usize { 12 }
fn default_vision_mlp_ratio() -> f64 { 4.0 }
fn default_one() -> usize { 1 }
fn default_four() -> usize { 4 }
fn default_diffusion_steps() -> usize { 100 }

impl Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}
