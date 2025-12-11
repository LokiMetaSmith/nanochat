import torch
import unittest
from nanochat.gpt import GPT, GPTConfig
from nanochat.robotics import RoboticsConfig

class TestRoboticsIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.max_seq_len = 32

        # Vision Setup
        self.vision_image_size = 32
        self.vision_patch_size = 4
        self.num_vision_patches = (self.vision_image_size // self.vision_patch_size) ** 2

        # Robotics Setup
        self.sensor_dim = 16
        self.surface_dim = 32
        self.sensor_tokens = 2
        self.surface_tokens = 3
        self.num_robot_tokens = self.sensor_tokens + self.surface_tokens

        self.total_seq_len = self.max_seq_len + self.num_vision_patches + self.num_robot_tokens

        self.config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            vocab_size=1000,
            sequence_len=self.total_seq_len,
            # Vision
            use_vision=True,
            vision_image_size=self.vision_image_size,
            vision_patch_size=self.vision_patch_size,
            vision_width=32,
            vision_layers=2,
            vision_heads=2,
            # Robotics
            use_robotics=True,
            robotics_sensor_dim=self.sensor_dim,
            robotics_surface_dim=self.surface_dim,
            robotics_sensor_tokens=self.sensor_tokens,
            robotics_surface_tokens=self.surface_tokens,
            robotics_action_loss_weight=1.0,
            robotics_use_diffusion=True, # Test Diffusion
            robotics_diffusion_steps=5
        )
        self.model = GPT(self.config).to(self.device)

    def test_full_modalities_forward(self):
        B = 2
        C, H, W = 3, self.vision_image_size, self.vision_image_size

        # Inputs
        idx = torch.randint(0, 1000, (B, self.max_seq_len)).to(self.device)
        images = torch.randn(B, C, H, W).to(self.device)
        sensors = torch.randn(B, self.sensor_dim).to(self.device)
        surface = torch.randn(B, self.surface_dim).to(self.device)
        action_targets = torch.randn(B, self.surface_dim).to(self.device)

        # Targets
        padding_len = self.num_vision_patches + self.num_robot_tokens
        vision_padding = torch.full((B, padding_len), -1, dtype=torch.long, device=self.device)
        targets_text = torch.randint(0, 1000, (B, self.max_seq_len)).to(self.device)
        targets = torch.cat([vision_padding, targets_text], dim=1)

        # Forward (Training with Diffusion Loss)
        loss = self.model(idx, images=images, sensors=sensors, surface=surface, targets=targets, action_targets=action_targets)

        self.assertTrue(torch.isfinite(loss))

        # Backward
        loss.backward()

        # Check gradients
        # Diffusion Head (Denoiser) should have gradients
        self.assertIsNotNone(self.model.robotics_interface.diffusion_head.denoiser.net[0].weight.grad)

    def test_robotics_inference(self):
        # Test Inference (Diffusion Sampling)
        B = 1
        idx = torch.randint(0, 1000, (B, self.max_seq_len)).to(self.device)
        sensors = torch.randn(B, self.sensor_dim).to(self.device)

        # Forward without targets (Inference)
        # Should call diffusion_head.sample()
        logits, action_pred = self.model(idx, images=None, sensors=sensors, surface=None, targets=None)

        self.assertIsNotNone(action_pred)
        self.assertEqual(action_pred.shape, (B, self.surface_dim))

if __name__ == '__main__':
    unittest.main()
