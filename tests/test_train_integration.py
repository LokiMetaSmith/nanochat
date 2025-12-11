import torch
import unittest
from nanochat.gpt import GPT, GPTConfig
from nanochat.robotics import RoboticsConfig

class TestTrainIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.max_seq_len = 32
        self.vision_image_size = 32
        self.vision_patch_size = 4
        self.num_patches = (self.vision_image_size // self.vision_patch_size) ** 2

        self.config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            vocab_size=1000,
            sequence_len=1024, # Large enough
            use_vision=True,
            vision_image_size=self.vision_image_size,
            vision_patch_size=self.vision_patch_size,
            vision_width=32,
            vision_layers=2,
            vision_heads=2
        )
        self.model = GPT(self.config).to(self.device)

    def test_training_padding_logic(self):
        B = 2
        C, H, W = 3, self.vision_image_size, self.vision_image_size

        # Simulate DataLoader output (Text Only)
        idx = torch.randint(0, 1000, (B, self.max_seq_len)).to(self.device)
        targets_raw = torch.randint(0, 1000, (B, self.max_seq_len)).to(self.device)

        # Simulate Vision Data
        images = torch.randn(B, C, H, W).to(self.device)

        # Simulate Padding Logic from base_train.py
        vision_padding = torch.full((B, self.num_patches), -1, dtype=targets_raw.dtype, device=self.device)
        targets_padded = torch.cat([vision_padding, targets_raw], dim=1)

        # Forward Pass
        # Training mode returns single loss
        loss = self.model(idx, images=images, targets=targets_padded)

        self.assertTrue(torch.isfinite(loss))

        # Backward Pass
        loss.backward()
        self.assertIsNotNone(self.model.vision_encoder.patch_embed.proj.weight.grad)

if __name__ == '__main__':
    unittest.main()
