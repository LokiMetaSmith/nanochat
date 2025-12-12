import unittest
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig
from nanochat.lora import LoRALinear

class TestLoRA(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            sequence_len=32,
            vocab_size=100,
            use_lora=True,
            lora_rank=4,
            lora_alpha=8,
            lora_targets=["q_proj", "v_proj"]
        )
        self.model = GPT(self.config)
        self.model.init_weights()

        # IMPORTANT: GPT.init_weights zeroes out lm_head and c_proj weights.
        # Since these are FROZEN in LoRA mode, gradients from loss will be zeroed out
        # before reaching LoRA layers if we don't initialize them to non-zero.
        nn.init.normal_(self.model.lm_head.weight)
        for block in self.model.transformer.h:
            nn.init.normal_(block.attn.c_proj.weight)
            nn.init.normal_(block.mlp.c_proj.weight)

    def test_lora_structure(self):
        """Verify that configured layers are replaced by LoRALinear."""
        for block in self.model.transformer.h:
            # Check Attention
            self.assertIsInstance(block.attn.c_q, LoRALinear)
            self.assertIsInstance(block.attn.c_v, LoRALinear)
            # Check Unchanged layers
            self.assertIsInstance(block.attn.c_k, nn.Linear)
            self.assertIsInstance(block.attn.c_proj, nn.Linear)

    def test_lora_initialization_equivalence(self):
        """
        Verify that initialized LoRA model produces same output as base model.
        (Since LoRA B is zero-initialized, A @ B = 0).
        """
        # Create a base model without LoRA
        base_config = GPTConfig(
            n_layer=2, n_head=2, n_kv_head=2, n_embd=64, sequence_len=32, vocab_size=100, use_lora=False
        )
        base_model = GPT(base_config)
        base_model.init_weights()

        # Create LoRA model
        lora_model = GPT(self.config)
        lora_model.init_weights()

        # Manually copy weights from base_model to lora_model to ensure equivalence
        with torch.no_grad():
            lora_model.transformer.wte.weight.copy_(base_model.transformer.wte.weight)
            lora_model.lm_head.weight.copy_(base_model.lm_head.weight)

            for i in range(len(base_model.transformer.h)):
                base_block = base_model.transformer.h[i]
                lora_block = lora_model.transformer.h[i]

                # Copy c_q (LoRA wrapped)
                lora_block.attn.c_q.base_layer.weight.copy_(base_block.attn.c_q.weight)
                lora_block.attn.c_k.weight.copy_(base_block.attn.c_k.weight)
                # Copy c_v (LoRA wrapped)
                lora_block.attn.c_v.base_layer.weight.copy_(base_block.attn.c_v.weight)

                # Copy c_proj (Linear)
                lora_block.attn.c_proj.weight.copy_(base_block.attn.c_proj.weight)

                # Copy MLP
                lora_block.mlp.c_fc.weight.copy_(base_block.mlp.c_fc.weight)
                lora_block.mlp.c_proj.weight.copy_(base_block.mlp.c_proj.weight)

        # Forward pass comparison
        x = torch.randint(0, 100, (1, 32))
        base_logits, _ = base_model(x)
        lora_logits, _ = lora_model(x)

        diff = (base_logits - lora_logits).abs().max().item()
        self.assertLess(diff, 1e-5, f"Logits mismatch: max diff {diff}")

    def test_lora_freezing(self):
        """Verify that only LoRA parameters are trainable."""
        optimizers = self.model.setup_optimizers(use_8bit_optimizer=False)
        optimizer = optimizers[0]

        # Check that base parameters have requires_grad=False
        self.assertFalse(self.model.transformer.h[0].attn.c_q.base_layer.weight.requires_grad)
        self.assertFalse(self.model.lm_head.weight.requires_grad)

        # Check that LoRA parameters have requires_grad=True
        self.assertTrue(self.model.transformer.h[0].attn.c_q.lora_A.requires_grad)
        self.assertTrue(self.model.transformer.h[0].attn.c_q.lora_B.requires_grad)

    def test_lora_training_step(self):
        """Run a training step and verify parameter updates."""
        optimizers = self.model.setup_optimizers(use_8bit_optimizer=False)
        optimizer = optimizers[0]

        # Initialize B to random for test so grad_A is non-zero
        # (If B=0, dL/dA=0)
        with torch.no_grad():
             nn.init.normal_(self.model.transformer.h[0].attn.c_q.lora_B)

        # Snapshot LoRA weights
        lora_A_old = self.model.transformer.h[0].attn.c_q.lora_A.clone()
        lora_B_old = self.model.transformer.h[0].attn.c_q.lora_B.clone()
        base_W_old = self.model.transformer.h[0].attn.c_q.base_layer.weight.clone()

        x = torch.randint(0, 100, (2, 32))
        y = torch.randint(0, 100, (2, 32))

        self.model.train()
        loss = self.model(x, targets=y)
        loss.backward()

        grad_B = self.model.transformer.h[0].attn.c_q.lora_B.grad
        self.assertIsNotNone(grad_B, "LoRA B gradient is None")
        self.assertGreater(grad_B.abs().sum().item(), 0.0, "LoRA B gradient is zero")

        optimizer.step()

        # Check updates
        lora_A_new = self.model.transformer.h[0].attn.c_q.lora_A
        lora_B_new = self.model.transformer.h[0].attn.c_q.lora_B
        base_W_new = self.model.transformer.h[0].attn.c_q.base_layer.weight

        diff_B = (lora_B_old - lora_B_new).abs().sum().item()
        diff_W = (base_W_old - base_W_new).abs().sum().item()

        self.assertGreater(diff_B, 0.0, "LoRA B should have updated")
        self.assertEqual(diff_W, 0.0, "Base weight should NOT have updated")

if __name__ == '__main__':
    unittest.main()
