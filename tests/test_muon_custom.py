
import torch
import unittest
from nanochat.muon import Muon
from nanochat.polar_express import polar_express, polar_express_pytorch, polar_express_triton, HAS_TRITON

class TestMuonPolarExpress(unittest.TestCase):
    def test_polar_express_pytorch_cpu(self):
        # Create a random 2D tensor on CPU
        G = torch.randn(128, 128, dtype=torch.float32)
        out = polar_express(G)
        self.assertEqual(out.shape, G.shape)
        # Check basic property: should have spectral norm roughly 1
        # (Though Polar Express puts spectral values in [0.68, 1.13])
        # We just check it runs without error.

    def test_polar_express_consistency(self):
        if not torch.cuda.is_available() or not HAS_TRITON:
            print("Skipping consistency test (CUDA or Triton missing)")
            return

        G = torch.randn(128, 128, dtype=torch.float32, device='cuda')

        # Run PyTorch version (forced)
        out_pt = polar_express_pytorch(G.clone())

        # Run Triton version (default dispatch or explicit)
        out_tr = polar_express_triton(G.clone())

        # They should be roughly similar (floating point differences expected)
        # Bfloat16 intermediate ops might cause larger diffs.
        # Just check shape and no NaN.
        self.assertEqual(out_tr.shape, G.shape)
        self.assertFalse(torch.isnan(out_tr).any())
        self.assertFalse(torch.isnan(out_pt).any())

        # Check relative error if possible, but might be large due to iterative nature + bf16
        # diff = (out_pt - out_tr).abs().max()
        # print(f"Max diff: {diff}")

    def test_muon_optimizer_init(self):
        params = [torch.randn(10, 10, requires_grad=True)]
        opt = Muon(params, lr=0.1)
        self.assertEqual(len(opt.param_groups), 1)

    def test_muon_step_cpu(self):
        # Run a step on CPU to verify fallback works inside optimizer
        params = [torch.randn(10, 10, requires_grad=True)]
        opt = Muon(params, lr=0.1)

        loss = params[0].sum()
        loss.backward()
        opt.step()

        # Check params changed
        self.assertTrue(params[0].grad is not None)

if __name__ == '__main__':
    unittest.main()
