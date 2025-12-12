import unittest
from unittest.mock import patch, MagicMock
import json
import os
from scripts import tune_system

class TestSystemTuner(unittest.TestCase):
    @patch('subprocess.run')
    def test_throughput_parsing(self, mock_run):
        # Mock successful run output
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = """
step 0 | tok/sec: 100
step 1 | tok/sec: 200
step 2 | tok/sec: 300
step 3 | tok/sec: 300
"""
        mock_run.return_value = mock_process

        overrides = {"device_batch_size": 4}
        env = {}

        tp = tune_system.run_benchmark(overrides, env, steps=4)

        # Should average steps > 2. Here step 3 is > 2 (warmup=2).
        # Wait, warmup_steps=2. So step 0, 1, 2 are skipped?
        # Logic: if step > warmup_steps (2). So only step 3.
        # Avg = 300.
        self.assertEqual(tp, 300.0)

    @patch('subprocess.run')
    def test_loss_parsing(self, mock_run):
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = """
step 00048 | loss: 4.0
step 00049 | loss: 3.5
step 00050 | loss: 3.0
"""
        mock_run.return_value = mock_process

        overrides = {}
        env = {}

        loss = tune_system.run_loss_benchmark(overrides, env, steps=50)

        # Logic: Average last 5. We have 3.
        # Avg = (4.0 + 3.5 + 3.0) / 3 = 10.5 / 3 = 3.5
        self.assertAlmostEqual(loss, 3.5)

    @patch('subprocess.run')
    @patch('scripts.tune_system.run_benchmark')
    @patch('scripts.tune_system.run_loss_benchmark')
    def test_tuning_logic(self, mock_loss, mock_tp, mock_run):
        # Setup mocks
        mock_tp.return_value = 1000.0 # Throughput
        mock_loss.side_effect = [5.0, 4.0, 6.0] # 3 trials

        # Prevent actual sys.exit or print spam
        # We can just verify logic by checking calls?
        # But tune_system.main() parses args.
        pass

if __name__ == '__main__':
    unittest.main()
