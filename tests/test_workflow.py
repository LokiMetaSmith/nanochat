import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import argparse

# Ensure we can import the script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts import workflow

class TestWorkflow(unittest.TestCase):

    @patch('scripts.workflow.run_command')
    @patch('scripts.workflow.run_background')
    @patch('scripts.workflow.detect_nproc_per_node')
    @patch('scripts.workflow.setup_tokenizer')
    def test_workflow_tiny(self, mock_setup_tok, mock_nproc, mock_bg, mock_run):
        # Setup mocks
        mock_nproc.return_value = 1
        mock_bg.return_value = MagicMock()
        mock_bg.return_value.returncode = 0

        # Simulate CLI args
        with patch('argparse.ArgumentParser.parse_known_args',
                   return_value=(argparse.Namespace(job='tiny', tune_optimizer=False, tune_lr=False, tune_hyperparams=False, try_all_variations=False), [])):
            workflow.main()

        # Verify calls
        mock_setup_tok.assert_called_once()
        # Check dataset download
        mock_bg.assert_called()
        # Check torchrun calls
        # We expect calls to base_train, base_loss, base_eval, etc.
        # Just check a few signatures

        calls = [str(call) for call in mock_run.mock_calls]
        self.assertTrue(any("scripts.base_train" in c and "configs/tiny.json" in c for c in calls))
        self.assertTrue(any("scripts.base_eval" in c and "--max-per-task=16" in c for c in calls))

    @patch('scripts.workflow.run_command')
    @patch('scripts.workflow.run_background')
    @patch('scripts.workflow.detect_nproc_per_node')
    @patch('scripts.workflow.setup_tokenizer')
    def test_workflow_speed(self, mock_setup_tok, mock_nproc, mock_bg, mock_run):
        # Setup mocks
        mock_nproc.return_value = 8
        mock_bg.return_value = MagicMock()
        mock_bg.return_value.returncode = 0

        # Simulate CLI args
        with patch('argparse.ArgumentParser.parse_known_args',
                   return_value=(argparse.Namespace(job='speed', tune_optimizer=False, tune_lr=False, tune_hyperparams=False, try_all_variations=False), [])):
            workflow.main()

        # Verify calls
        calls = [str(call) for call in mock_run.mock_calls]
        self.assertTrue(any("scripts.base_train" in c and "configs/speedrun.json" in c for c in calls))
        # Speedrun base_eval has no extra args
        self.assertFalse(any("scripts.base_eval" in c and "--max-per-task=16" in c for c in calls))
        self.assertTrue(any("nproc_per_node=8" in c for c in calls))

if __name__ == '__main__':
    unittest.main()
