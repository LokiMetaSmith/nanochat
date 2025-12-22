import sys
import os
import argparse
from unittest.mock import patch, MagicMock

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scripts.workflow as workflow

def print_command(cmd, env=None, check=True):
    # Normalize cmd to string for readable output
    cmd_str = " ".join(cmd)
    # Simplify python path for readability
    cmd_str = cmd_str.replace(sys.executable, "python")
    print(f"[EXEC] {cmd_str}")

def mock_bg(cmd, env=None):
    cmd_str = " ".join(cmd)
    cmd_str = cmd_str.replace(sys.executable, "python")
    print(f"[BG]   {cmd_str}")
    mock = MagicMock()
    mock.returncode = 0
    return mock

def verify(job_type):
    print(f"\n--- Verifying Workflow: {job_type} ---")

    # Mock mocks
    with patch('scripts.workflow.run_command', side_effect=print_command) as mock_run, \
         patch('scripts.workflow.run_background', side_effect=mock_bg) as mock_bg_proc, \
         patch('scripts.workflow.detect_nproc_per_node', return_value=1), \
         patch('scripts.workflow.setup_tokenizer') as mock_tok, \
         patch('argparse.ArgumentParser.parse_known_args', return_value=(argparse.Namespace(job=job_type, tune_optimizer=False, tune_lr=False, tune_hyperparams=False, try_all_variations=False), [])), \
         patch('os.path.exists', return_value=False): # Run config doesn't exist

        # We also need to mock setup_tokenizer internal calls if we want to see them,
        # but the function itself is mocked above.
        # Let's unmock setup_tokenizer to see its internal commands?
        # Actually workflow.setup_tokenizer calls run_command, so if we don't mock setup_tokenizer, we see the calls.

        # Reloading workflow to unpatch setup_tokenizer if it was patched globally? No, patch is context manager.
        pass

    # Re-run with setup_tokenizer unmocked to see full trace
    with patch('scripts.workflow.run_command', side_effect=print_command), \
         patch('scripts.workflow.run_background', side_effect=mock_bg), \
         patch('scripts.workflow.detect_nproc_per_node', return_value=1), \
         patch('argparse.ArgumentParser.parse_known_args', return_value=(argparse.Namespace(job=job_type, tune_optimizer=False, tune_lr=False, tune_hyperparams=False, try_all_variations=False), [])), \
         patch('os.path.exists', return_value=False):

        workflow.main()

if __name__ == "__main__":
    verify("tiny")
    verify("speed")
