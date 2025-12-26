import os
import sys
import argparse
import subprocess
import shutil
import time
import signal
import torch

def get_base_dir():
    return os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

def ensure_base_dir():
    base_dir = get_base_dir()
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def run_command(cmd, env=None, check=True):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=check)

def run_background(cmd, env=None):
    print(f"Running background: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)

def detect_nproc_per_node():
    if torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.cuda.device_count()
    return 1

def setup_tokenizer():
    # Build rustbpe if not importable
    try:
        import rustbpe
    except ImportError:
        print("rustbpe not found. Building...")
        # Assume run.sh or environment has setup maturin availability
        # In unified run.sh, this should be handled, but as a fallback:
        subprocess.run(["maturin", "develop", "--release", "--manifest-path", "rustbpe/Cargo.toml"], check=True)

    # Train/Eval Tokenizer
    print("Training/Evaluating Tokenizer...")
    # Note: These scripts use 'python -m ...' style
    run_command([sys.executable, "-m", "scripts.tok_train", "--max_chars=2000000000"])
    run_command([sys.executable, "-m", "scripts.tok_eval"])

def download_dataset_background(job_type):
    # Download first chunk synchronously
    print("Downloading initial dataset chunk...")
    run_command([sys.executable, "-m", "nanochat.dataset", "-n", "8"])

    # Download rest in background
    total_shards = 240 # Default from scripts
    print(f"Downloading remaining {total_shards} shards in background...")
    return run_background([sys.executable, "-m", "nanochat.dataset", "-n", str(total_shards)])

def tune_system(config_file, tuner_args):
    print("Starting System/Optimizer Tuning...")
    cmd = [sys.executable, "-m", "scripts.tune_system", "--config", config_file] + tuner_args
    run_command(cmd)
    if os.path.exists("run_config.json"):
        print("Tuning complete. Switching configuration to run_config.json")
        return "run_config.json"
    return config_file

def main():
    parser = argparse.ArgumentParser(description="Nanochat Workflow Orchestrator")
    parser.add_argument("--job", choices=["tiny", "speed"], required=True, help="Job type (tiny or speed)")
    parser.add_argument("--tune-optimizer", action="store_true", help="Enable optimizer tuning")
    parser.add_argument("--tune-lr", action="store_true", help="Enable LR tuning")
    parser.add_argument("--tune-hyperparams", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--try-all-variations", action="store_true", help="Try all tuning variations")
    parser.add_argument("--skip-workarounds", action="store_true", help="Skip stability workarounds to test driver fixes")

    args, unknown_args = parser.parse_known_args()

    # Setup environment
    base_dir = ensure_base_dir()
    os.environ["NANOCHAT_BASE_DIR"] = base_dir
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    wandb_run = os.environ.get("WANDB_RUN", "dummy")

    # Reset Report
    run_command([sys.executable, "-m", "nanochat.report", "reset"])

    # Configuration selection
    if args.job == "tiny":
        config_file = "configs/tiny.json"
    else: # speed
        config_file = "configs/speedrun.json"

    # Handle Tuning
    tuner_args = []
    if args.tune_optimizer: tuner_args.append("--tune-optimizer")
    if args.tune_lr: tuner_args.append("--tune-lr")
    if args.tune_hyperparams: tuner_args.append("--tune-hyperparams")
    if args.try_all_variations: tuner_args.append("--try-all-variations")

    # Tokenizer & Dataset
    setup_tokenizer()
    bg_download_proc = download_dataset_background(args.job)

    # System Tuning
    if args.skip_workarounds:
        tuner_args.append("--skip-workarounds")
        # Explicitly set it in the current environment to ensure it persists
        # even if something goes wrong with loading run_config.json
        print("Enabling NANOCHAT_SKIP_WORKAROUNDS=1 globally.")
        os.environ["NANOCHAT_SKIP_WORKAROUNDS"] = "1"

    if tuner_args:
        config_file = tune_system(config_file, tuner_args)

    # Load environment variables from run_config.json if it exists and we used it
    # We check basename to handle paths like ./run_config.json
    if os.path.basename(config_file) == "run_config.json" and os.path.exists(config_file):
        try:
            with open(config_file) as f:
                data = json.load(f)
                if "env_vars" in data and isinstance(data["env_vars"], dict):
                    print(f"Loading environment variables from {config_file}...")
                    for k, v in data["env_vars"].items():
                        print(f"  Setting {k}={v}")
                        os.environ[k] = str(v)
        except Exception as e:
            print(f"Warning: Failed to load env_vars from {config_file}: {e}")

    # Wait for download if needed? The original scripts just let it run.
    # But wait, original script: `wait $DATASET_DOWNLOAD_PID` right after tok_eval.
    # We should probably wait here before starting training to ensure some data exists.
    print("Waiting for dataset download to complete...")
    bg_download_proc.wait()
    if bg_download_proc.returncode != 0:
        print("Dataset download failed!")
        sys.exit(1)

    # Detect Process Count
    nproc = detect_nproc_per_node()
    print(f"Using {nproc} processes per node.")

    if nproc == 1 and not (torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip)):
        if "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]

    # Training Sequence
    # Helper for torchrun
    def torchrun(module, *cmd_args):
        cmd = [
            "torchrun", "--standalone", f"--nproc_per_node={nproc}",
            "-m", module, config_file
        ] + list(cmd_args)
        run_command(cmd)

    # 1. Base Train
    torchrun("scripts.base_train", f"--run={wandb_run}")

    # 2. Base Loss
    torchrun("scripts.base_loss")

    # 3. Base Eval
    # base_eval args in tinyrun: -- --max-per-task=16
    # base_eval args in speedrun: (none)
    eval_args = ["--", "--max-per-task=16"] if args.job == "tiny" else []
    torchrun("scripts.base_eval", *eval_args)

    # 4. Identity Data
    print("Ensuring identity data...")
    run_command([sys.executable, "-m", "scripts.ensure_identity_data"])

    # 5. Mid Train
    torchrun("scripts.mid_train", f"--run={wandb_run}")

    # 6. Chat Eval (Mid)
    torchrun("scripts.chat_eval", "--", "-i", "mid")

    # 7. Chat SFT
    torchrun("scripts.chat_sft", f"--run={wandb_run}")

    # 8. Chat Eval (SFT)
    torchrun("scripts.chat_eval", "--", "-i", "sft")

    # Report
    run_command([sys.executable, "-m", "nanochat.report", "generate"])

if __name__ == "__main__":
    main()
