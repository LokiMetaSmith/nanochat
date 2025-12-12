import os
import sys
import subprocess
import re
import time
import shutil
import json
import argparse
import itertools
from typing import Dict, Any, List, Tuple

def run_benchmark(config_overrides: Dict[str, Any], env_vars: Dict[str, str], base_config_path: str = None, base_config: Dict[str, Any] = None, extra_args: List[str] = [], steps: int = 5, minimal_validation: bool = True) -> float:
    """
    Runs a short training session with the given configuration and environment variables.
    Returns the average tokens per second (tok/sec) or -1.0 if failed.
    """

    # Construct command
    # Use -u for unbuffered output to ensure we capture stdout even if it hangs/crashes
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"]

    # Pass the base config file if provided
    # This allows base_train to safely load all keys (including unknown ones) from the JSON
    if base_config_path:
        cmd.append(base_config_path)

    # Add extra CLI args (user overrides)
    # We add them BEFORE config_overrides so that the tuning loop parameters (batch size) take precedence
    if extra_args:
        cmd.extend(extra_args)

    # Add config overrides as flags
    # These will override values from the base config file
    for key, value in config_overrides.items():
        cmd.append(f"--{key}={value}")

    # Force low number of iterations for speed
    cmd.append(f"--num_iterations={steps}")
    cmd.append("--run=dummy") # Don't log to wandb
    cmd.append("--core_metric_every=-1") # Disable heavy evaluation
    cmd.append("--save_every=-1") # Disable intermediate checkpointing

    # Optimize validation overhead:
    # If minimal_validation is True, reduce eval_tokens to a small multiple of the batch size
    # to ensure validation is near-instant, preventing timeouts on small batch sizes.
    if minimal_validation:
        bs = int(config_overrides.get("device_batch_size", 16))

        # We need max_seq_len to calculate eval_tokens.
        # If it's not in overrides, we need to read it from base_config_path or assume default.
        seq_len = 2048 # default
        if "max_seq_len" in config_overrides:
             seq_len = int(config_overrides["max_seq_len"])
        elif base_config:
             seq_len = int(base_config.get("max_seq_len", 2048))
        elif base_config_path:
            try:
                with open(base_config_path) as f:
                    base_conf = json.load(f)
                    seq_len = int(base_conf.get("max_seq_len", 2048))
            except:
                pass

        eval_tokens = bs * seq_len * 2
        cmd.append(f"--eval_tokens={eval_tokens}")

    # Merge environment variables
    current_env = os.environ.copy()
    current_env.update(env_vars)

    print(f"Running benchmark with overrides: {config_overrides} env: {env_vars}", flush=True)

    try:
        # Capture output to parse tok/sec
        result = subprocess.run(
            cmd,
            env=current_env,
            capture_output=True,
            text=True,
            timeout=1200 # 20 minute timeout per run
        )

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}", flush=True)
            # Check for OOM
            if "OutOfMemoryError" in result.stderr or "OutOfMemoryError" in result.stdout:
                print("Failure reason: OutOfMemoryError", flush=True)
            else:
                print(f"Stderr tail: {result.stderr[-5000:]}", flush=True)
            return -1.0

        # Parse output for tok/sec
        tok_sec_values = []
        # Skip the first few steps as they might be slow (compilation, warmup)
        warmup_steps = 2

        for line in result.stdout.splitlines():
            match = re.search(r"tok/sec:\s*([\d,]+)", line)
            if match:
                step_match = re.search(r"step\s+(\d+)", line)
                if step_match:
                    step = int(step_match.group(1))
                    if step > warmup_steps:
                        val = float(match.group(1).replace(',', ''))
                        tok_sec_values.append(val)

        if not tok_sec_values:
            print("Could not parse tok/sec from output", flush=True)
            return -1.0

        avg_tok_sec = sum(tok_sec_values) / len(tok_sec_values)
        print(f"Result: {avg_tok_sec:.2f} tok/sec", flush=True)
        return avg_tok_sec

    except subprocess.TimeoutExpired as e:
        print(f"Run timed out after {e.timeout} seconds", flush=True)
        print("--- Stdout during timeout ---", flush=True)
        if e.stdout:
            print(e.stdout, flush=True)
        else:
            print("(No stdout captured)", flush=True)

        print("--- Stderr during timeout ---", flush=True)
        if e.stderr:
            print(e.stderr, flush=True)
        else:
            print("(No stderr captured)", flush=True)

        return -1.0
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return -1.0

def run_loss_benchmark(config_overrides: Dict[str, Any], env_vars: Dict[str, str], base_config_path: str = None, base_config: Dict[str, Any] = None, extra_args: List[str] = [], steps: int = 50) -> float:
    """
    Runs a training session for a fixed number of steps and returns the final (smoothed) loss.
    Returns float("inf") if failed.
    """

    # Construct command
    cmd = [sys.executable, "-u", "-m", "scripts.base_train"]

    if base_config_path:
        cmd.append(base_config_path)

    if extra_args:
        cmd.extend(extra_args)

    for key, value in config_overrides.items():
        cmd.append(f"--{key}={value}")

    cmd.append(f"--num_iterations={steps}")
    cmd.append("--run=dummy")
    cmd.append("--core_metric_every=-1")
    cmd.append("--save_every=-1")

    # We still need a valid eval_tokens calculation to avoid errors in base_train
    bs = int(config_overrides.get("device_batch_size", 16))
    seq_len = 2048
    if "max_seq_len" in config_overrides:
         seq_len = int(config_overrides["max_seq_len"])
    elif base_config:
         seq_len = int(base_config.get("max_seq_len", 2048))
    elif base_config_path:
        try:
            with open(base_config_path) as f:
                base_conf = json.load(f)
                seq_len = int(base_conf.get("max_seq_len", 2048))
        except:
            pass

    eval_tokens = bs * seq_len * 2
    cmd.append(f"--eval_tokens={eval_tokens}")

    current_env = os.environ.copy()
    current_env.update(env_vars)

    print(f"Running loss benchmark (steps={steps}) with overrides: {config_overrides}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            env=current_env,
            capture_output=True,
            text=True,
            timeout=3600 # 1 hour timeout for convergence tests
        )

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}", flush=True)
            if "OutOfMemoryError" in result.stderr or "OutOfMemoryError" in result.stdout:
                print("Failure reason: OutOfMemoryError", flush=True)
            else:
                print(f"Stderr tail: {result.stderr[-2000:]}", flush=True)
            return float("inf")

        # Parse output for loss
        final_loss = float("inf")
        # Look for lines like: step 00050/... | loss: 6.123456 | ...
        # We take the loss from the last few steps
        losses = []
        for line in result.stdout.splitlines():
            # Regex to capture loss
            match = re.search(r"loss:\s*([\d\.]+)", line)
            if match:
                step_match = re.search(r"step\s+(\d+)", line)
                if step_match:
                    losses.append(float(match.group(1)))

        if not losses:
            print("Could not parse loss from output", flush=True)
            return float("inf")

        # Average the last 5 losses to smooth out noise
        last_n = min(len(losses), 5)
        final_loss = sum(losses[-last_n:]) / last_n

        print(f"Result: {final_loss:.4f} loss", flush=True)
        return final_loss

    except subprocess.TimeoutExpired as e:
        print(f"Run timed out", flush=True)
        return float("inf")
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        return float("inf")


def main():
    print("Starting System Auto-Tuning...", flush=True)

    parser = argparse.ArgumentParser(description="Auto-tune system performance")
    parser.add_argument("--config", type=str, default=None, help="Path to base JSON configuration file")
    parser.add_argument("--tune-lr", action="store_true", help="Enable Learning Rate tuning (slow)")
    parser.add_argument("--tune-optimizer", action="store_true", help="Enable Optimizer tuning (slow)")
    # New flag: tune-hyperparams (alias for enabling comprehensive hyperparam tuning including LoRA)
    parser.add_argument("--tune-hyperparams", action="store_true", help="Enable comprehensive hyperparameter tuning (LR, Sched, LoRA)")
    args, unknown = parser.parse_known_args()

    # Consolidate flags
    if args.tune_hyperparams:
        args.tune_lr = True
        args.tune_optimizer = True

    # Load Base Configuration for reference (but don't rely on it for cmd construction unless needed)
    base_config = {}
    if args.config:
        print(f"Loading base configuration from {args.config}", flush=True)
        try:
            with open(args.config) as f:
                base_config = json.load(f)
        except Exception as e:
             print(f"Error loading config file: {e}", flush=True)
             sys.exit(1)
    else:
        print("Using default configuration (Depth 10)", flush=True)
        base_config = {"depth": 10, "max_seq_len": 2048}

    # Parse unknown args to update base_config for local logic (e.g. depth, seq_len)
    for arg in unknown:
        if arg.startswith("--"):
            key_val = arg[2:]
            if "=" in key_val:
                key, val = key_val.split("=", 1)
                # Try to convert to int/float/bool
                if val.lower() == "true": val = True
                elif val.lower() == "false": val = False
                else:
                    try:
                        if "." in val: val = float(val)
                        else: val = int(val)
                    except ValueError:
                        pass # keep as string
                base_config[key] = val

    # 1. Hardware Detection (Basic)
    is_rocm = False
    try:
        if os.path.exists("/dev/kfd"):
             is_rocm = True
    except:
        pass

    # Check for Strix Halo (gfx1151)
    is_strix_halo = False
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.5.1":
        is_strix_halo = True
    elif shutil.which('rocminfo'):
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            if 'gfx1151' in result.stdout:
                is_strix_halo = True
        except:
            pass

    print(f"Detected Platform: {'ROCm/AMD' if is_rocm else 'CUDA/NVIDIA/CPU'}", flush=True)
    if is_strix_halo:
        print("Detected Variant: Strix Halo (gfx1151)", flush=True)

    # 2. Define Search Space
    # Batch sizes to try. We start small and go up.
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # Compilation flags
    compile_options = [True, False] if is_rocm else [True]
    compile_dynamic_options = [False] # Default to static shapes

    if is_strix_halo:
        print("NOTE: Strix Halo detected. Enabling dynamic=True/False investigation.", flush=True)
        # We allow compile=True to be tested, specifically to investigate dynamic=True vs False
        compile_options = [True, False]
        compile_dynamic_options = [False, True]

    # Environment variable combinations
    env_configs = [{}]
    if is_rocm:
        # Tuning ROCm specific flags
        env_configs = [
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"},
            {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
        ]

    MINIMAL_VALIDATION = True
    if MINIMAL_VALIDATION:
        print("NOTE: Minimal validation enabled (eval_tokens reduced) to prevent timeouts.", flush=True)

    best_throughput = 0.0
    best_overrides = None
    best_env = None

    results = []

    # Grid search for Throughput
    print("\nPhase 1: Throughput Tuning (Batch Size & Compilation)", flush=True)
    depth = base_config.get("depth", 10) # default fallback
    seq_len = int(base_config.get("max_seq_len", 2048))

    for env_vars in env_configs:
        for compile_opt in compile_options:
            current_dynamic_options = compile_dynamic_options if compile_opt else [False]

            for dynamic_opt in current_dynamic_options:
                for bs in batch_sizes:
                    # Construct overrides
                    overrides = {
                        "device_batch_size": bs,
                        "depth": depth,
                        "compile": str(compile_opt),
                        "compile_dynamic": str(dynamic_opt),
                        "eval_tokens": bs * seq_len, # Scale validation to avoid timeout (1 step)
                    }

                    throughput = run_benchmark(overrides, env_vars, base_config_path=args.config, base_config=base_config, extra_args=unknown, minimal_validation=MINIMAL_VALIDATION)

                    if throughput > 0:
                        results.append((overrides, env_vars, throughput))
                        if throughput > best_throughput:
                            best_throughput = throughput
                            best_overrides = overrides
                            best_env = env_vars
                    else:
                        # If we failed (likely OOM), larger batch sizes will likely also fail
                        # So break the inner loop
                        print(f"Batch size {bs} failed, stopping search for this env config.", flush=True)
                        break

    if not results:
        print("No successful runs found.", flush=True)
        sys.exit(1)

    # Sort by throughput
    results.sort(key=lambda x: x[2], reverse=True)

    for ovr, env, tp in results:
        env_str = " ".join([f"{k}={v}" for k,v in env.items()]) if env else "Default Env"
        print(f"Throughput: {tp:,.2f} tok/sec | BS: {ovr['device_batch_size']} | Compile: {ovr['compile']} | Dynamic: {ovr.get('compile_dynamic', 'False')} | Env: {env_str}", flush=True)

    print("\n" + "="*40, flush=True)
    print(f"Best Throughput: {best_throughput:,.2f} tok/sec", flush=True)
    print("="*40, flush=True)

    # -------------------------------------------------------------------------
    # Phase 2: Hyperparameter Tuning (LRs, Optimizer, LoRA)
    # We use the best throughput config as the base
    # -------------------------------------------------------------------------

    final_config = base_config.copy()
    if best_overrides:
        final_config.update(best_overrides)
        # Type conversion for JSON
        if "compile" in final_config and final_config["compile"] == "True": final_config["compile"] = True
        if "compile" in final_config and final_config["compile"] == "False": final_config["compile"] = False

    # Also check if we should tune torch.compile modes if compilation is enabled
    if final_config.get("compile") is True:
        # Simple heuristic: try 'reduce-overhead' and 'default'
        # 'max-autotune' is often too slow for this quick tuner
        print("\nPhase 1.5: Tuning Compilation Mode", flush=True)
        # We already have result for default (mode="") which is best_throughput
        best_mode = None
        modes_to_try = ["reduce-overhead"] # Add more if desired

        # We use the best batch size from Phase 1
        bs = final_config["device_batch_size"]

        for mode in modes_to_try:
            overrides = final_config.copy()
            overrides["compile_mode"] = mode

            # Use the best env from throughput tuning
            tp = run_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                minimal_validation=MINIMAL_VALIDATION
            )

            if tp > best_throughput:
                best_throughput = tp
                best_mode = mode
                print(f"New best compilation mode: {mode} (Throughput: {tp:,.2f})", flush=True)
                final_config.update(overrides)

        if best_mode:
            print(f"Selected compilation mode: {best_mode}", flush=True)
        else:
            print("Default compilation mode retained.", flush=True)

    if args.tune_lr:
        print("\nPhase 2: Learning Rate Tuning", flush=True)
        # Define search space
        # We will tune matrix_lr (Muon) and embedding_lr (Adam)
        # Center around defaults or current values
        base_matrix_lr = float(final_config.get("matrix_lr", 0.02))
        base_embed_lr = float(final_config.get("embedding_lr", 0.2))

        # Grid: 0.5x, 1.0x, 2.0x
        multipliers = [0.5, 1.0, 2.0]

        best_loss = float("inf")
        best_lr_config = {}

        lr_combinations = list(itertools.product(multipliers, multipliers))

        for m_mult, e_mult in lr_combinations:
            m_lr = base_matrix_lr * m_mult
            e_lr = base_embed_lr * e_mult

            overrides = final_config.copy()
            overrides["matrix_lr"] = m_lr
            overrides["embedding_lr"] = e_lr

            # Use the best env from throughput tuning
            loss = run_loss_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                steps=50 # Run for 50 steps to see convergence
            )

            if loss < best_loss:
                best_loss = loss
                best_lr_config = {"matrix_lr": m_lr, "embedding_lr": e_lr}

        if best_lr_config:
            print(f"Found better LRs: {best_lr_config} with loss {best_loss:.4f}")
            final_config.update(best_lr_config)

    # Tuning schedule parameters (warmup_ratio)
    if args.tune_lr: # Include schedule tuning with LR tuning
        print("\nPhase 2.5: Schedule Tuning (warmup_ratio)", flush=True)
        base_warmup = float(final_config.get("warmup_ratio", 0.0))
        warmups = [0.0, 0.05, 0.1]

        best_warmup_loss = float("inf")
        best_warmup_config = {}

        for w in warmups:
            overrides = final_config.copy()
            overrides["warmup_ratio"] = w

            loss = run_loss_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                steps=50
            )

            if loss < best_warmup_loss:
                best_warmup_loss = loss
                best_warmup_config = {"warmup_ratio": w}

        if best_warmup_config:
             print(f"Found better warmup_ratio: {best_warmup_config} with loss {best_warmup_loss:.4f}")
             final_config.update(best_warmup_config)

    if args.tune_optimizer:
        print("\nPhase 3: Optimizer Tuning", flush=True)
        # Example: Tune weight decay
        base_wd = float(final_config.get("weight_decay", 0.0))
        # Try 0.0, 0.01, 0.1
        wds = [0.0, 0.01, 0.1]

        best_loss = float("inf")
        best_opt_config = {}

        for wd in wds:
            overrides = final_config.copy()
            overrides["weight_decay"] = wd

            loss = run_loss_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                steps=50
            )

            if loss < best_loss:
                best_loss = loss
                best_opt_config = {"weight_decay": wd}

        if best_opt_config:
             print(f"Found better Optimizer params: {best_opt_config} with loss {best_loss:.4f}")
             final_config.update(best_opt_config)

    # -------------------------------------------------------------------------
    # Phase 4: LoRA Tuning (if use_lora=True)
    # -------------------------------------------------------------------------
    # Check if LoRA is enabled in config or overrides
    use_lora = final_config.get("use_lora", False)
    # Check if CLI args enable it (as string/bool)
    if isinstance(use_lora, str):
        use_lora = use_lora.lower() == "true"

    if args.tune_hyperparams and use_lora:
        print("\nPhase 4: LoRA Hyperparameter Tuning", flush=True)
        # Tune Rank and Alpha
        # Rank: Try [8, 16, 32]
        # Alpha: Try [1x, 2x] of Rank

        ranks = [8, 16, 32]
        alpha_ratios = [1, 2] # alpha = rank * ratio

        best_lora_loss = float("inf")
        best_lora_config = {}

        for r in ranks:
            for ratio in alpha_ratios:
                alpha = r * ratio
                overrides = final_config.copy()
                overrides["lora_rank"] = r
                overrides["lora_alpha"] = alpha

                loss = run_loss_benchmark(
                    overrides,
                    best_env,
                    base_config_path=args.config,
                    base_config=base_config,
                    extra_args=unknown,
                    steps=50
                )

                if loss < best_lora_loss:
                    best_lora_loss = loss
                    best_lora_config = {"lora_rank": r, "lora_alpha": alpha}

        if best_lora_config:
             print(f"Found better LoRA params: {best_lora_config} with loss {best_lora_loss:.4f}")
             final_config.update(best_lora_config)

    # -------------------------------------------------------------------------
    # Final Output
    # -------------------------------------------------------------------------

    print("\nRecommended Updated Configuration:", flush=True)
    print("You can update your config file with these values:")
    print("-" * 20)

    print(json.dumps(final_config, indent=4), flush=True)
    print("-" * 20)

    if best_env:
        print("Recommended Environment Variables:", flush=True)
        for k, v in best_env.items():
            print(f"  export {k}={v}", flush=True)

    # Command line suggestion
    cmd_args = " ".join([f"--{k}={v}" for k,v in final_config.items()])
    print(f"\nRun command with updated profile:\npython -m scripts.base_train {args.config if args.config else ''} --device_batch_size={final_config['device_batch_size']} --compile={final_config['compile']} --run=$WANDB_RUN", flush=True)

    # Export results to JSON
    json_output = {
        "parameters": final_config,
        "env_vars": best_env if best_env else {}
    }

    # We also include the 'throughput' in the export for reference
    json_output["throughput"] = best_throughput

    with open("run_config.json", "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"\nBest configuration exported to run_config.json", flush=True)

if __name__ == "__main__":
    main()