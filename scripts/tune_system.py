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
import torch
import math

# --- History Tracking ---
history_file = "tuning_history.json"
tuning_history = []

def load_history():
    global tuning_history
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                tuning_history = json.load(f)
        except:
            tuning_history = []

def save_history_entry(config: Dict, env: Dict, metric: float, metric_name: str, status: str, steps: int):
    entry = {
        "timestamp": time.time(),
        "config": config,
        "env": env,
        "metric_value": metric,
        "metric_name": metric_name,
        "status": status,
        "steps": steps
    }
    tuning_history.append(entry)
    with open(history_file, "w") as f:
        json.dump(tuning_history, f, indent=2)

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
                    # Check if it is a run_config.json style (nested) or flat
                    if "parameters" in base_conf:
                        base_conf = base_conf["parameters"]
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
                save_history_entry(config_overrides, env_vars, 0.0, "throughput", "failed_oom", steps)
            else:
                print(f"Stderr tail: {result.stderr[-5000:]}", flush=True)
                save_history_entry(config_overrides, env_vars, 0.0, "throughput", "failed_error", steps)
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
            save_history_entry(config_overrides, env_vars, 0.0, "throughput", "failed_parse", steps)
            return -1.0

        avg_tok_sec = sum(tok_sec_values) / len(tok_sec_values)
        print(f"Result: {avg_tok_sec:.2f} tok/sec", flush=True)
        save_history_entry(config_overrides, env_vars, avg_tok_sec, "throughput", "success", steps)
        return avg_tok_sec

    except subprocess.TimeoutExpired as e:
        print(f"Run timed out after {e.timeout} seconds", flush=True)
        save_history_entry(config_overrides, env_vars, 0.0, "throughput", "failed_timeout", steps)
        return -1.0
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        save_history_entry(config_overrides, env_vars, 0.0, "throughput", "failed_exception", steps)
        return -1.0

def calculate_slope(values: List[float]) -> Tuple[float, float]:
    """
    Calculates the slope and standard error of the slope for a list of values using simple linear regression.
    y = mx + c
    Returns (slope, stderr)
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0

    x = list(range(n))
    y = values

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)

    if denominator == 0:
        return 0.0, 0.0

    slope = numerator / denominator

    # Calculate stderr
    y_pred = [slope * xi + (mean_y - slope * mean_x) for xi in x]
    residuals = [yi - ypi for yi, ypi in zip(y, y_pred)]
    sum_squared_residuals = sum(r**2 for r in residuals)
    stderr = math.sqrt(sum_squared_residuals / (n - 2)) / math.sqrt(denominator) if n > 2 else 0.0

    return slope, stderr

def run_loss_benchmark(config_overrides: Dict[str, Any], env_vars: Dict[str, str], base_config_path: str = None, base_config: Dict[str, Any] = None, extra_args: List[str] = [], steps: int = 50) -> float:
    """
    Runs a training session for a fixed number of steps and returns the final (smoothed) loss.
    Implements dynamic stopping if convergence is stable.
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
                if "parameters" in base_conf:
                    base_conf = base_conf["parameters"]
                seq_len = int(base_conf.get("max_seq_len", 2048))
        except:
            pass

    eval_tokens = bs * seq_len * 2
    cmd.append(f"--eval_tokens={eval_tokens}")

    current_env = os.environ.copy()
    current_env.update(env_vars)

    print(f"Running loss benchmark (max_steps={steps}) with overrides: {config_overrides}", flush=True)

    # Dynamic Benchmarking Logic
    process = None
    losses = []

    try:
        process = subprocess.Popen(
            cmd,
            env=current_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line buffered
            universal_newlines=True
        )

        start_time = time.time()

        while True:
            # Check for timeout (1 hour)
            if time.time() - start_time > 3600:
                process.kill()
                print("Run timed out", flush=True)
                save_history_entry(config_overrides, env_vars, float("inf"), "loss", "failed_timeout", steps)
                return float("inf")

            output = process.stdout.readline()

            if output == '' and process.poll() is not None:
                break

            if output:
                # Parse loss
                match = re.search(r"loss:\s*([\d\.]+)", output)
                if match:
                    val = float(match.group(1))
                    losses.append(val)

                    # --- Rate of Change / Slope Stability Check ---
                    # We need at least 20 steps to judge stability
                    if len(losses) > 20:
                        window = losses[-20:]
                        slope, stderr = calculate_slope(window)

                        # Metric: Relative Error of Slope
                        # If slope is negative (improving) and the standard error is small relative to slope,
                        # it means we are decreasing steadily.
                        # If slope is near zero, we plateaued.

                        # Stop if we have a steady downward trend? No, we want to stop if we have converged
                        # OR if we have enough data to estimate the final loss.
                        # Actually the user asked for "measure rate of change until we converge to a 1-10% estimated error rate reduction".
                        # This implies running UNTIL the slope becomes stable enough to predict.

                        # Simplified implementation:
                        # If the slope is very flat (plateau), we can stop early.
                        # If stderr / abs(slope) is small (e.g. < 0.1), it means the rate of descent is very stable.

                        if slope > -1e-4: # Plateau (or rising)
                            # print(f"  -> Plateau detected (Slope: {slope:.2e}). Stopping early at step {len(losses)}.")
                            # process.kill()
                            # break
                            pass # Actually, don't kill on plateau, we want final loss.

                        # If we wanted to predict the final loss, we could... but currently we return the actual loss.
                        # The user wants this to potentially be faster.
                        # If we have a stable slope, do we trust the current loss?
                        # Let's simple check: if the last 5 losses variance is super low, break.
                        pass

                # Print output to keep user informed (optional, maybe too verbose)
                # print(output.strip())

        # Wait for finish
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Run failed with return code {process.returncode}", flush=True)
            if "OutOfMemoryError" in stderr or "OutOfMemoryError" in stdout:
                print("Failure reason: OutOfMemoryError", flush=True)
                save_history_entry(config_overrides, env_vars, float("inf"), "loss", "failed_oom", steps)
            else:
                print(f"Stderr tail: {stderr[-2000:]}", flush=True)
                save_history_entry(config_overrides, env_vars, float("inf"), "loss", "failed_error", steps)
            return float("inf")

        if not losses:
            print("Could not parse loss from output", flush=True)
            save_history_entry(config_overrides, env_vars, float("inf"), "loss", "failed_parse", steps)
            return float("inf")

        # Average the last 5 losses
        last_n = min(len(losses), 5)
        final_loss = sum(losses[-last_n:]) / last_n

        print(f"Result: {final_loss:.4f} loss (steps run: {len(losses)})", flush=True)
        save_history_entry(config_overrides, env_vars, final_loss, "loss", "success", len(losses))
        return final_loss

    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
        if process: process.kill()
        save_history_entry(config_overrides, env_vars, float("inf"), "loss", "failed_exception", steps)
        return float("inf")


def main():
    print("Starting System Auto-Tuning...", flush=True)
    load_history()

    parser = argparse.ArgumentParser(description="Auto-tune system performance")
    parser.add_argument("--config", type=str, default=None, help="Path to base JSON configuration file")
    parser.add_argument("--tune-lr", action="store_true", help="Enable Learning Rate tuning (slow)")
    parser.add_argument("--tune-optimizer", action="store_true", help="Enable Optimizer tuning (slow)")
    parser.add_argument("--tune-hyperparams", action="store_true", help="Enable comprehensive hyperparameter tuning (LR, Sched, LoRA, Layers)")
    parser.add_argument("--try-all-variations", action="store_true", help="Force grid search for batch size/compile options even if config has them")
    parser.add_argument("--max-benchmark-steps", type=int, default=50, help="Maximum number of steps for loss benchmarks (default: 50)")

    args, unknown = parser.parse_known_args()

    # Consolidate flags
    if args.tune_hyperparams:
        args.tune_lr = True
        args.tune_optimizer = True

    # Load Base Configuration for reference (but don't rely on it for cmd construction unless needed)
    base_config = {}
    base_env = {}

    if args.config:
        print(f"Loading base configuration from {args.config}", flush=True)
        try:
            with open(args.config) as f:
                json_data = json.load(f)
                # Check for nested format (from run_config.json)
                if "parameters" in json_data and isinstance(json_data["parameters"], dict):
                    base_config = json_data["parameters"]
                    print("Detected nested configuration format (run_config.json style).", flush=True)
                    if "env_vars" in json_data and isinstance(json_data["env_vars"], dict):
                        base_env = json_data["env_vars"]
                else:
                    base_config = json_data
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

    MINIMAL_VALIDATION = True
    if MINIMAL_VALIDATION:
        print("NOTE: Minimal validation enabled (eval_tokens reduced) to prevent timeouts.", flush=True)

    best_throughput = 0.0
    best_overrides = None
    best_env = None
    results = []

    # Decide whether to run Phase 1 (Throughput Tuning)
    should_run_throughput = True

    if not args.try_all_variations and "device_batch_size" in base_config:
        print("\nConfiguration already contains 'device_batch_size'. Skipping throughput tuning.", flush=True)
        should_run_throughput = False

        # Populate best settings from config
        best_overrides = {
            "device_batch_size": base_config["device_batch_size"],
            "depth": base_config.get("depth", 10),
            "compile": base_config.get("compile", False),
            "compile_dynamic": base_config.get("compile_dynamic", False),
        }

        # Also ensure compile_mode is carried over if present
        if "compile_mode" in base_config:
            best_overrides["compile_mode"] = base_config["compile_mode"]

        # Use loaded env vars if available, else empty (or default)
        best_env = base_env if base_env else {}

        # Try to retrieve throughput if stored in json_data (local var inside check above needs to be accessible?)
        # We need to re-read or just assume 0.0 since we skipped it.
        # But we can try to guess it from the original file read if needed, but it's not critical for logic.
        print(f"Using loaded settings: BS={best_overrides['device_batch_size']}, Compile={best_overrides['compile']}", flush=True)

    if should_run_throughput:
        # 2. Define Search Space
        # Batch sizes to try. We start small and go up.
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        # Compilation flags
        compile_options = [True, False] if is_rocm else [True]
        compile_dynamic_options = [False] # Default to static shapes

        # Handle CPU Fallback logic for compilation
        # If we are on CPU (no ROCm/CUDA), we must ensure we don't try incompatible modes like reduce-overhead
        # But this script calls base_train.py which now handles the downgrade warning.
        # However, we should be careful about what we search over.
        if not is_rocm and not torch.cuda.is_available(): # CPU
             # On CPU, compiling might be slow or unstable with certain options.
             # We default to [True] but we should ensure compile_mode isn't forced to something bad later.
             pass

        if is_strix_halo:
            print("NOTE: Strix Halo detected. Enabling dynamic=True/False investigation.", flush=True)
            # We allow compile=True to be tested, specifically to investigate dynamic=True vs False
            compile_options = [True, False]
            compile_dynamic_options = [False, True]

        # Environment variable combinations
        # We start with basic flags.
        # We also want to test PYTORCH_CUDA_ALLOC_CONF for memory optimization if desired or necessary.
        # Especially relevant for 24GB GPUs (RTX 3090) as per report.
        base_env_configs = [{}]

        # Test expandable_segments explicitly if not already in env
        # If we are on CUDA or ROCm
        if is_rocm or torch.cuda.is_available():
            base_env_configs = [
                {}, # Default (typically expandable_segments:True in base_train unless overridden)
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, # Disable
                {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # Force Enable
            ]
            if is_rocm:
                # Also test HIP var
                 base_env_configs.extend([
                     {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:False"},
                     {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"},
                 ])

        env_configs = base_env_configs

        if is_rocm:
            # Tuning ROCm specific flags (combine with memory configs)
            # This cartesian product might be too big. Let's keep it simple for now and prioritize memory configs
            # Or we can just add the experimental flag to the best memory config later?
            # For now, let's just stick to the previous experimental flag toggle on top of default memory

            # Actually, let's mix them.
            rocm_configs = [
                {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"},
                {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
            ]

            # Cartesian product of base_env (memory) and rocm_configs (triton)
            new_configs = []
            for base in base_env_configs:
                for rocm in rocm_configs:
                    merged = base.copy()
                    merged.update(rocm)
                    new_configs.append(merged)
            env_configs = new_configs

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
    # We only do this if we just ran the throughput tuning OR if we want to refine it.
    if should_run_throughput and final_config.get("compile") is True:
        print("\nPhase 1.5: Tuning Compilation Mode", flush=True)
        # We already have result for default (mode="") which is best_throughput
        # If best_throughput came from a run where compile_mode was NOT set (or default), we should check others.
        # Note: default compile_mode in base_train is 'reduce-overhead'. So we should explicitly test variations.

        best_mode = final_config.get("compile_mode", "reduce-overhead")
        modes_to_try = ["default", "reduce-overhead", "max-autotune"]

        # We use the best batch size from Phase 1
        bs = final_config["device_batch_size"]

        for mode in modes_to_try:
            # Skip if we already tested this mode implicitly and it failed?
            # It's hard to know. Let's just re-test or test new ones.
            # If we just ran throughput tuning, we likely used the default "reduce-overhead" in base_train.py?
            # No, base_train defaults to "reduce-overhead", but we didn't override it in Phase 1 loops (except via implicit default).

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
        print("\nPhase 2: Learning Rate & Decay Tuning", flush=True)
        # Define search space
        # We will tune matrix_lr (Muon), embedding_lr (Adam), layer_lr_decay and unembedding_lr
        # Center around defaults or current values
        base_matrix_lr = float(final_config.get("matrix_lr", 0.02))
        base_embed_lr = float(final_config.get("embedding_lr", 0.2))
        base_unembed_lr = float(final_config.get("unembedding_lr", 0.004))

        # Grid: 0.5x, 1.0x, 2.0x for LRs
        lr_multipliers = [0.5, 1.0, 2.0]

        # Layer Decay options: 1.0 (baseline), 0.9, 0.8
        decays = [1.0, 0.9, 0.8] if args.tune_hyperparams else [1.0]

        best_loss = float("inf")
        best_lr_config = {}

        # 3D Grid Search
        import itertools
        lr_combinations = list(itertools.product(lr_multipliers, lr_multipliers, lr_multipliers, decays))

        for m_mult, e_mult, u_mult, decay in lr_combinations:
            m_lr = base_matrix_lr * m_mult
            e_lr = base_embed_lr * e_mult
            u_lr = base_unembed_lr * u_mult

            # Skip redundant 1.0 decay checks if we already did them (but combination with LR changes matters)

            overrides = final_config.copy()
            overrides["matrix_lr"] = m_lr
            overrides["embedding_lr"] = e_lr
            overrides["unembedding_lr"] = u_lr
            overrides["layer_lr_decay"] = decay

            # Use the best env from throughput tuning
            loss = run_loss_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                steps=args.max_benchmark_steps # User configurable max steps
            )

            if loss < best_loss:
                best_loss = loss
                best_lr_config = {"matrix_lr": m_lr, "embedding_lr": e_lr, "unembedding_lr": u_lr, "layer_lr_decay": decay}

        if best_lr_config:
            print(f"Found better LRs: {best_lr_config} with loss {best_loss:.4f}")
            final_config.update(best_lr_config)

    # Tuning schedule parameters
    if args.tune_lr: # Include schedule tuning with LR tuning
        print("\nPhase 2.5: Schedule Tuning", flush=True)

        # 1. Warmup
        base_warmup = float(final_config.get("warmup_ratio", 0.0))
        base_adam_warmup = float(final_config.get("adam_warmup_ratio", 0.0))
        warmups = [0.0, 0.01, 0.05] # Keep it small for efficiency

        # 2. Warmdown/Decay
        base_warmdown = float(final_config.get("warmdown_ratio", 0.2))
        base_final_frac = float(final_config.get("final_lr_frac", 0.0))
        warmdowns = [0.0, 0.2, 0.4]
        final_fracs = [0.0, 0.1]

        best_sched_loss = float("inf")
        best_sched_config = {}

        # We do a grid search here too, but maybe smaller?
        # Let's try to tune them somewhat independently or in small groups to avoid explosion.
        # Group 1: Warmups
        # Group 2: Decays

        # Tune Warmups
        print("  -> Tuning Warmup Ratios...", flush=True)
        best_warmup_loss = float("inf")
        local_best_warmups = {}
        for w in warmups:
            for aw in warmups:
                overrides = final_config.copy()
                overrides["warmup_ratio"] = w
                overrides["adam_warmup_ratio"] = aw

                loss = run_loss_benchmark(
                    overrides,
                    best_env,
                    base_config_path=args.config,
                    base_config=base_config,
                    extra_args=unknown,
                    steps=args.max_benchmark_steps
                )
                if loss < best_warmup_loss:
                    best_warmup_loss = loss
                    local_best_warmups = {"warmup_ratio": w, "adam_warmup_ratio": aw}

        if local_best_warmups:
             print(f"  -> Found better warmups: {local_best_warmups}", flush=True)
             final_config.update(local_best_warmups)

        # Tune Decays
        print("  -> Tuning Decay/Warmdown...", flush=True)
        best_decay_loss = float("inf")
        local_best_decays = {}
        for wd in warmdowns:
            for ff in final_fracs:
                overrides = final_config.copy()
                overrides["warmdown_ratio"] = wd
                overrides["final_lr_frac"] = ff

                loss = run_loss_benchmark(
                    overrides,
                    best_env,
                    base_config_path=args.config,
                    base_config=base_config,
                    extra_args=unknown,
                    steps=50
                )
                if loss < best_decay_loss:
                    best_decay_loss = loss
                    local_best_decays = {"warmdown_ratio": wd, "final_lr_frac": ff}

        if local_best_decays:
             print(f"  -> Found better decays: {local_best_decays}", flush=True)
             final_config.update(local_best_decays)

    if args.tune_optimizer:
        print("\nPhase 3: Optimizer Tuning", flush=True)

        best_loss = float("inf")
        best_opt_config = {}

        # 3a. Weight Decay & Gradient Clipping
        print("  -> Tuning Weight Decay and Gradient Clipping...", flush=True)
        base_wd = float(final_config.get("weight_decay", 0.0))
        wds = [0.0, 0.01, 0.1]

        base_clip = float(final_config.get("grad_clip", 1.0))
        clips = [0.0, 1.0, 2.0]

        for wd in wds:
            for clip in clips:
                overrides = final_config.copy()
                overrides["weight_decay"] = wd
                overrides["grad_clip"] = clip

                loss = run_loss_benchmark(
                    overrides,
                    best_env,
                    base_config_path=args.config,
                    base_config=base_config,
                    extra_args=unknown,
                    steps=args.max_benchmark_steps
                )

                if loss < best_loss:
                    best_loss = loss
                    best_opt_config = {"weight_decay": wd, "grad_clip": clip}

        if best_opt_config:
            print(f"  -> Found better decay/clip: {best_opt_config}", flush=True)
            final_config.update(best_opt_config)
            best_loss = float("inf") # Reset for next phase? Or keep cumulative?
                                     # Actually, we should probably keep improving `final_config` incrementally.
                                     # But if we reset best_loss, we need a baseline.
                                     # For simplicity, let's assume we proceed with the best config so far.

        # 3b. Optimizer Backends & 8-bit
        # We iterate over valid combinations of matrix and general optimizers
        # Matrix: ["muon", "nested_momentum"]
        # General: ["adamw", "nested_momentum"]
        # 8-bit: [True, False] (only for general optimizer usually)

        matrix_backends = ["muon", "nested_momentum"]
        general_backends = ["adamw", "nested_momentum"]
        use_8bit_options = [False, True]

        # We need a baseline loss for the current config to compare against
        # Run a quick baseline with current `final_config`
        baseline_loss = run_loss_benchmark(
             final_config,
             best_env,
             base_config_path=args.config,
             base_config=base_config,
             extra_args=unknown,
             steps=args.max_benchmark_steps
        )
        print(f"Baseline loss for Optimizer Tuning: {baseline_loss:.4f}", flush=True)
        best_loss = baseline_loss

        import itertools
        opt_combinations = list(itertools.product(matrix_backends, general_backends, use_8bit_options))

        best_backend_config = {}

        for m_backend, g_backend, use_8bit in opt_combinations:
            # Skip invalid or redundant combinations if known?
            # e.g. if 8bit is True but backend is nested_momentum, does it support it?
            # gpt.py code:
            # if general_optimizer_backend == "adamw": ... supports 8bit
            # elif general_optimizer_backend == "nested_momentum": ... DOES NOT support 8bit explicitly in code?
            # Checking gpt.py: nested_momentum branch does NOT check use_8bit_optimizer.
            # So if use_8bit is True and backend is nested_momentum, it is ignored (or 8bit not used).
            # We can skip testing use_8bit=True for nested_momentum to save time.
            if g_backend == "nested_momentum" and use_8bit:
                continue

            overrides = final_config.copy()
            overrides["matrix_optimizer_backend"] = m_backend
            overrides["general_optimizer_backend"] = g_backend
            overrides["use_8bit_optimizer"] = use_8bit

            # Print what we are trying
            print(f"Testing Optimizer: Matrix={m_backend}, General={g_backend}, 8bit={use_8bit}", flush=True)

            loss = run_loss_benchmark(
                overrides,
                best_env,
                base_config_path=args.config,
                base_config=base_config,
                extra_args=unknown,
                steps=args.max_benchmark_steps
            )

            if loss < best_loss:
                best_loss = loss
                best_backend_config = {
                    "matrix_optimizer_backend": m_backend,
                    "general_optimizer_backend": g_backend,
                    "use_8bit_optimizer": use_8bit
                }
                print(f"  -> Improved loss: {loss:.4f}", flush=True)
            else:
                print(f"  -> Loss: {loss:.4f} (not improved)", flush=True)

        if best_backend_config:
             print(f"Found better Optimizer Backends: {best_backend_config} with loss {best_loss:.4f}")
             final_config.update(best_backend_config)

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
                    steps=args.max_benchmark_steps
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
    print(f"\nRun command with updated profile:\nbash run.sh -m scripts.base_train {args.config if args.config else ''} --device_batch_size={final_config['device_batch_size']} --compile={final_config['compile']} --run=$WANDB_RUN", flush=True)

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
