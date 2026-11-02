
## Archived: 2026-11-02

* The `save_checkpoint` function in `nanochat/checkpoint_manager.py` ensures directory existence on all ranks to prevent race conditions and utilizes atomic writes (writing to a temporary file then renaming) for model artifacts.
* The `forward_model` function in `nanochat/core_eval.py` checks for and unpacks tuple outputs (e.g., `(logits, _)` from `model(prompt_ids)` to accommodate the auxiliary action predictions returned by `GPT.forward` in inference mode.
* The project supports 8-bit AdamW optimization via `bitsandbytes`, enabled by the `--use_8bit_optimizer` flag in `scripts/base_train.py`.
* The `nanochat/scheduler` package (specifically `cosine.py`) defines `get_lr_multiplier` and `get_muon_momentum`. These are pure functions requiring explicit state arguments (e.g., `num_iterations`) to avoid global scope dependencies.
* `scripts/base_eval.py` downloads and sets up the evaluation bundle (`eval_bundle`) by clearing any existing directory and extracting the downloaded zip file directly to the project base, avoiding the use of temporary directories.
* `scripts/tune_system.py` detects AMD Strix Halo hardware and explicitly enables `compile=True` testing with `compile_dynamic` options (`True` and `False`) to facilitate performance investigations, modifying previous stability restrictions.
* The GGUF export implementation in `scripts/export_model.py` uses a custom `"nanochat"` architecture tag, making it experimental and potentially incompatible with standard `llama.cpp` builds.
* The project supports ROCm 7.10, which corresponds to `torch==2.9.1+rocm7.10.0`.
* Tests verifying device detection logic (`nanochat.common.autodetect_device_type`) must explicitly patch `torch.version.hip` to `None` when asserting non-ROCm outcomes, as the attribute exists on ROCm builds even when no GPU is available.
* Switching optimizers (e.g., from AdamW to NestedMomentum) when loading a checkpoint works for model weights but results in a cold start for the optimizer state (resetting momentum buffers).
* `run.sh` explicitly exports `.venv/bin` to `PATH` and attempts to sync AMD dependencies before hardware detection to ensure `rocminfo` is available and discoverable.
* The `compute_cleanup` function in `nanochat/common.py` verifies if a process group is initialized via `is_ddp_initialized()` before attempting to destroy it, preventing `AssertionError` during script termination.
* `PyYAML` is a required dependency for evaluation scripts such as `scripts/base_eval.py`.
* `GPT.init_weights` initializes `lm_head` and `c_proj` weights to zero; this prevents gradient flow to LoRA layers during testing unless these frozen base weights are manually re-initialized to non-zero values.
* The `nanochat-rs` directory contains a dedicated `pyproject.toml` for building the `nanochat_rs` extension using `maturin`.
* `scripts/base_train.py` defines global variables for LoRA (`use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_targets` [default: `["q_proj", "v_proj"]`]) to enable CLI configuration, with `lora_targets` defaulting to `["q_proj", "v_proj"]`.
* `nanochat/checkpoint_manager.py` logic checks `isinstance(v, torch.Tensor)` before converting `bfloat16` types to `float` for CPU/MPS inference, protecting against errors when model state dictionaries contain non-tensor metadata.
* The default git branch for the repository is `master`.
* The `GPT.setup_optimizers` method scales learning rates for general parameters (embeddings, head, matrix) based on the square root of their specific dimensions.
* `scripts/chat_sft.py` performs Supervised Fine-Tuning (SFT) on the chat model. It explicitly requires `targets` to be passed as a keyword argument (e.g., `model(inputs, targets=targets)`) to trigger the correct training path and return a scalar loss.
* The `GPT.forward` method in `nanochat/gpt.py` interprets the second positional argument as `images`. To perform training (returning scalar loss), `targets` must be passed as a keyword argument (e.g., `model(x, targets=y)`); passing it positionally (e.g., `model(x, y)`) triggers inference mode and returns a tuple `(logits, action_pred)`. The `GPT.forward` method was updated to avoid calculating full `logits` when `targets` are present.
* `scripts/base_train.py` invokes `torch.compiler.cudagraph_mark_step_begin()` (if available) *outside* the gradient accumulation loop (before `for micro_step in range...`) to ensure correct allocator reset behavior when using `compile_mode='reduce-overhead'`.
* The `GPT.forward` method accepts an optional `kv_cache` argument (instance of `nanochat.engine.KVCache`) to support cached inference.
* The project dependencies in `pyproject.toml` now include `lpips` and `timm` for visual tokenizer training and perceptual loss calculation.
* When `general_optimizer_backend='adamw'` (default), the `GPT` model uses `torch.distributed.optim.ZeroRedundancyOptimizer` with fused `AdamW` in distributed environments, or a standard fused `AdamW` otherwise.
* `nanochat/configurator.py` implements a `print0` function to restrict configuration logging to the master process (Rank 0) in distributed environments, preventing console spam.
* The project utilizes a Rust-based tokenizer extension (`rustbpe`) which requires compilation via `maturin`. To preserve hardware-specific dependencies (like ROCm), the command must include the active extras: `uv run --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml`.
* `scripts/base_train.py` sets `TORCH_BLAS_PREFER_HIPBLASLT=0` when Strix Halo (gfx1151) is detected to prevent `HIPBLAS_STATUS_NOT_SUPPORTED` warnings and fallback issues.
* The `GPT` model calculates an auxiliary `action_loss` using either MSE (for regression) or diffusion loss (predicting noise) based on the `RoboticsConfig`, allowing the model to generate latent control vectors.
* `scripts/workflow.py` defines `eval_args` for the "tiny" job without an explicit `--` separator (e.g., `["--max-per-task=16"]`) to ensure correct argument parsing in `scripts/base_eval.py`.
* The `load_model` function in `nanochat/checkpoint_manager.py` accepts optional `model_tag` and `step` arguments to allow loading a specific checkpoint version instead of defaulting to the latest.
* `nanochat/common.py` distinguishes between `is_ddp_requested()` (checks env vars like `RANK`) and `is_ddp_initialized()` (checks `dist.is_initialized()`) to ensure robust initialization and cleanup.
* `scripts/tune_system.py` conditionally enables verbose PyTorch compilation logging (`TORCH_COMPILE_DEBUG=1`, `TORCH_LOGS="+dynamo"`) specifically when Strix Halo hardware is detected to aid in debugging hangs.
* `GPT.forward` is optimized to skip the computationally expensive full logits calculation (`self.lm_head(x)`) during training (when `targets` are provided), relying instead on `chunked_cross_entropy` for loss computation.
* `nanochat/engine.py` uses an independent sampling strategy for each batch element during generation to prevent token broadcasting issues, while supporting multimodal tuple outputs.
* The `GPT` model automatically pads the vocabulary size (e.g. to be divisible by 64) in `__init__`, but `forward` slices the logits back to the original `config.vocab_size` during inference.
* The `rustbpe` tokenizer extension utilizes the `rayon` crate to support parallel execution.
* `scripts/serve.py` uses environment variables `NANOCHAT_PORT`, `NANOCHAT_HOST`, `NANOCHAT_NUM_GPUS`, `NANOCHAT_SOURCE` (default 'sft'), `NANOCHAT_MODEL_TAG`, `NANOCHAT_STEP`, `NANOCHAT_DEVICE_TYPE`, `NANOCHAT_DTYPE`, and `NANOCHAT_MOCK` for configuration.
* The `torchrun` helper in `scripts/workflow.py` accepts an `include_config` argument (defaulting to `True`) to conditionally pass the configuration file, as evaluation scripts (`scripts.base_eval`, `scripts.chat_eval`) do not accept it as a positional argument.
* The `chunked_cross_entropy` function is decorated with `@torch.compiler.disable` to force eager execution and serve as a graph break.
* The `tests/test_workflow.py` test suite mocks `argparse` namespace objects including the `skip_workarounds` attribute to align with the current `scripts/workflow.py` implementation.
* `scripts/workflow.py` defaults `NANOCHAT_BASE_DIR` to `~/.cache/nanochat` and sets `OMP_NUM_THREADS` to "1" if undefined.
* `scripts/tune_system.py` enforces increased timeouts (3600s for throughput, 7200s for loss benchmarks) to accommodate slow compilation times on hardware like Strix Halo.
* In `scripts/base_train.py`, the `adam_warmup_ratio` parameter specifically controls the warmup duration for the AdamW optimizer (general parameters), allowing the warmup to be configured independently from the Muon optimizer's `warmup_ratio`.
* The project now maintains a `.env.example` file in the root directory documenting all supported environment variables.
* `pyproject.toml` includes the `kernels` dependency to facilitate experimental Flash Attention 3 support, which is loaded dynamically via `kernels.get_kernel` from the Hugging Face Hub.
* `scripts/tune_system.py` accepts a `--try-all-variations` flag to force a full throughput grid search, overriding the default behavior of using existing configuration settings.
* `scripts/base_train.py` sets `PYTORCH_ALLOC_CONF` (and `PYTORCH_HIP_ALLOC_CONF` on ROCm) to `expandable_segments:True` only if the variable is not already defined in the environment, allowing external overrides.
* To resolve `RuntimeError` with `compile_mode='reduce-overhead'` on Strix Halo, `chunked_cross_entropy` uses `@torch.compiler.disable`, and `GPT.forward` explicitly clones both `x` and `targets` before passing them to the loss function to prevent graph boundary leaks.
* `nanochat/gpt.py` attempts to load Flash Attention 3 and falls back to `torch.nn.functional.scaled_dot_product_attention` if the import fails or dependencies (like CUDA runtime libraries) are missing.
* When Flash Attention 3 is active in `CausalSelfAttention.forward`, the code maintains the `(Batch, Seq, Head, Dim)` tensor layout, skipping the transpose to `(Batch, Head, Seq, Dim)` used for standard SDPA.
* `scripts/base_eval.py` accepts `--model-tag` and `--step` arguments to select specific checkpoints for evaluation.
* `scripts/base_eval.py` and `scripts/chat_eval.py` utilize `argparse.ArgumentParser.parse_known_args()` to safely ignore configuration file arguments and separators passed by `workflow.py`, preventing `unrecognized arguments` errors.
