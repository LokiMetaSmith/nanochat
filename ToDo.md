# Remaining Tasks & Roadmap

## 🤖 NanoBot (Robotics & Multimodal)
- [x] **Vision Integration**: Implement lightweight Vision Transformer and Projector (`nanochat/vision.py`).
- [x] **Robotics Interface**: Implement Sensor and Surface projectors (`nanochat/robotics.py`).
- [x] **Action Generation**: Implement **Diffusion Policy Head** (`nanochat/diffusion.py`) for continuous control.
- [x] **Inference Kernel (Rust)**: Create `nanochat-rs` using HuggingFace Candle for on-robot deployment.
- [x] **Model Export**: Update `scripts/export_model.py` to embed multimodal config metadata for the Rust kernel.
- [x] **Real Data Pipeline**:
    - Update `nanochat/dataloader.py` to load real images (Parquet bytes/paths) and sensor logs instead of synthetic noise.
    - Create a data collector script to stream telemetry from ESP32 to Parquet (`scripts/collect_telemetry.py`).
- [x] **Continual Learning Loop**: Connect the training loop for online fine-tuning via `continual=True` and data polling.

## 🚀 Optimization & Strix Halo Specifics
- [x] **MXFP4 Investigation**: Research and implement OCP Microscaling (MXFP4) support for inference using AMD Quark, once the ecosystem matures for APUs.
- [x] **System Tuner Expansion**: Enhance `scripts/tune_system.py` to auto-tune:
    - [x] Learning rates and schedules.
    - [x] Optimizer hyperparameters (momentum, weight decay) and backends (Muon vs NestedMomentum).
    - [x] Compilation flags (`torch.compile` modes: default, reduce-overhead, max-autotune).
- [x] **LoRA Support**: Implement Low-Rank Adaptation (`nanochat/lora.py`) and integrate with `tune_system.py`.
- [x] **Torch Compile Dynamics**: Investigate `dynamic=True` vs `False` in `scripts/base_train.py` for variable sequence lengths on RDNA 3.5. (Implemented support for `dynamic` flag in `base_train.py` and `tune_system.py` to facilitate investigation).
- [ ] **Distributed Tuning**: Benchmark RCCL vs Gloo backends specifically for APU-based distributed setups (if scaling to multi-node APUs).

## 🛠 Codebase Maintenance & Tech Debt
- [x] **DDP Detection**: Refactor `is_ddp()` in `nanochat/common.py` to use a more robust detection method.
- [x] **Tokenizer Efficiency**: Optimize `prepend_id` insertion in `nanochat/tokenizer.py` (currently uses `list.insert(0)`, which is O(N)).
- [x] **Liger Kernels / Memory**: Implemented **Chunked Cross Entropy** in `nanochat/gpt.py` to reduce memory usage. (Note: Liger Kernels were not used, manual chunking was preferred for custom softcap support).
- [ ] **Checkpointing**:
    - [x] Fix potentially redundant model re-initialization in `checkpoint_manager.py`.
    - [x] Ensure optimizer state saving across ranks is robust (`scripts/base_train.py`).
- [x] **Evaluation Cleanup**: Refactor `scripts/base_eval.py` to remove heavy dependencies (like pandas) and simplify file handling.
- [x] AdamW Warmup: Verified independent warmup schedule for AdamW parameters; enabled via `adam_warmup_ratio`.

## ✨ New Features
- [x] **Model Export**:
    - Add a script to export checkpoints to **GGUF** format for efficient inference on Strix Halo NPU (via llama.cpp).
    - Add HuggingFace `safetensors` export support.
- [x] **Inference Server**: Create a production-ready API server (FastAPI) to serve the model, replacing the simple `chat_cli.py`.
- [ ] **RLHF Expansion**: Extend Reinforcement Learning (RL) support beyond the current GSM8K-only implementation.
- [ ] **Advanced UI**: Develop a more robust chat interface (React/Web) or integrate with existing open-source UIs (e.g., Open WebUI).
- [ ] **Data Pipeline**:
    - [x] Add data integrity verification for downloaded shards.
    - Optimize data loading for APU unified memory architectures.
