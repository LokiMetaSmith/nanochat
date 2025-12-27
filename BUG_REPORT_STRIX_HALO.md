# Bug Report: PyTorch CUDAGraphs Instability on AMD Strix Halo (gfx1151)

**Date:** 2025-12-26
**Hardware:** AMD Strix Halo APU (gfx1151)
**Software:** PyTorch 2.9.1+rocm7.10 (Nightly/Preview), ROCm 7.10

## Issue Summary
When training a GPT-style transformer model using `torch.compile(mode="reduce-overhead")` (which utilizes CUDAGraphs), the training loop crashes with a `RuntimeError` during the backward pass, or hangs indefinitely (timeout > 1200s). This occurs even with single-batch training (`device_batch_size=1`). Standard training (`compile=False` or `mode="default"`) functions correctly.

## Error Trace
The primary error indicates memory corruption or unsafe reuse of static CUDAGraph buffers:

```text
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
Stack trace:
  File ".../nanochat/gpt.py", line 864, in torch_dynamo_resume_in_forward_at_651
    x = block(x, cos_sin, kv_cache)
  ...
  File ".../nanochat/gpt.py", line 204, in forward
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

Additionally, `HIPBLAS_STATUS_NOT_SUPPORTED` warnings are emitted unless `TORCH_BLAS_PREFER_HIPBLASLT=0` is set:
```text
UserWarning: bgemm_internal_cublaslt error: HIPBLAS_STATUS_NOT_SUPPORTED when calling hipblasLtMatmul ...
```

## Attempted Mitigations (Failed)
The following standard CUDAGraphs fixes were implemented but did **not** resolve the issue (hanging/crashing persisted):

1.  **Explicit Tensor Cloning:** Both the model output `x` and the `targets` tensor were explicitly cloned (`.clone()`) before being passed to the loss function.
2.  **Graph Breaks:** The loss function (`chunked_cross_entropy`) was decorated with `@torch.compiler.disable` to force eager execution and isolate it from the compiled graph.
3.  **Step Marker Placement:** `torch.compiler.cudagraph_mark_step_begin()` was moved outside the gradient accumulation micro-step loop to ensure the memory allocator resets correctly per logical step.
4.  **Environment Variables:**
    *   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (Tried both 0 and 1)
    *   `TORCH_BLAS_PREFER_HIPBLASLT=0` (Silenced warnings, didn't fix crash)
    *   `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (and False)

## Reproduction
Run the system tuner or base training script on a Strix Halo machine:

```bash
# Crash/Hang Reproduction
bash run.sh -m scripts.base_train \
    --device_batch_size=1 \
    --depth=20 \
    --compile=True \
    --compile_mode=reduce-overhead \
    --eval_tokens=2048 \
    --num_iterations=10
```

## Workaround
The only stable configuration found is to **disable CUDAGraphs** entirely on this platform:
*   `compile=False`
*   OR `compile_mode="default"` / `compile_mode="max-autotune"` (without reduce-overhead)

## Recommendation for Fix
This appears to be a driver/compiler-level issue with memory management for CUDAGraphs on the `gfx1151` architecture in the current ROCm preview. The `RuntimeError` suggests that the static memory addresses assigned to intermediate tensors are being aggressively reused or overwritten by the next graph replay before the previous backward pass is complete, or that the "safe" cloning mechanism is failing to actually decouple the memory on this specific APU unified memory architecture.
