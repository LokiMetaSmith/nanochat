from collections import deque
import time
import io
import os

import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def process_image(image_bytes, target_size, device):
    """
    Decode, resize, and normalize image bytes.
    Returns tensor (C, H, W).
    """
    try:
        if image_bytes is None:
            return torch.zeros((3, target_size, target_size), device=device)

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((target_size, target_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
        return img_tensor
    except Exception as e:
        print(f"Error processing image: {e}")
        return torch.zeros((3, target_size, target_size), device=device)

def tokenizing_distributed_data_loader_with_state(
    B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda",
    resume_state_dict=None, vision_config=None, robotics_config=None, continual=False
):
    """
    Stream pretraining text and multimodal data from parquet files.

    Args:
        continual (bool): If True, the loader will poll for new files instead of terminating.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    device = str(device) # ensure string

    # infinite iterator over document batches
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx

        # State for continual learning
        known_files = set()

        while True: # iterate infinitely (multi-epoch or continual)
            # Refresh file list
            all_paths = list_parquet_files()
            if continual:
                # In continual mode, we only process files we haven't finished yet or are new
                pass

            # Simple logic: just reload list. If static, it loops over same files.
            # If dynamic, new files appear at the end.

            # Filter for split
            if split == "train":
                parquet_paths = all_paths[:-1] if len(all_paths) > 1 else all_paths
            else:
                parquet_paths = all_paths[-1:] if len(all_paths) > 0 else []

            if not parquet_paths and continual:
                print("No data found, waiting...")
                time.sleep(5)
                continue

            # If we finished all known files, reset to 0 or wait?
            # Standard training: reset to 0 (Epochs).
            # Continual: wait for new file at index `len(parquet_paths)`.
            if pq_idx >= len(parquet_paths):
                if continual:
                    print(f"Caught up with data stream (processed {pq_idx} files). Waiting for new data...")
                    time.sleep(1) # poll every second
                    continue
                else:
                    pq_idx = 0 # Loop back for standard training

            while pq_idx < len(parquet_paths):
                filepath = parquet_paths[pq_idx]
                try:
                    pf = pq.ParquetFile(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}. Skipping.")
                    pq_idx += 1
                    continue

                # Resume logic
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank

                while rg_idx < pf.num_row_groups:
                    try:
                        rg = pf.read_row_group(rg_idx)
                        batch = rg.to_pylist() # List of dicts

                        # Process in chunks
                        for i in range(0, len(batch), tokenizer_batch_size):
                            yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)

                    except Exception as e:
                        print(f"Error reading row group {rg_idx} in {filepath}: {e}")

                    rg_idx += ddp_world_size

                pq_idx += 1
            first_pass = False

    batches = document_batches()

    # Tokenizer setup
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    needed_tokens = B * T + 1

    # Buffers
    token_buffer = deque()
    # Multimodal buffers
    image_buffer = deque()

    use_cuda_optimizations = "cuda" in str(device)

    while True:
        # Fill buffers
        while len(token_buffer) < needed_tokens:
            try:
                batch_data, (pq_idx, rg_idx) = next(batches)
            except StopIteration:
                break

            # Extract columns
            texts = [item.get('text', '') for item in batch_data]

            # Tokenize text
            token_lists = tokenizer.encode(texts, prepend=bos_token, num_threads=tokenizer_threads)

            for idx, tokens in enumerate(token_lists):
                token_buffer.extend(tokens)

                item = batch_data[idx]
                img_bytes = item.get('image', None)
                sens = item.get('sensors', None)
                surf = item.get('surface', None)

                meta_obj = {
                    'image': img_bytes,
                    'sensors': sens,
                    'surface': surf
                }

                for _ in range(len(tokens)):
                    image_buffer.append(meta_obj)

        tokens_to_pop = B * T + 1

        # Extract atoms
        batch_tokens = []
        batch_meta_list = []
        for _ in range(tokens_to_pop):
            batch_tokens.append(token_buffer.popleft())
            batch_meta_list.append(image_buffer.popleft())

        # Create Tensors
        scratch = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        # Reconstruct Multimodal Batches
        b_images = []
        b_sensors = []
        b_surface = []

        for b in range(B):
            idx = b * T
            if idx >= len(batch_meta_list): break

            meta = batch_meta_list[idx]

            # Process Image
            if vision_config:
                img = process_image(meta['image'], vision_config['image_size'], device)
                b_images.append(img)

            # Process Sensors
            if robotics_config:
                s_dim = robotics_config.sensor_dim
                sens = meta['sensors']
                if sens is not None:
                    sens_t = torch.tensor(sens, dtype=torch.float32, device=device)
                else:
                    sens_t = torch.zeros(s_dim, dtype=torch.float32, device=device)
                b_sensors.append(sens_t)

                sf_dim = robotics_config.surface_dim
                surf = meta['surface']
                if surf is not None:
                    surf_t = torch.tensor(surf, dtype=torch.float32, device=device)
                else:
                    surf_t = torch.zeros(sf_dim, dtype=torch.float32, device=device)
                b_surface.append(surf_t)

        final_images = torch.stack(b_images) if b_images else None
        final_sensors = torch.stack(b_sensors) if b_sensors else None
        final_surface = torch.stack(b_surface) if b_surface else None

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}

        if robotics_config:
             yield inputs, targets, final_images, final_sensors, final_surface, state_dict
        elif vision_config:
             yield inputs, targets, final_images, state_dict
        else:
             yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, device="cuda", **kwargs):
    # helper wrapper
    kwargs["device"] = device

    loader = tokenizing_distributed_data_loader_with_state(*args, **kwargs)

    for item in loader:
        # Legacy consumers (like base_eval.py) expect ONLY (inputs, targets)
        # Multimodal consumers (like base_train.py) handle unpacking
        # We check the length of item to decide.
        # But this wrapper is a generator, we can't look ahead at consumer.

        # FIX: Always return (x, y) ONLY from this wrapper to preserve backward compatibility.
        # Training scripts should call `tokenizing_distributed_data_loader_with_state` directly if they need extras.
        # `base_train.py` already uses `with_state` for the training loop.
        # `base_train.py` uses this wrapper for `val_loader`.

        # If we return just (x, y), val loop works (metrics are text-only).
        yield item[0], item[1]
