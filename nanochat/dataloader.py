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
                # But typically we just want to iterate over the sorted list.
                # If we reach the end, we wait.
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
    sensor_buffer = deque()
    surface_buffer = deque()
    action_buffer = deque()

    use_cuda_optimizations = "cuda" in str(device)

    while True:
        # Fill buffers
        while len(token_buffer) < needed_tokens:
            try:
                batch_data, (pq_idx, rg_idx) = next(batches)
            except StopIteration:
                # Should not happen in infinite loop, but safety break
                break

            # Extract columns
            texts = [item.get('text', '') for item in batch_data]

            # Tokenize text
            token_lists = tokenizer.encode(texts, prepend=bos_token, num_threads=tokenizer_threads)

            for idx, tokens in enumerate(token_lists):
                token_buffer.extend(tokens)

                # Handle Multimodal Alignment
                # Assuming 1-to-1 mapping between text document and image/sensor entry
                # We replicate the image/sensor data for *each token* in the document?
                # NO. The model expects (B, T) text, and (B, C, H, W) images.
                # Usually one image per sequence (document).
                #
                # Current Architecture: Prepend visual tokens to the WHOLE sequence.
                # So we need 1 image for every `T` tokens (roughly).
                #
                # Challenge: `token_buffer` is a continuous stream of tokens.
                # `B * T` chunks are sliced from it.
                #
                # Strategy: We associate metadata with the BOS token of each document.
                # Or simplify: We queue up images/sensors and pop one for every Sequence Batch `B`.
                # This assumes documents are roughly `T` long or we are okay with mismatch.
                #
                # Better Strategy for "Continual Robot Data":
                # Each "row" in Parquet is a "Step". It contains:
                # - Text (Instruction or Log)
                # - Image (Current View)
                # - Sensor (Current State)
                # - Surface (Latent)
                #
                # We want to feed this as a sequence.
                # If we treat each Row as a sample, we can just batch Rows.
                # But `tokenizing_distributed_data_loader` is designed for continuous text streaming.
                #
                # Compromise:
                # We store the *raw* multimodal data in parallel deques.
                # When we pop `needed_tokens` from `token_buffer`, we also pop the corresponding
                # multimodal data associated with those tokens.
                #
                # Since multiple tokens come from one row, we duplicate the metadata?
                # Or we just take the metadata of the *first* document in the window.
                #
                # Let's go with: 1 Image/Sensor per BATCH element (Sequence).
                # We need to buffer at the document level, not token level?
                #
                # Let's simplify:
                # The `image_buffer` stores (image_bytes) for every DOCUMENT.
                # When we slice a sequence from `token_buffer`, we check which document it belongs to.
                #
                # Actually, for the NanoBot use case, "One Document = One Trajectory Step" is too short.
                # Likely "One Document = One Episode".
                #
                # Let's assume the dataloader yields (Text, Image) pairs.
                # We queue (TokenList, ImageBytes).
                # When we construct a batch of size B*T, we are combining tokens from multiple docs.
                # The model `forward` accepts `images` of shape (B, ...).
                # This implies 1 image per sequence in the batch.
                #
                # So we associate the image with the START of the token sequence.

                item = batch_data[idx]

                # Store data for this document
                # We repeat this data for the length of the tokens to ensure alignment?
                # No, we just store it in a parallel deque matching token length is too expensive.
                #
                # Implementation:
                # `token_buffer`: [t1, t2, t3, ... tN]
                # `meta_buffer`:  [M1, M1, M1, ... M1] (where M1 is image/sensor data)
                #
                # Then we slice.
                # `batch_meta` = meta_buffer[::T] (sample every T?)
                # Or just take the meta of the first token in the sequence.

                img_bytes = item.get('image', None)
                sens = item.get('sensors', None)
                surf = item.get('surface', None)

                # Optimization: Don't store full bytes for every token.
                # Store (doc_id, metadata) map and push doc_id to buffer.
                # But simple is better for now.
                # We will push metadata object to a deque for every token.
                # Python references are cheap.

                meta_obj = {
                    'image': img_bytes,
                    'sensors': sens,
                    'surface': surf
                }

                for _ in range(len(tokens)):
                    image_buffer.append(meta_obj)

        # Pop B sequences of length T+1
        # Actually we pop B * (T+1) tokens linearly?
        # Standard implementation pops `needed_tokens` which is B*T+1 (flat).
        # Then reshapes to (B, T).
        # This implies the batch is a single long stream broken into B rows?
        # Or B independent streams?
        #
        # `inputs_cpu = scratch[:-1].view(B, T)`
        # This means the buffer is treated as one long stream, wrapped around.
        # Row 0 is T tokens. Row 1 is NEXT T tokens.
        #
        # So for each Row b in B:
        #   We take T tokens.
        #   We need 1 Image for this Row.
        #   We take the image associated with the *first* token of this row.

        batch_tokens = []
        batch_images = []
        batch_sensors = []
        batch_surface = []

        # We need B rows. Each row has length T.
        # Plus 1 for targets offset.
        # We process sequence by sequence.

        # NOTE: The original logic popped `needed_tokens` (B*T+1) at once.
        # This effectively mixes the stream across batch rows if reshaped blindly.
        # `view(B, T)` on a flat list [0..B*T] puts [0..T] in row 0, [T..2T] in row 1.
        # This preserves locality.

        tokens_to_pop = B * T + 1

        # Extract atoms
        batch_meta_list = []
        for _ in range(tokens_to_pop):
            batch_tokens.append(token_buffer.popleft())
            batch_meta_list.append(image_buffer.popleft())

        # Create Tensors
        scratch = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        # Reconstruct Multimodal Batches
        # For each row b (0..B), we look at the metadata of the first token (index b*T).

        b_images = []
        b_sensors = []
        b_surface = []

        for b in range(B):
            idx = b * T
            if idx >= len(batch_meta_list): break # Should not happen

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

                # Surface / Action Targets
                # Assuming surface is also target for now
                sf_dim = robotics_config.surface_dim
                surf = meta['surface']
                if surf is not None:
                    surf_t = torch.tensor(surf, dtype=torch.float32, device=device)
                else:
                    surf_t = torch.zeros(sf_dim, dtype=torch.float32, device=device)
                b_surface.append(surf_t)

        # Stack multimodal tensors
        # Images: (B, C, H, W)
        final_images = torch.stack(b_images) if b_images else None

        # Sensors: (B, Dim)
        final_sensors = torch.stack(b_sensors) if b_sensors else None

        # Surface: (B, Dim)
        final_surface = torch.stack(b_surface) if b_surface else None

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx}

        # Yield based on config
        # Return signature: inputs, targets, [images], [sensors], [surface], [state]
        # To remain compatible with `base_train.py` unpacking:
        # if vision: x, y, images, state
        # if robot: x, y, images, sensors, surface, state

        if robotics_config:
             yield inputs, targets, final_images, final_sensors, final_surface, state_dict
        elif vision_config:
             yield inputs, targets, final_images, state_dict
        else:
             yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, device="cuda", **kwargs):
    # helper wrapper
    kwargs["device"] = device
    vision_config = kwargs.get("vision_config")
    robotics_config = kwargs.get("robotics_config")

    loader = tokenizing_distributed_data_loader_with_state(*args, **kwargs)

    for item in loader:
        # Strip state_dict for simple iteration if needed,
        # but base_train expects unpacking.
        # We just yield the tuple as is, minus state dict?
        # base_train expects: x, y, [extras], state
        yield item
