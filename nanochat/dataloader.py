from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, vision_config=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    Now supports multimodal training (vision) if `vision_config` is provided.
    Currently, this generates SYNTHETIC images (random noise) to test the pipeline.
    In the future, this should load images from a parquet column or image folder.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left

    # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
    use_cuda_optimizations = device == "cuda" or (device and "cuda" in device)

    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        # Handle Vision (Generate Synthetic Images)
        images = None
        if vision_config:
            # Generate random noise images: (B, C, H, W)
            # In a real implementation, we would load images corresponding to the text
            img_h, img_w = vision_config['image_size'], vision_config['image_size']
            channels = vision_config.get('channels', 3)
            # Create on CPU first (to emulate loading), then move to device
            # or create directly on device if it's purely synthetic for test
            images = torch.randn(B, channels, img_h, img_w, device=device, dtype=torch.float32)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training

        if vision_config:
            yield inputs, targets, images, state_dict
        else:
            yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, device="cuda", **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    kwargs["device"] = device

    # Check if vision config is in kwargs
    vision_config = kwargs.get("vision_config", None)

    loader = tokenizing_distributed_data_loader_with_state(*args, **kwargs)

    for item in loader:
        if vision_config:
            # inputs, targets, images, state_dict
            yield item[0], item[1], item[2]
        else:
            # inputs, targets, state_dict
            yield item[0], item[1]
