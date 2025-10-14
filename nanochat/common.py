"""
Common utilities for nanochat.
"""

import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1


def is_macos():
    """Check if running on macOS."""
    import platform
    return platform.system() == "Darwin"


def get_device_type():
    """Get the device type string for autocast: 'cuda' or 'cpu'."""
    # Use CPU if on macOS or if CUDA is not available
    if is_macos() or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def resolve_autocast_dtype(device_type: str, requested_dtype: str | torch.dtype | None=None) -> torch.dtype:
    """Return a safe autocast dtype for the given device."""
    if isinstance(requested_dtype, torch.dtype):
        dtype = requested_dtype
    elif isinstance(requested_dtype, str):
        key = requested_dtype.lower()
        if key in ("bfloat16", "bf16"):
            dtype = torch.bfloat16
        elif key in ("float16", "fp16", "half"):
            dtype = torch.float16
        elif key in ("float32", "fp32"):
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype string: {requested_dtype}")
    elif requested_dtype is None:
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    else:
        raise TypeError(f"Unsupported dtype specifier type: {type(requested_dtype)!r}")

    if device_type != "cuda" and dtype != torch.float32:
        logger.warning(
            "Falling back to float32 autocast on %s (requested %s unsupported)",
            device_type,
            dtype,
        )
        dtype = torch.float32

    return dtype


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def compute_init():
    """Basic initialization that we keep doing over and over, so make common."""

    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    on_macos = is_macos()

    # Reproducibility
    torch.manual_seed(42)
    if has_cuda:
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True

    # Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        # Determine device
        if on_macos or not has_cuda:
            device = torch.device("cpu")
            if on_macos:
                logger.info("Running on macOS with CPU")
            else:
                logger.info("Running on CPU (CUDA not available)")
            logger.warning("DDP requested but will run on CPU")
        else: # has_cuda and not on_macos
            device = torch.device("cuda", ddp_local_rank)
            torch.cuda.set_device(device) # make "cuda" default to this device
            dist.init_process_group(backend="nccl", device_id=device)
            dist.barrier()
            logger.info(f"Running on CUDA with DDP (rank {ddp_rank}/{ddp_world_size})")
    else: # not ddp
        device = torch.device("cuda" if has_cuda and not on_macos else "cpu")
        logger.info(f"Running on {'CUDA (single GPU)' if has_cuda and not on_macos else 'CPU'}")

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
