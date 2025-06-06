import torch
import logging

log = logging.getLogger(__name__)

def print_gpu_usage(note=""):
    if torch.cuda.is_available():
        log.info(f"[GPU] {note} | Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | "
              f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB | "
              f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")