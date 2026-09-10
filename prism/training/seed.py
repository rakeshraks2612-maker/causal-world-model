"""PRISM Reproducibility and Deterministic RNG Manager."""

from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global seeds across Python, NumPy, and PyTorch for deterministic execution.
    
    Args:
        seed: Integer seed value
        deterministic: If True, configures PyTorch cuDNN / backend for determinism
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
