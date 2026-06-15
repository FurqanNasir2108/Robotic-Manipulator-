"""
Random seed and reproducibility utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic but potentially slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)