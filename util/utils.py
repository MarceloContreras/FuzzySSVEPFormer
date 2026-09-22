import numpy as np
import torch
import os
import random
import re

from pathlib import Path
from collections import defaultdict


def seed_everything(seed=42):
    """
    Function to put a seed to every step and make code reproducible
    Input:
    - seed: random state for the events
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_checkpoint(folder, modelo, ventana, dataset, sujeto):
    folder = Path(folder)

    patron = re.compile(
        rf"^{re.escape(modelo)}_"
        rf"{re.escape(str(ventana))}s_"
        rf"{re.escape(dataset)}_"
        rf"S{sujeto}(?:\.|_|$)"
    )

    matches = [
        archivo for archivo in folder.glob("*.pth") if patron.match(archivo.name)
    ]

    return sorted(matches)
