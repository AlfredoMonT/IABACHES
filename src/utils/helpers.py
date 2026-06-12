import os
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_label_for_image(img_path, label_dirs):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ld in label_dirs:
        candidate = os.path.join(ld, base + ".txt")
        if os.path.exists(candidate):
            return candidate
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
