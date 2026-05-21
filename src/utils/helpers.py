"""
helpers.py
==========
Funciones auxiliares compartidas entre todos los módulos del proyecto.

Curso: Inteligencia Artificial Aplicada — UAC 2026
"""

import numpy as np
import random
import os


def set_seed(seed: int = 42) -> None:
    """
    Fija las semillas aleatorias para reproducibilidad total del experimento.
    Obligatorio en proyectos académicos reproducibles.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Semilla fijada: {seed}")


def print_separator(title: str = "", char: str = "=", width: int = 60) -> None:
    """Imprime un separador visual en consola."""
    if title:
        print(f"\n{char*width}")
        print(f"  {title}")
        print(f"{char*width}")
    else:
        print(char * width)


def count_images_per_class(data_dir: str, classes: list) -> dict:
    """Cuenta cuántas imágenes hay por clase en un directorio."""
    from pathlib import Path
    counts = {}
    for cls in classes:
        cls_dir = Path(data_dir) / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg")) + \
                   list(cls_dir.glob("*.jpeg")) + \
                   list(cls_dir.glob("*.png"))
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts
