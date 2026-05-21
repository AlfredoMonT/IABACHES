"""
image_loader.py
---------------
Funciones para cargar y validar imágenes del dataset de baches.
Universidad Andina del Cusco — Proyecto IABACHES
"""

import cv2
import numpy as np
from pathlib import Path


def cargar_imagen(img_path: str | Path, color: str = 'rgb') -> np.ndarray | None:
    """
    Carga una imagen desde disco.

    Args:
        img_path: Ruta a la imagen.
        color: 'rgb', 'gray' o 'bgr'.

    Returns:
        Array numpy de la imagen o None si falla.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    if color == 'rgb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color == 'gray':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # BGR por defecto de OpenCV


def listar_imagenes(directorio: str | Path, extensiones: tuple = ('.jpg', '.jpeg', '.png')) -> list:
    """
    Lista todas las imágenes en un directorio.

    Args:
        directorio: Carpeta donde buscar.
        extensiones: Tupla de extensiones válidas.

    Returns:
        Lista de Path objects de imágenes encontradas.
    """
    directorio = Path(directorio)
    imagenes = []
    for ext in extensiones:
        imagenes.extend(directorio.glob(f'*{ext}'))
    return sorted(imagenes)


def leer_etiqueta_yolo(label_path: str | Path, img_w: int, img_h: int) -> list:
    """
    Lee un archivo de etiqueta YOLO y convierte a coordenadas de píxeles.

    Args:
        label_path: Ruta al archivo .txt con anotaciones YOLO.
        img_w: Ancho de la imagen en píxeles.
        img_h: Alto de la imagen en píxeles.

    Returns:
        Lista de dicts con keys: clase, x, y, w, h (en píxeles).
    """
    label_path = Path(label_path)
    bboxes = []

    if not label_path.exists():
        return bboxes

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                clase = int(parts[0])
                xc, yc, bw, bh = [float(p) for p in parts[1:]]
                # Convertir normalizado → píxeles
                x = int((xc - bw / 2) * img_w)
                y = int((yc - bh / 2) * img_h)
                w = int(bw * img_w)
                h = int(bh * img_h)
                bboxes.append({'clase': clase, 'x': max(0, x), 'y': max(0, y), 'w': w, 'h': h})

    return bboxes
