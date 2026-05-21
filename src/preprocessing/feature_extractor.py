"""
feature_extractor.py
---------------------
Extracción de características HOG y GLCM para clasificación con Random Forest.
Universidad Andina del Cusco — Proyecto IABACHES
"""

import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from pathlib import Path


# Configuración por defecto del pipeline HOG
HOG_CONFIG = {
    'img_size': (64, 64),
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
}


def aplicar_clahe(img_gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Aplica CLAHE (ecualización adaptativa de histograma) para mejorar contraste.
    Especialmente útil para imágenes nocturnas o con sombras fuertes.

    Args:
        img_gray: Imagen en escala de grises.
        clip_limit: Límite de contraste (default 2.0).

    Returns:
        Imagen con contraste mejorado.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def extraer_hog(img: np.ndarray, config: dict = HOG_CONFIG) -> np.ndarray:
    """
    Extrae el vector de características HOG de una imagen.

    HOG captura la distribución local de gradientes de intensidad, lo que
    describe eficientemente la textura y forma de los bordes de un bache.

    Args:
        img: Imagen (RGB, BGR o gris). Se convierte internamente.
        config: Diccionario de hiperparámetros HOG.

    Returns:
        Vector 1D de características HOG.
    """
    # Asegurar escala de grises
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Resize estándar
    img_resized = cv2.resize(img_gray, config['img_size'])

    # CLAHE para normalizar contraste
    img_clahe = aplicar_clahe(img_resized)

    # HOG
    hog_vector = hog(
        img_clahe,
        orientations=config['orientations'],
        pixels_per_cell=config['pixels_per_cell'],
        cells_per_block=config['cells_per_block'],
        feature_vector=True
    )

    return hog_vector


def pipeline_completo(img: np.ndarray, bbox: tuple = None) -> np.ndarray | None:
    """
    Pipeline completo: recorte → resize → CLAHE → HOG.

    Args:
        img: Imagen numpy (RGB o BGR).
        bbox: Tupla (x, y, w, h) en píxeles para recortar. None = imagen completa.

    Returns:
        Vector HOG 1D o None si falla.
    """
    if img is None or img.size == 0:
        return None

    # Recortar si se da bounding box
    if bbox is not None:
        x, y, w, h = bbox
        img = img[y:y+h, x:x+w]
        if img.size == 0:
            return None

    return extraer_hog(img)
