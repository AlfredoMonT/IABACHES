"""
hog_extractor.py
================
Módulo de extracción de características HOG (Histogram of Oriented Gradients)
para la clasificación de baches con Random Forest.

Responsable: Orellana Cusihuaman Luis Anthony
Curso: Inteligencia Artificial Aplicada — UAC 2026

¿Por qué HOG?
    HOG captura la distribución local de gradientes de intensidad en una imagen.
    Los bordes irregulares de un bache tienen un patrón de gradientes muy diferente
    al asfalto liso, lo que lo convierte en el descriptor ideal para este problema.

    Un vector HOG convierte una imagen 2D en un array 1D tabular que el Random Forest
    puede procesar directamente, sin necesidad de GPU.
"""

import numpy as np
import cv2
from skimage.feature import hog
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


# ─── Parámetros HOG ────────────────────────────────────────────────────────────
HOG_PARAMS = {
    "orientations": 9,          # Número de bins de ángulo (0°-180°)
    "pixels_per_cell": (8, 8),  # Tamaño de cada celda local
    "cells_per_block": (2, 2),  # Normalización por bloques (reduce efecto de luz)
    "block_norm": "L2-Hys",     # Normalización robusta recomendada en literatura
    "visualize": False,         # No necesitamos imagen visual en producción
    "feature_vector": True      # Salida como vector 1D plano
}


# ─── Extracción HOG ────────────────────────────────────────────────────────────

def extract_hog_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extrae el vector de características HOG de una imagen en escala de grises.

    Para una imagen de 64x64 píxeles con los parámetros definidos:
        - Celdas: 8x8 → imagen tiene 8x8 = 64 celdas
        - Bloques: cada bloque tiene 2x2 celdas → 7x7 = 49 bloques
        - Por bloque: 2x2 celdas x 9 orientaciones = 36 valores
        - Vector total: 49 bloques x 36 = 1,764 características

    Args:
        gray_img: Imagen en escala de grises de tamaño (64, 64).

    Returns:
        Vector NumPy 1D de características HOG (shape: [1764,]).
    """
    features = hog(gray_img, **HOG_PARAMS)
    return features


def extract_glcm_features(gray_img: np.ndarray) -> np.ndarray:
    """
    Extrae características de textura mediante GLCM
    (Gray Level Co-occurrence Matrix).

    Complementa HOG capturando propiedades de textura del asfalto:
    contraste, correlación, energía y homogeneidad.

    Args:
        gray_img: Imagen en escala de grises.

    Returns:
        Vector de 4 características GLCM: [contraste, correlación, energía, homogeneidad].
    """
    # Reducir niveles de gris para eficiencia computacional
    img_8 = (gray_img / 16).astype(np.uint8)
    glcm = graycomatrix(img_8, distances=[1], angles=[0], levels=16,
                        symmetric=True, normed=True)

    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return np.array([contrast, correlation, energy, homogeneity])


def extract_combined_features(gray_img: np.ndarray,
                               use_glcm: bool = True) -> np.ndarray:
    """
    Combina HOG + GLCM en un único vector de características.

    HOG aporta información morfológica (forma del bache).
    GLCM aporta información de textura (rugosidad del asfalto).
    Juntos dan al Random Forest más dimensiones para discriminar.

    Args:
        gray_img: Imagen en escala de grises (64x64).
        use_glcm: Si True, concatena features GLCM al vector HOG.

    Returns:
        Vector combinado [HOG | GLCM] o solo HOG si use_glcm=False.
    """
    hog_feats = extract_hog_features(gray_img)

    if use_glcm:
        glcm_feats = extract_glcm_features(gray_img)
        return np.concatenate([hog_feats, glcm_feats])

    return hog_feats


# ─── Extracción en lote ────────────────────────────────────────────────────────

def build_feature_matrix(data_dir: str,
                          classes: list = ['pothole', 'normal'],
                          use_glcm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye la matriz de características X y el vector de etiquetas y
    a partir de un directorio con imágenes organizadas por clase.

    Estructura esperada de data_dir:
        data_dir/
            pothole/  → etiqueta 1
            normal/   → etiqueta 0

    Args:
        data_dir: Directorio con subdirectorios por clase.
        classes: Lista de clases. El índice en la lista es la etiqueta numérica.
        use_glcm: Si True, incluye features GLCM adicionales.

    Returns:
        X: Matriz de características shape (n_samples, n_features).
        y: Vector de etiquetas shape (n_samples,).
    """
    X_list, y_list = [], []

    for label, cls in enumerate(classes):
        cls_dir = Path(data_dir) / cls
        images  = list(cls_dir.glob("*.jpg")) + \
                  list(cls_dir.glob("*.jpeg")) + \
                  list(cls_dir.glob("*.png"))

        print(f"📂 Clase '{cls}' (label={label}): {len(images)} imágenes")

        for img_path in tqdm(images, desc=f"Extrayendo features: {cls}"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            features = extract_combined_features(img, use_glcm=use_glcm)
            X_list.append(features)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n✅ Matriz de características generada:")
    print(f"   X shape: {X.shape} | y shape: {y.shape}")
    print(f"   Features por imagen: {X.shape[1]}")
    print(f"   Distribución: {dict(zip(classes, [np.sum(y==i) for i in range(len(classes))]))}")

    return X, y


# ─── Ejecución directa ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔬 Módulo de extracción HOG cargado.")
    print(f"   Parámetros HOG: {HOG_PARAMS}")
    print(f"   Dimensión HOG esperada (64x64): ~1,764 features")
    print(f"   Dimensión HOG + GLCM: ~1,768 features")
