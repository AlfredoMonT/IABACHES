"""
image_processor.py
==================
Módulo de preprocesamiento de imágenes para el pipeline de detección de baches.

Responsable: Orellana Cusihuaman Luis Anthony
Curso: Inteligencia Artificial Aplicada — UAC 2026

Funciones principales:
- Carga y redimensionamiento de imágenes
- Conversión a escala de grises
- Ecualización adaptativa de histograma (CLAHE) para imágenes nocturnas
- División del dataset en train/val/test
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random


# ─── Configuración global ──────────────────────────────────────────────────────
IMG_SIZE = (64, 64)          # Tamaño fijo de salida para todos los parches
RANDOM_SEED = 42             # Semilla para reproducibilidad
TRAIN_RATIO = 0.70           # 70% entrenamiento
VAL_RATIO   = 0.15           # 15% validación
TEST_RATIO  = 0.15           # 15% prueba


# ─── Funciones de preprocesamiento individual ──────────────────────────────────

def load_image(image_path: str) -> np.ndarray | None:
    """
    Carga una imagen desde disco.

    Args:
        image_path: Ruta completa a la imagen.

    Returns:
        Array NumPy BGR o None si la carga falla.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️  No se pudo cargar: {image_path}")
    return img


def resize_image(img: np.ndarray, size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Redimensiona la imagen al tamaño fijo definido.
    Se usa interpolación INTER_AREA para reducción (mejor calidad que INTER_LINEAR).

    Args:
        img: Imagen BGR de entrada.
        size: Tupla (ancho, alto) de salida. Default: (64, 64).

    Returns:
        Imagen redimensionada.
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convierte la imagen BGR a escala de grises.
    HOG trabaja sobre intensidades de píxel, no canales de color.

    Args:
        img: Imagen BGR de entrada.

    Returns:
        Imagen en escala de grises (1 canal).
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray_img: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Soluciona el riesgo identificado de baja visibilidad en imágenes nocturnas:
    realza los bordes del bache independientemente del nivel de luminosidad.

    Args:
        gray_img: Imagen en escala de grises.
        clip_limit: Límite de contraste. Mayor valor = más contraste (default 2.0).
        tile_size: Tamaño de la cuadrícula local. Default (8,8).

    Returns:
        Imagen con contraste mejorado.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def preprocess_image(image_path: str,
                     apply_contrast: bool = True) -> np.ndarray | None:
    """
    Pipeline completo de preprocesamiento para una sola imagen:
    Carga → Resize → Grayscale → CLAHE (opcional)

    Args:
        image_path: Ruta a la imagen original.
        apply_contrast: Si True, aplica CLAHE para mejorar contraste.

    Returns:
        Imagen preprocesada lista para extracción HOG, o None si falla.
    """
    img = load_image(image_path)
    if img is None:
        return None

    img = resize_image(img)
    img = to_grayscale(img)

    if apply_contrast:
        img = apply_clahe(img)

    return img


# ─── División del dataset ──────────────────────────────────────────────────────

def split_dataset(source_dir: str,
                  output_dir: str,
                  classes: list = ['pothole', 'normal'],
                  seed: int = RANDOM_SEED) -> dict:
    """
    Divide el dataset en train/val/test manteniendo la proporción por clase.

    Estructura esperada de source_dir:
        source_dir/
            pothole/  ← imágenes de baches
            normal/   ← imágenes de asfalto normal

    Estructura generada en output_dir:
        output_dir/
            train/pothole/, train/normal/
            val/pothole/,   val/normal/
            test/pothole/,  test/normal/

    Args:
        source_dir: Directorio con las imágenes organizadas por clase.
        output_dir: Directorio de salida para el dataset dividido.
        classes: Lista de nombres de clases.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        Diccionario con conteos por split y clase.
    """
    random.seed(seed)
    counts = {}

    for cls in classes:
        src_cls = Path(source_dir) / cls
        images = list(src_cls.glob("*.jpg")) + \
                 list(src_cls.glob("*.jpeg")) + \
                 list(src_cls.glob("*.png"))

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }

        counts[cls] = {s: len(imgs) for s, imgs in splits.items()}

        for split_name, split_images in splits.items():
            dest = Path(output_dir) / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(split_images,
                                 desc=f"Copiando {split_name}/{cls}"):
                shutil.copy2(img_path, dest / img_path.name)

    return counts


# ─── Procesamiento en lote ─────────────────────────────────────────────────────

def batch_preprocess(input_dir: str,
                     output_dir: str,
                     apply_contrast: bool = True) -> int:
    """
    Preprocesa todas las imágenes de un directorio y las guarda en otro.

    Args:
        input_dir: Directorio con imágenes originales.
        output_dir: Directorio de salida para imágenes preprocesadas.
        apply_contrast: Si True, aplica CLAHE.

    Returns:
        Número de imágenes procesadas exitosamente.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ['.jpg', '.jpeg', '.png']
    images = [f for f in input_path.iterdir()
              if f.suffix.lower() in extensions]

    processed = 0
    for img_path in tqdm(images, desc=f"Preprocesando {input_dir}"):
        result = preprocess_image(str(img_path), apply_contrast)
        if result is not None:
            out_file = output_path / img_path.name
            cv2.imwrite(str(out_file), result)
            processed += 1

    print(f"✅ {processed}/{len(images)} imágenes preprocesadas en {output_dir}")
    return processed


# ─── Ejecución directa ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔧 Módulo de preprocesamiento cargado correctamente.")
    print(f"   Tamaño de imagen: {IMG_SIZE}")
    print(f"   Split: {TRAIN_RATIO*100:.0f}% train / "
          f"{VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test")
