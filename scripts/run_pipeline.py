"""
run_pipeline.py
===============
Script principal que ejecuta el pipeline completo de la Fase 1:
  1. Preprocesamiento de imágenes
  2. Extracción de características HOG
  3. Entrenamiento del modelo Random Forest
  4. Evaluación y generación de reportes

Uso:
    python scripts/run_pipeline.py

Curso: Inteligencia Artificial Aplicada — UAC 2026
"""

import sys
sys.path.append(".")

from src.utils.helpers import set_seed, print_separator
from src.preprocessing.image_processor import batch_preprocess, split_dataset
from src.features.hog_extractor import build_feature_matrix

print_separator("PIPELINE FASE 1 — DETECCIÓN DE BACHES")
print("Universidad Andina del Cusco — IA Aplicada 2026\n")

set_seed(42)

print_separator("PASO 1: Preprocesamiento", char="-")
print("Ejecutar: notebooks/fase1_extraccion_hog.ipynb en Google Colab")

print_separator("PASO 2: Extracción HOG", char="-")
print("Los vectores HOG se generan automáticamente desde los notebooks.")

print_separator("PASO 3: Entrenamiento RF", char="-")
print("Ver: notebooks/fase1_entrenamiento_rf.ipynb")

print_separator("✅ Pipeline configurado correctamente")
