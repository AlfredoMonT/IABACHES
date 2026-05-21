"""
download_dataset.py
===================
Script para descargar el dataset de Kaggle automáticamente.

Uso:
    python scripts/download_dataset.py

Prerequisito:
    Tener configurado ~/.kaggle/kaggle.json con tu API key de Kaggle.
    Ver: https://www.kaggle.com/docs/api
"""

import os
import subprocess
from pathlib import Path

DATASET_ID = "muskanverma24/pothole-detection-dataset-yolov11-optimized"
OUTPUT_DIR = "data/raw"

def download():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Descargando dataset: {DATASET_ID}")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET_ID,
        "-p", OUTPUT_DIR,
        "--unzip"
    ], check=True)
    print(f"✅ Dataset descargado en: {OUTPUT_DIR}")

if __name__ == "__main__":
    download()
