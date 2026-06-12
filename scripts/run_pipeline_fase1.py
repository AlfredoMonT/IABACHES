import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import glob
import random
import numpy as np
import cv2
from tqdm import tqdm

from src.preprocessing.image_processor import apply_clahe, extract_positive_patch, extract_negative_patch
from src.features.hog_extractor import extract_hog_batch
from src.models.random_forest_model import train, save_model
from src.evaluation.metrics import compute_metrics, print_report, plot_confusion_matrix, plot_metrics_bar, save_metrics
from src.utils.helpers import set_seed, find_label_for_image, ensure_dir

PATCH_SIZE = 64
MAX_SAMPLES = 99999
DATA_DIR = "data/raw"
MODEL_OUT = "models/saved/random_forest_fase1.pkl"
FIGURES_DIR = "reports/figures"
METRICS_OUT = "reports/phase1/metrics_fase1.json"

set_seed(42)
ensure_dir(FIGURES_DIR)
ensure_dir("reports/phase1")

print("Buscando imagenes...")
all_images = (
    glob.glob(os.path.join(DATA_DIR, "**/*.jpg"), recursive=True)
    + glob.glob(os.path.join(DATA_DIR, "**/*.png"), recursive=True)
)
all_labels = [
    l for l in glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
    if "classes" not in l and "notes" not in l
]
label_dirs = list(set([os.path.dirname(l) for l in all_labels]))

print(f"Imagenes encontradas: {len(all_images)}")
print(f"Labels encontrados:   {len(all_labels)}")

random.shuffle(all_images)
patches_pos, patches_neg = [], []

for img_path in tqdm(all_images, desc="Procesando"):
    if len(patches_pos) >= MAX_SAMPLES and len(patches_neg) >= MAX_SAMPLES:
        break

    label_path = find_label_for_image(img_path, label_dirs)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = apply_clahe(gray)

    if label_path and os.path.getsize(label_path) > 0:
        with open(label_path) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        bboxes = []
        for parts in lines:
            if len(parts) >= 5:
                _, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append((cx, cy, bw, bh))
                if len(patches_pos) < MAX_SAMPLES:
                    patch = extract_positive_patch(gray, cx, cy, bw, bh, PATCH_SIZE)
                    if patch is not None:
                        patches_pos.append(patch)
        if len(patches_neg) < MAX_SAMPLES and bboxes:
            neg = extract_negative_patch(gray, bboxes, PATCH_SIZE)
            if neg is not None:
                patches_neg.append(neg)
    else:
        H, W = gray.shape
        if H >= PATCH_SIZE and W >= PATCH_SIZE and len(patches_neg) < MAX_SAMPLES:
            px = random.randint(0, W - PATCH_SIZE)
            py = random.randint(0, H - PATCH_SIZE)
            patches_neg.append(gray[py:py + PATCH_SIZE, px:px + PATCH_SIZE])

print(f"Parches positivos: {len(patches_pos)}")
print(f"Parches negativos: {len(patches_neg)}")

print("Extrayendo HOG...")
X_pos = extract_hog_batch(patches_pos)
X_neg = extract_hog_batch(patches_neg)
X = np.vstack([X_pos, X_neg])
y = np.array([1] * len(X_pos) + [0] * len(X_neg))

print(f"Forma del dataset: {X.shape}")

print("Entrenando Random Forest...")
model, X_train, X_test, y_train, y_test = train(X, y)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print_report(y_test, y_pred)
metrics = compute_metrics(y_test, y_pred, y_proba)
print(metrics)

plot_confusion_matrix(y_test, y_pred, save_path=os.path.join(FIGURES_DIR, "confusion_matrix_fase1.png"))
plot_metrics_bar(metrics, save_path=os.path.join(FIGURES_DIR, "metrics_fase1.png"))
save_metrics(metrics, METRICS_OUT)
save_model(model, MODEL_OUT)

print(f"Modelo guardado en {MODEL_OUT}")
print(f"Metricas guardadas en {METRICS_OUT}")
