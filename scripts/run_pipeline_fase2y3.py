import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import glob
import shutil
import random
import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR


DATASET_DIR = "data/raw"
YOLO_DIR = "data/yolo_format"
YOLO_MODEL_OUT = "models/saved/best_yolo11n.pt"
RTDETR_MODEL_OUT = "models/saved/best_rtdetr.pt"


def prepare_yolo_dataset():
    base = Path(YOLO_DIR)
    for split in ["train", "val"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_images = (
        glob.glob(os.path.join(DATASET_DIR, "**/*.jpg"), recursive=True)
        + glob.glob(os.path.join(DATASET_DIR, "**/*.png"), recursive=True)
    )
    all_labels = [
        l for l in glob.glob(os.path.join(DATASET_DIR, "**/*.txt"), recursive=True)
        if "classes" not in l and "notes" not in l
    ]
    label_dirs = list(set([os.path.dirname(l) for l in all_labels]))

    def find_label(img_path):
        b = os.path.splitext(os.path.basename(img_path))[0]
        for ld in label_dirs:
            c = os.path.join(ld, b + ".txt")
            if os.path.exists(c):
                return c
        return None

    pairs = [(img, find_label(img)) for img in all_images if find_label(img)]
    random.seed(42)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for img, lbl in train_pairs:
        shutil.copy(img, base / "images" / "train" / os.path.basename(img))
        shutil.copy(lbl, base / "labels" / "train" / os.path.basename(lbl))

    for img, lbl in val_pairs:
        shutil.copy(img, base / "images" / "val" / os.path.basename(img))
        shutil.copy(lbl, base / "labels" / "val" / os.path.basename(lbl))

    yaml_content = f"path: {base.resolve()}\ntrain: images/train\nval: images/val\n\nnc: 1\nnames: [pothole]\n"
    yaml_path = base / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    return str(yaml_path)


def train_yolo(yaml_path, epochs=50):
    model = YOLO("yolo11n.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="yolo11n_baches",
        patience=10,
        augment=True,
        verbose=True,
    )
    best = "runs/detect/yolo11n_baches/weights/best.pt"
    os.makedirs("models/saved", exist_ok=True)
    shutil.copy(best, YOLO_MODEL_OUT)
    print(f"YOLO11n guardado en {YOLO_MODEL_OUT}")
    return YOLO_MODEL_OUT


def train_rtdetr(yaml_path, epochs=50):
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,
        name="rtdetr_baches",
        patience=10,
        verbose=True,
    )
    best = "runs/detect/rtdetr_baches/weights/best.pt"
    os.makedirs("models/saved", exist_ok=True)
    shutil.copy(best, RTDETR_MODEL_OUT)
    print(f"RT-DETR guardado en {RTDETR_MODEL_OUT}")
    return RTDETR_MODEL_OUT


def compare_models(yaml_path):
    import time
    yolo = YOLO(YOLO_MODEL_OUT)
    rtdetr = RTDETR(RTDETR_MODEL_OUT)

    val_yolo = yolo.val(data=yaml_path, verbose=False)
    val_rtdetr = rtdetr.val(data=yaml_path, verbose=False)

    sample = str(list(Path(YOLO_DIR, "images", "val").iterdir())[0])

    def fps(model, img, n=20):
        model(img, verbose=False)
        t0 = time.time()
        for _ in range(n):
            model(img, verbose=False)
        elapsed = time.time() - t0
        return n / elapsed

    fps_yolo = fps(yolo, sample)
    fps_rtdetr = fps(rtdetr, sample)

    print("\n--- Comparativa ---")
    print(f"Random Forest | F1: 62.4%  | mAP50:  -     | FPS:  9")
    print(f"YOLO11n       | F1: {val_yolo.box.f1.mean()*100:.1f}%  | mAP50: {val_yolo.box.map50*100:.1f}%  | FPS: {fps_yolo:.0f}")
    print(f"RT-DETR       | F1: {val_rtdetr.box.f1.mean()*100:.1f}%  | mAP50: {val_rtdetr.box.map50*100:.1f}%  | FPS: {fps_rtdetr:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["yolo", "rtdetr", "both", "compare"], default="both")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    yaml_path = prepare_yolo_dataset()

    if args.mode in ("yolo", "both"):
        train_yolo(yaml_path, args.epochs)

    if args.mode in ("rtdetr", "both"):
        train_rtdetr(yaml_path, args.epochs)

    if args.mode in ("compare", "both"):
        compare_models(yaml_path)
