import cv2
import numpy as np


PATCH_SIZE = 64


def to_grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray_img, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


def resize_patch(img, size=PATCH_SIZE):
    return cv2.resize(img, (size, size))


def preprocess_patch(img_bgr, size=PATCH_SIZE):
    gray = to_grayscale(img_bgr)
    enhanced = apply_clahe(gray)
    resized = resize_patch(enhanced, size)
    return resized


def extract_positive_patch(gray_img, cx, cy, bw, bh, size=PATCH_SIZE):
    H, W = gray_img.shape
    x1 = max(0, int((cx - bw / 2) * W))
    y1 = max(0, int((cy - bh / 2) * H))
    x2 = min(W, int((cx + bw / 2) * W))
    y2 = min(H, int((cy + bh / 2) * H))
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    crop = gray_img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))


def extract_negative_patch(gray_img, bboxes, size=PATCH_SIZE, max_attempts=30):
    import random
    H, W = gray_img.shape
    for _ in range(max_attempts):
        px = random.randint(0, max(0, W - size))
        py = random.randint(0, max(0, H - size))
        overlap = False
        for cx, cy, bw, bh in bboxes:
            bx1 = int((cx - bw / 2) * W)
            by1 = int((cy - bh / 2) * H)
            bx2 = int((cx + bw / 2) * W)
            by2 = int((cy + bh / 2) * H)
            if not (px + size < bx1 or px > bx2 or py + size < by1 or py > by2):
                overlap = True
                break
        if not overlap:
            crop = gray_img[py:py + size, px:px + size]
            return cv2.resize(crop, (size, size))
    return None
