import numpy as np
from skimage.feature import hog


HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=False,
    feature_vector=True,
)


def extract_hog(patch):
    normalized = patch.astype(np.float32) / 255.0
    return hog(normalized, **HOG_PARAMS)


def extract_hog_visual(patch):
    normalized = patch.astype(np.float32) / 255.0
    params = {**HOG_PARAMS, "visualize": True}
    features, hog_image = hog(normalized, **params)
    return features, hog_image


def extract_hog_batch(patches):
    return np.array([extract_hog(p) for p in patches])
