"""Image quality metrics computed against the ground-truth colour image."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Peak signal-to-noise ratio in dB for uint8 RGB images; optionally only inside a mask."""
    diff = (pred.astype(np.float64) - truth.astype(np.float64)) ** 2
    mse = diff[mask].mean() if mask is not None else diff.mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def ssim(pred: np.ndarray, truth: np.ndarray) -> float:
    """Structural similarity index for uint8 RGB images."""
    return float(structural_similarity(truth, pred, channel_axis=2, data_range=255))


def object_mask(truth: np.ndarray, white_threshold: int = 245) -> np.ndarray:
    """Pixels that belong to the object in images with a plain white background."""
    return truth.min(axis=2) < white_threshold
