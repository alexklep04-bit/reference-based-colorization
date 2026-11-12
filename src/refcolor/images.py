"""Image loading, resizing and colour-space helpers.

All models in this project work in the CIELAB colour space: the grayscale input is the
lightness channel L (0..100) and a model predicts the two chrominance channels a and b.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage import color


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as uint8 RGB, applying the EXIF orientation stored by phone cameras."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(img)


def pad_to_square(rgb: np.ndarray, fill: int = 255) -> np.ndarray:
    """Pad the shorter side symmetrically so that the whole image is kept."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), fill, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    out[top : top + h, left : left + w] = rgb
    return out


def center_crop_square(rgb: np.ndarray) -> np.ndarray:
    """Crop the longer side symmetrically (used for photographs without a plain background)."""
    h, w = rgb.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return rgb[top : top + side, left : left + side]


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize a square image to size x size pixels with Lanczos resampling."""
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.LANCZOS))


def lightness(rgb: np.ndarray) -> np.ndarray:
    """Return the CIELAB lightness channel L in [0, 100]; this is the grayscale model input."""
    return color.rgb2lab(rgb)[:, :, 0].astype(np.float32)


def lab_to_rgb(l: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Combine lightness (H x W) and chrominance (H x W x 2) into a uint8 RGB image."""
    lab = np.concatenate([l[:, :, None], ab], axis=2).astype(np.float64)
    return (np.clip(color.lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8)


def gray_to_rgb(l: np.ndarray) -> np.ndarray:
    """Grayscale image as shown to the user: the lightness channel with zero chrominance."""
    return lab_to_rgb(l, np.zeros((*l.shape, 2), dtype=np.float32))
