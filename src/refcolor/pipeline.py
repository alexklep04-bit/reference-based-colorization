"""Prototype pipeline: colorize a grayscale photograph of any size with or without a reference."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .automatic import AutomaticColorizer
from .exemplar import ExemplarColorizer, render
from .images import lightness

MODEL_SIDE = 512  # longer side of the image given to the reference-based model


@dataclass
class Colorization:
    rgb: np.ndarray  # result at the resolution of the input photograph
    warped_rgb: np.ndarray | None = None  # reference colours as matched to the target (diagnostic)
    similarity: np.ndarray | None = None  # matching confidence map (diagnostic)


@lru_cache(maxsize=1)
def exemplar_model() -> ExemplarColorizer:
    return ExemplarColorizer()


@lru_cache(maxsize=1)
def automatic_model() -> AutomaticColorizer:
    return AutomaticColorizer()


def _model_size(height: int, width: int) -> tuple[int, int]:
    """Size used inside the model: longer side about MODEL_SIDE, height multiple of 16, width of 32."""
    scale = MODEL_SIDE / max(height, width)
    return max(64, round(height * scale / 16) * 16), max(64, round(width * scale / 32) * 32)


def _cover(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio so that the image covers height x width, then centre-crop."""
    h, w = rgb.shape[:2]
    scale = max(height / h, width / w)
    new_h, new_w = max(height, round(h * scale)), max(width, round(w * scale))
    resized = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))
    top, left = (new_h - height) // 2, (new_w - width) // 2
    return resized[top : top + height, left : left + width]


def colorize_with_reference(target_rgb: np.ndarray, reference_rgb: np.ndarray, chroma_gain: float = 1.0) -> Colorization:
    """Colorize a (grayscale) photograph using a colour photograph of the same object.

    Only the lightness of the target is used, so a colour target is treated as its grayscale version.
    The colours are predicted at reduced resolution and combined with the full-resolution lightness.
    """
    target_l = lightness(target_rgb)
    h, w = _model_size(*target_l.shape)
    target_small = np.asarray(Image.fromarray(target_rgb).resize((w, h), Image.LANCZOS))
    pred = exemplar_model().predict(lightness(target_small), _cover(reference_rgb, h, w))
    rgb = render(target_l, pred.ab_half, chroma_gain=chroma_gain)
    return Colorization(rgb=rgb, warped_rgb=pred.warped_rgb, similarity=pred.similarity)


def colorize_automatically(target_rgb: np.ndarray) -> Colorization:
    """Colorize a (grayscale) photograph without any reference."""
    return Colorization(rgb=automatic_model().colorize(lightness(target_rgb)))
