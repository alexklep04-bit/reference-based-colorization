"""Automatic colorization without a reference: Zhang et al., "Colorful Image Colorization", ECCV 2016.

Official code and weights: https://github.com/richzhang/colorization
Used as the no-reference baseline. As in the official demo, the network predicts chrominance at
256 x 256 pixels, which is then resized to the resolution of the input.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .images import lab_to_rgb
from .nets import ECCVGenerator
from .paths import AUTOMATIC_WEIGHTS_URL, WEIGHTS

MODEL_SIZE = 256


class AutomaticColorizer:
    def __init__(self) -> None:
        state = torch.hub.load_state_dict_from_url(
            AUTOMATIC_WEIGHTS_URL,
            model_dir=str(WEIGHTS / "automatic"),
            map_location="cpu",
            check_hash=True,  # the file name carries the first characters of its SHA-256 hash
            weights_only=True,
        )
        self.model = ECCVGenerator()
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def colorize(self, target_l: np.ndarray) -> np.ndarray:
        """Colorize a grayscale image given as CIELAB lightness (H x W, values in [0, 100])."""
        target_l = np.asarray(target_l, dtype=np.float32)
        h, w = target_l.shape
        l_small = np.asarray(Image.fromarray(target_l).resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC))
        ab_small = self.model(torch.from_numpy(l_small.copy())[None, None])
        ab = F.interpolate(ab_small, size=(h, w), mode="bilinear")[0].numpy()
        return lab_to_rgb(target_l, ab.transpose(1, 2, 0))
