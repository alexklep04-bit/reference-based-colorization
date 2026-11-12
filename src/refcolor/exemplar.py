"""Reference-based colorization with the pretrained model of Zhang et al. (CVPR 2019).

B. Zhang, M. He, J. Liao, P. V. Sander, L. Yuan, A. Bermak, D. Chen,
"Deep Exemplar-based Video Colorization", CVPR 2019.
Official code and weights: https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization

The official test script requires an NVIDIA GPU. This module reproduces its inference steps for a
single image (no frame propagation, i.e. the image colorization mode) so that the model runs on
the CPU of a laptop. The official pretrained weights are loaded unchanged.

Pipeline:
  1. VGG-19 features are extracted from the grayscale target and from the colour reference.
  2. The correspondence network (WarpNet) matches every target location with the most similar
     reference location and warps the reference colours onto the target.
  3. The colorization network (ColorVidNet) refines the warped colours and fills in regions
     without a reliable match.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import color

from .images import lab_to_rgb
from .nets import ColorVidNet, VGG19Features, WarpNet, feature_normalize, lab_to_rgb_tensor
from .paths import EXEMPLAR_WEIGHTS

L_MEAN = 50.0  # the model expects a zero-centred lightness channel
VGG_LAYERS = ["r12", "r22", "r32", "r42", "r52"]
# Values of the official test script
TEMPERATURE = 1e-10  # practically a hard (winner-takes-all) match
OFFICIAL_CHROMA_GAIN = 1.25  # the official script multiplies the predicted chrominance by 1.25
WLS_LAMBDA, WLS_SIGMA = 500, 4


@dataclass
class ExemplarPrediction:
    ab_half: np.ndarray  # raw chrominance predicted by the network, 2 x H/2 x W/2
    warped_rgb: np.ndarray  # reference colours warped onto the target, before refinement (H/2 x W/2)
    similarity: np.ndarray  # cosine similarity of the best match per location (H/2 x W/2)


@dataclass
class ExemplarResult:
    rgb: np.ndarray  # final colorization, H x W x 3 uint8
    warped_rgb: np.ndarray
    similarity: np.ndarray


def render(target_l: np.ndarray, ab_half: np.ndarray, chroma_gain: float = 1.0, wls_filter: bool = True) -> np.ndarray:
    """Turn the predicted chrominance into the final image (post-processing of the official script).

    The chrominance is upsampled to the resolution of target_l, multiplied by chroma_gain and
    smoothed along the edges of the lightness channel (fast global smoother, a WLS filter).

    chroma_gain = 1.25 reproduces the official script, which makes the result more vivid;
    chroma_gain = 1.0 keeps the predicted colours unchanged and is closer to the true colours.
    """
    target_l = np.asarray(target_l, dtype=np.float32)
    ab = F.interpolate(torch.from_numpy(ab_half)[None], size=target_l.shape, mode="bilinear")[0].numpy() * chroma_gain
    if wls_filter:
        guide = (target_l * 255 / 100).astype(np.uint8)
        smoother = cv2.ximgproc.createFastGlobalSmootherFilter(guide, WLS_LAMBDA, WLS_SIGMA)
        ab = np.stack([smoother.filter(ab[0]), smoother.filter(ab[1])])
    return lab_to_rgb(target_l, ab.transpose(1, 2, 0))


def _load(net: torch.nn.Module, path) -> torch.nn.Module:
    # weights_only=True: the file is read as plain tensors, no code stored in it can be executed
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


class ExemplarColorizer:
    def __init__(self) -> None:
        missing = [str(p) for p in EXEMPLAR_WEIGHTS.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Model weights not found, run scripts/setup_models.sh first: {missing}")
        self.vggnet = _load(VGG19Features(), EXEMPLAR_WEIGHTS["vgg"])
        self.warpnet = _load(WarpNet(), EXEMPLAR_WEIGHTS["warp"])
        self.colornet = _load(ColorVidNet(7), EXEMPLAR_WEIGHTS["color"])

    def _features(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """Normalised relu2_1 .. relu5_1 activations used for matching."""
        return [feature_normalize(f) for f in self.vggnet(rgb, VGG_LAYERS)[1:]]

    @torch.no_grad()
    def predict(self, target_l: np.ndarray, reference_rgb: np.ndarray) -> ExemplarPrediction:
        """Run the networks: grayscale target + colour reference -> chrominance at half resolution.

        target_l       H x W float array, CIELAB lightness in [0, 100]
        reference_rgb  H x W x 3 uint8 array of the same size
        H must be a multiple of 16 and W a multiple of 32.
        """
        target_l = np.asarray(target_l, dtype=np.float32)
        h, w = target_l.shape
        if reference_rgb.shape[:2] != (h, w) or h % 16 or w % 32:
            raise ValueError("target and reference must have the same size (height multiple of 16, width of 32)")

        # Target: only the lightness channel is given to the model (its chrominance is zero).
        target_lab = torch.zeros(1, 3, h, w)
        target_lab[0, 0] = torch.from_numpy(target_l) - L_MEAN
        ref_lab = torch.from_numpy(color.rgb2lab(reference_rgb).transpose(2, 0, 1)).float().unsqueeze(0)
        ref_lab[:, 0] -= L_MEAN

        # The model works at half resolution.
        target_half = F.interpolate(target_lab, scale_factor=0.5, mode="bilinear")
        ref_half = F.interpolate(ref_lab, scale_factor=0.5, mode="bilinear")
        target_l_half = target_half[:, 0:1]

        # 1. Features: the grayscale target is fed to VGG as a gray RGB image.
        target_gray_rgb = ((target_l_half + L_MEAN) / 100.0).repeat(1, 3, 1, 1)
        ref_rgb = lab_to_rgb_tensor(torch.cat((ref_half[:, 0:1] + L_MEAN, ref_half[:, 1:3]), dim=1))
        target_feats, ref_feats = self._features(target_gray_rgb), self._features(ref_rgb)

        # 2. Correspondence: warp the reference colours onto the target.
        warped_lab, similarity = self.warpnet(ref_half, target_feats, ref_feats, temperature=TEMPERATURE)

        # 3. Colorization: refine the warped colours (the "previous frame" input stays empty for images).
        previous_frame = torch.zeros_like(target_half)
        ab_half = self.colornet(torch.cat((target_l_half, warped_lab[:, 1:3], similarity, previous_frame), dim=1))

        warped_rgb = lab_to_rgb((target_l_half[0, 0] + L_MEAN).numpy(), warped_lab[0, 1:3].numpy().transpose(1, 2, 0))
        return ExemplarPrediction(ab_half=ab_half[0].numpy(), warped_rgb=warped_rgb, similarity=similarity[0, 0].numpy())

    def colorize(
        self, target_l: np.ndarray, reference_rgb: np.ndarray, chroma_gain: float = 1.0, wls_filter: bool = True
    ) -> ExemplarResult:
        """Colorize a grayscale target using a colour reference image of the same size."""
        pred = self.predict(target_l, reference_rgb)
        rgb = render(target_l, pred.ab_half, chroma_gain=chroma_gain, wls_filter=wls_filter)
        return ExemplarResult(rgb=rgb, warped_rgb=pred.warped_rgb, similarity=pred.similarity)
