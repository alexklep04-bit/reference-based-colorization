"""Check that this project reproduces the outputs published with the official implementations.

Both official repositories ship sample inputs together with the results of the official code:

* reference-based model (Zhang et al., CVPR 2019): a sample clip, a reference image and the
  colorized frames. For the first frame of a clip the model works in image mode (no previous
  frame), which is exactly what this project implements;
* automatic model (Zhang et al., ECCV 2016): a sample photograph and its colorization.

The script colorizes the same inputs on the CPU and measures the difference to the official results.

Usage (after cloning the two official repositories into third_party/):
    python src/verify_official_sample.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize

from refcolor.automatic import AutomaticColorizer
from refcolor.exemplar import OFFICIAL_CHROMA_GAIN, ExemplarColorizer
from refcolor.images import lightness, load_rgb
from refcolor.metrics import psnr, ssim
from refcolor.paths import RESULTS, ROOT

HEIGHT, WIDTH = 432, 768  # default frame size of the official test script of the reference-based model


def fit_official(rgb: np.ndarray) -> np.ndarray:
    """Resize and centre-crop to 432 x 768 the same way as the official test script (CenterPad)."""
    h, w = rgb.shape[:2]
    scale = WIDTH / w if h / w > HEIGHT / WIDTH else HEIGHT / h
    new_h, new_w = int(h * scale), int(w * scale)
    out = sk_resize(rgb, (new_h, new_w), mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
    top, left = (new_h - HEIGHT) // 2, (new_w - WIDTH) // 2
    return out[top : top + HEIGHT, left : left + WIDTH].astype(np.uint8)


def report(name: str, ours: np.ndarray, official: np.ndarray) -> None:
    diff = np.abs(ours.astype(float) - official.astype(float)).mean()
    print(f"{name}: PSNR = {psnr(ours, official):.2f} dB, SSIM = {ssim(ours, official):.4f}, mean absolute difference = {diff:.2f} of 255")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exemplar-repo", type=Path, default=ROOT / "third_party" / "exemplar_video")
    parser.add_argument("--automatic-repo", type=Path, default=ROOT / "third_party" / "colorization")
    parser.add_argument("--clip", default="v32")
    parser.add_argument("--ref", default="01")
    args = parser.parse_args()
    out_dir = RESULTS / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference-based model
    samples = args.exemplar_repo / "sample_videos"
    first_frame = sorted((samples / "clips" / args.clip).iterdir(), key=lambda p: int(p.stem))[0]
    frame = fit_official(load_rgb(first_frame))
    reference = fit_official(load_rgb(samples / "ref" / args.clip / f"{args.ref}.jpg"))
    official = load_rgb(samples / "output" / f"{args.clip}_{args.ref}" / "00000.jpg")
    ours = ExemplarColorizer().colorize(lightness(frame), reference, chroma_gain=OFFICIAL_CHROMA_GAIN).rgb
    report(f"reference-based model, clip {args.clip}, reference {args.ref}", ours, official)
    Image.fromarray(np.concatenate([reference, ours, official], axis=1)).save(
        out_dir / "exemplar_reference_ours_official.jpg", quality=92
    )

    # Automatic model
    photo = load_rgb(args.automatic_repo / "imgs" / "ansel_adams3.jpg")
    official = load_rgb(args.automatic_repo / "imgs_out" / "saved_eccv16.png")
    ours = AutomaticColorizer().colorize(lightness(photo))
    report("automatic model, ansel_adams3.jpg", ours, official)
    Image.fromarray(np.concatenate([ours, official], axis=1)).save(out_dir / "automatic_ours_official.jpg", quality=92)
    print("comparison images saved to", out_dir)


if __name__ == "__main__":
    main()
