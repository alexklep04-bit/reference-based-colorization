"""Command-line prototype: colorize a grayscale photograph using a colour reference photograph.

Usage:
    python src/colorize.py --target old_photo.jpg --reference colour_photo.jpg --out result.png
    python src/colorize.py --target old_photo.jpg --out result.png            (no reference: automatic)

Options:
    --vivid   multiply the predicted colours by 1.25, as the official script of the model does
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from refcolor.exemplar import OFFICIAL_CHROMA_GAIN
from refcolor.images import load_rgb
from refcolor.pipeline import colorize_automatically, colorize_with_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, required=True, help="grayscale photograph to colorize")
    parser.add_argument("--reference", type=Path, help="colour photograph of the same object")
    parser.add_argument("--out", type=Path, required=True, help="where to save the result")
    parser.add_argument("--vivid", action="store_true", help="boost the colours by 1.25 (official setting)")
    args = parser.parse_args()

    target = load_rgb(args.target)
    start = time.time()
    if args.reference:
        gain = OFFICIAL_CHROMA_GAIN if args.vivid else 1.0
        result = colorize_with_reference(target, load_rgb(args.reference), chroma_gain=gain)
        mode = f"reference-based (reference: {args.reference.name}, mean matching similarity {result.similarity.mean():.2f})"
    else:
        result = colorize_automatically(target)
        mode = "automatic, no reference"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.rgb).save(args.out)
    print(f"{args.target.name}: {target.shape[1]}x{target.shape[0]} px, {mode}, {time.time() - start:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
