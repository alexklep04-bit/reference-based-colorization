"""Pilot experiment: how does the viewpoint difference between reference and target affect the result?

For every object of the pilot set and several target views, the target is converted to grayscale
(CIELAB lightness) and colorized
  * without a reference (automatic model, Zhang et al. 2016),
  * with a reference showing the same object rotated by 0, 5, ..., 180 degrees (Zhang et al. 2019),
  * with a reference showing a different object (control: what a wrong reference does).
The original colour image of the target is the ground truth, so PSNR and SSIM measure directly how
well the true colours are recovered.

Usage:
    python src/run_pilot.py
Outputs:
    results/pilot/metrics.csv        one row per colorization
    results/pilot/strips/<id>.jpg    visual comparison for the frontal view of every object
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from PIL import Image

from refcolor.automatic import AutomaticColorizer
from refcolor.exemplar import ExemplarColorizer, render
from refcolor.images import gray_to_rgb, lightness, load_rgb, pad_to_square, resize
from refcolor.metrics import object_mask, psnr, ssim
from refcolor.paths import DATA, RESULTS

SIZE = 512  # images are padded to a square and resized; the models predict colour at 256 x 256
STEP_DEG = 5  # azimuth step between consecutive views of a spin
TARGETS_DEG = (0, 90, 180, 270)
OFFSETS_DEG = (0, 5, 10, 15, 30, 45, 60, 90, 135, 180)
CHROMA_GAINS = (1.25, 1.0)  # 1.25 = official post-processing, 1.0 = predicted colours unchanged
STRIP_OFFSETS_DEG = (15, 45, 90, 180)


def load_view(spin_id: str, azimuth_deg: int) -> np.ndarray:
    index = (azimuth_deg // STEP_DEG) % 72
    path = DATA / "abo" / "spins" / spin_id / f"{spin_id}_{index:02d}.jpg"
    return resize(pad_to_square(load_rgb(path)), SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N objects (quick test)")
    args = parser.parse_args()

    spin_ids = pd.read_csv(DATA / "abo_pilot.csv").spin_id.tolist()[: args.limit]
    out_dir = RESULTS / "pilot"
    (out_dir / "strips").mkdir(parents=True, exist_ok=True)

    exemplar, automatic = ExemplarColorizer(), AutomaticColorizer()
    rows, start = [], time.time()

    def record(image, truth, mask, **info):
        rows.append(dict(**info, psnr=psnr(image, truth), psnr_object=psnr(image, truth, mask), ssim=ssim(image, truth)))

    for n, spin_id in enumerate(spin_ids):
        other_id = spin_ids[(n + 1) % len(spin_ids)]
        for target_deg in TARGETS_DEG:
            truth = load_view(spin_id, target_deg)
            target_l, mask = lightness(truth), object_mask(truth)
            mask_half = mask[::2, ::2]
            base = dict(object=spin_id, target_deg=target_deg)
            strip = {"gray": gray_to_rgb(target_l)}

            record(strip["gray"], truth, mask, **base, method="gray", offset_deg=np.nan, chroma_gain=np.nan, similarity=np.nan)
            strip["automatic"] = automatic.colorize(target_l)
            record(strip["automatic"], truth, mask, **base, method="automatic", offset_deg=np.nan, chroma_gain=np.nan, similarity=np.nan)

            references = [("reference", off, load_view(spin_id, target_deg + off)) for off in OFFSETS_DEG]
            references.append(("other_object", np.nan, load_view(other_id, target_deg)))
            for method, offset, reference in references:
                pred = exemplar.predict(target_l, reference)
                similarity = float(pred.similarity[mask_half].mean())
                for gain in CHROMA_GAINS:
                    image = render(target_l, pred.ab_half, chroma_gain=gain)
                    record(image, truth, mask, **base, method=method, offset_deg=offset, chroma_gain=gain, similarity=similarity)
                    if gain == 1.0 and method == "reference" and offset in STRIP_OFFSETS_DEG:
                        strip[f"ref_{offset}"] = image
                        strip[f"refimg_{offset}"] = reference

            if target_deg == 0:
                top = [strip["gray"], strip["automatic"]] + [strip[f"ref_{o}"] for o in STRIP_OFFSETS_DEG] + [truth]
                blank = np.full_like(truth, 255)
                bottom = [blank, blank] + [strip[f"refimg_{o}"] for o in STRIP_OFFSETS_DEG] + [blank]
                Image.fromarray(np.concatenate([np.concatenate(top, 1), np.concatenate(bottom, 1)], 0)).save(
                    out_dir / "strips" / f"{spin_id}.jpg", quality=90
                )
        print(f"[{n + 1}/{len(spin_ids)}] {spin_id} done, {time.time() - start:.0f} s elapsed", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics.csv", index=False)
    print(f"{len(table)} rows written to {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
