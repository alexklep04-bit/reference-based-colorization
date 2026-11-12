"""Download a small pilot set of same-object, multi-view images with exactly known viewpoint angles.

Source: Amazon Berkeley Objects (ABO), "spins" subset -- turntable sequences of 72 images per
product, one every 5 degrees of azimuth, on a white background. Licence: CC BY 4.0.
J. Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022.
https://amazon-berkeley-objects.s3.amazonaws.com/index.html

Selection rule (deterministic for a given seed):
  1. spins with 72 views, shorter side >= 500 px, aspect ratio between 1:2 and 2:1;
  2. candidates are visited in random order (fixed seed);
  3. a candidate is accepted if
       - the object is not nearly achromatic: mean CIELAB chroma of the object pixels >= --min-chroma
         (colorizing black, white or gray objects is trivial and says nothing about colour transfer);
       - it stays visible from the side: its area at 90/180/270 degrees is >= --min-area-ratio of the
         area at 0 degrees (excludes flat objects such as posters, which are edge-on at 90 degrees);
  4. the first --objects accepted candidates form the pilot set.

Usage:
    python src/prepare_abo_pilot.py --objects 12 --seed 0
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

from refcolor.images import pad_to_square, resize
from refcolor.metrics import object_mask
from refcolor.paths import DATA

BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/spins/"
ABO_DIR = DATA / "abo"
CHECK_VIEWS = (0, 18, 36, 54)  # 0, 90, 180 and 270 degrees


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def load_metadata() -> pd.DataFrame:
    ABO_DIR.mkdir(parents=True, exist_ok=True)
    path = ABO_DIR / "spins.csv.gz"
    if not path.exists():
        path.write_bytes(download(BASE_URL + "metadata/spins.csv.gz"))
    return pd.read_csv(path)


def describe(view_bytes: list[bytes]) -> tuple[float, float]:
    """Mean chroma of the object in the first view and the smallest relative area over the views."""
    areas, chroma = [], 0.0
    for i, data in enumerate(view_bytes):
        rgb = resize(pad_to_square(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))), 256)
        mask = object_mask(rgb)
        areas.append(mask.mean())
        if i == 0 and mask.any():
            lab = color.rgb2lab(rgb)
            chroma = float(np.hypot(lab[..., 1], lab[..., 2])[mask].mean())
    return chroma, float(min(areas) / max(areas[0], 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-chroma", type=float, default=10.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.5)
    args = parser.parse_args()

    meta = load_metadata()
    spins = meta.groupby("spin_id").agg(views=("azimuth", "count"), height=("height", "first"), width=("width", "first"))
    eligible = spins[
        (spins.views == 72) & (spins[["height", "width"]].min(axis=1) >= 500) & (spins.width / spins.height).between(0.5, 2.0)
    ]
    order = eligible.sample(frac=1.0, random_state=args.seed).index
    paths = meta.set_index(["spin_id", "azimuth"]).path
    print(f"{len(eligible)} eligible spins; looking for {args.objects} objects")

    selected, visited = [], 0
    with ThreadPoolExecutor(8) as pool:
        for spin_id in order:
            if len(selected) == args.objects:
                break
            visited += 1
            views = list(pool.map(download, [BASE_URL + "original/" + paths[(spin_id, v)] for v in CHECK_VIEWS]))
            chroma, area_ratio = describe(views)
            accepted = chroma >= args.min_chroma and area_ratio >= args.min_area_ratio
            print(f"  {spin_id}  chroma={chroma:5.1f}  area_ratio={area_ratio:4.2f}  {'accepted' if accepted else 'skipped'}")
            if accepted:
                selected.append(
                    dict(spin_id=spin_id, mean_chroma=round(chroma, 1), min_area_ratio=round(area_ratio, 2),
                         height=int(eligible.height[spin_id]), width=int(eligible.width[spin_id]))
                )

        print(f"visited {visited} candidates, accepted {len(selected)}; downloading all 72 views of each")
        for row in selected:
            folder = ABO_DIR / "spins" / row["spin_id"]
            folder.mkdir(parents=True, exist_ok=True)
            todo = [(v, folder / f"{row['spin_id']}_{v:02d}.jpg") for v in range(72)]
            todo = [(v, f) for v, f in todo if not f.exists()]
            for (v, f), data in zip(todo, pool.map(download, [BASE_URL + "original/" + paths[(row["spin_id"], v)] for v, _ in todo])):
                f.write_bytes(data)

    out = DATA / "abo_pilot.csv"
    pd.DataFrame(selected).to_csv(out, index=False)
    print(f"selection written to {out}")


if __name__ == "__main__":
    main()
