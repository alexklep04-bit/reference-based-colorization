"""Figures and summary table of the pilot experiment (reads results/pilot/metrics.csv).

Usage:
    python src/make_figures.py
Outputs (docs/figures/):
    pilot_psnr_vs_angle.png     colour accuracy against the viewpoint difference
    pilot_objects_near_far.png  per object: similar viewpoint vs. distant viewpoint
    pilot_examples.jpg          visual comparison for three objects
    example_<id>.jpg            the same comparison, one object per file
    pilot_objects.jpg           the twelve objects of the pilot set
and results/pilot/summary.md with the numbers behind the charts.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from refcolor.images import load_rgb, pad_to_square, resize
from refcolor.paths import DATA, RESULTS, ROOT

FIGURES = ROOT / "docs" / "figures"
# Colours: one accent hue for the subject of the chart, grays for context
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
INK, INK_SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEAR_DEG, FAR_DEG = (0, 5, 10, 15), (90, 135, 180)
EXAMPLES = ("4c7e45bc", "833aecc5", "5dbb18b5")  # uniform colour, hidden back side, fine pattern
STRIP_LABELS = ["Grayscale input", "Automatic\n(no reference)", "Reference at 15°", "Reference at 45°",
                "Reference at 90°", "Reference at 180°", "Ground truth"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
})


def style(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def view(spin_id: str, azimuth_deg: int, size: int) -> np.ndarray:
    path = DATA / "abo" / "spins" / spin_id / f"{spin_id}_{(azimuth_deg // 5) % 72:02d}.jpg"
    return resize(pad_to_square(load_rgb(path)), size)


def chart_psnr_vs_angle(data: pd.DataFrame) -> dict:
    same = data[(data.method == "reference") & (data.chroma_gain == 1.0)].groupby("offset_deg").psnr_object.agg(["mean", "sem"])
    context = {
        "Reference shows a different object": data[(data.method == "other_object") & (data.chroma_gain == 1.0)].psnr_object.mean(),
        "Grayscale input, no colour": data[data.method == "gray"].psnr_object.mean(),
        "Automatic colorization, no reference": data[data.method == "automatic"].psnr_object.mean(),
    }
    fig, ax = plt.subplots(figsize=(8, 4.3))
    style(ax)
    x = same.index.to_numpy()
    ax.fill_between(x, same["mean"] - same["sem"], same["mean"] + same["sem"], color=BLUE, alpha=0.10, linewidth=0)
    ax.plot(x, same["mean"], color=BLUE, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.text(92, same["mean"].loc[90.0] + 1.5, "Reference shows the same object", color=INK, fontsize=10, fontweight="bold", ha="center")
    for deg in (0.0, 180.0):
        ax.annotate(f"{same['mean'].loc[deg]:.1f} dB", (deg, same["mean"].loc[deg]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=INK_SECONDARY)
    for (label, value), va, dy in zip(context.items(), ("bottom", "bottom", "top"), (0.25, 0.25, -0.3)):
        ax.axhline(value, color=MUTED, linewidth=1.2)
        ax.text(180, value + dy, f"{label}  {value:.1f} dB", ha="right", va=va, fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(-6, 186)
    ax.set_ylim(18, 41)
    ax.set_xticks([0, 15, 30, 45, 60, 90, 135, 180])
    ax.set_yticks([20, 25, 30, 35, 40])
    ax.set_xlabel("Viewpoint difference between the reference and the target, degrees", fontsize=9.5)
    ax.text(0, 1.03, "PSNR on the object region, dB (higher = closer to the true colours)", transform=ax.transAxes,
            fontsize=9.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_psnr_vs_angle.png")
    plt.close(fig)
    return {"same": same, "context": context}


def chart_near_far(data: pd.DataFrame) -> pd.DataFrame:
    ref = data[(data.method == "reference") & (data.chroma_gain == 1.0)]
    near = ref[ref.offset_deg.isin(NEAR_DEG)].groupby("object").psnr_object.mean()
    far = ref[ref.offset_deg.isin(FAR_DEG)].groupby("object").psnr_object.mean()
    table = pd.DataFrame({"near": near, "far": far})
    table["drop"] = table.near - table.far
    table = table.sort_values("drop")  # largest drop on top of the chart

    fig, ax = plt.subplots(figsize=(8, 5.2))
    style(ax, grid_axis="x")
    ax.spines["bottom"].set_visible(False)
    rows = np.arange(len(table))
    ax.hlines(rows, table.far, table.near, color=GRID, linewidth=3, zorder=1)
    ax.scatter(table.near, rows, s=80, color=BLUE_LIGHT, edgecolor="white", linewidth=2, zorder=2, label="Reference within 15° of the target")
    ax.scatter(table.far, rows, s=80, color=BLUE, edgecolor="white", linewidth=2, zorder=3, label="Reference 90° to 180° away")
    for row, (spin_id, r) in zip(rows, table.iterrows()):
        thumb = OffsetImage(view(spin_id, 0, 96), zoom=0.27)
        ax.add_artist(AnnotationBbox(thumb, (0, row), xycoords=("axes fraction", "data"), xybox=(-20, 0),
                                     boxcoords="offset points", frameon=False, box_alignment=(0.5, 0.5)))
        if r["drop"] >= 1.0:
            ax.text(r["far"] - 0.5, row, f"−{r['drop']:.1f} dB", ha="right", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(table) - 0.3)
    ax.set_xlim(20, 45)
    ax.set_xlabel("PSNR on the object region, dB", fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=9.5, handletextpad=0.3, columnspacing=2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.11)
    fig.savefig(FIGURES / "pilot_objects_near_far.png")
    plt.close(fig)
    return table


def example_figure(spin_ids: tuple[str, ...], path) -> None:
    """Rows of results (with the references used underneath) cut from the strips saved by run_pilot.py."""
    cell = 256
    fig_w = 7 * 1.45
    row_h = 1.45 * 1.55
    fig, axes = plt.subplots(len(spin_ids), 1, figsize=(fig_w, row_h * len(spin_ids) + 0.45), squeeze=False)
    for ax, spin_id in zip(axes[:, 0], spin_ids):
        strip = np.asarray(Image.open(RESULTS / "pilot" / "strips" / f"{spin_id}.jpg"))
        size = strip.shape[0] // 2
        top = [resize(strip[:size, i * size : (i + 1) * size], cell) for i in range(7)]
        refs = [resize(strip[size:, i * size : (i + 1) * size], cell // 2) for i in range(2, 6)]
        canvas = np.full((cell + cell // 2 + 14, 7 * cell, 3), 255, dtype=np.uint8)
        for i, image in enumerate(top):
            canvas[:cell, i * cell : (i + 1) * cell] = image
        for i, image in enumerate(refs):
            x0 = (i + 2) * cell + cell // 4
            canvas[cell + 14 :, x0 : x0 + cell // 2] = image
        ax.imshow(canvas)
        ax.axis("off")
        ax.text(2 * cell - 14, cell + 14 + cell // 4, "reference\nused", ha="right", va="center", fontsize=8.5, color=MUTED)
    for i, label in enumerate(STRIP_LABELS):
        axes[0, 0].text(i * cell + cell / 2, -12, label, ha="center", va="bottom", fontsize=9,
                        color=INK if i in (0, 1, 6) else INK_SECONDARY, fontweight="bold" if i in (1, 6) else "normal")
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.42 / fig.get_figheight(), bottom=0.005, hspace=0.04)
    fig.savefig(path, dpi=180, pil_kwargs={"quality": 92})
    plt.close(fig)


def objects_sheet(spin_ids: list[str]) -> None:
    tile = 150
    rows = [np.concatenate([view(s, deg, tile) for deg in (0, 45, 90, 180)], axis=1) for s in spin_ids]
    half = (len(rows) + 1) // 2
    gap = np.full((half * tile, 30, 3), 255, dtype=np.uint8)
    sheet = np.concatenate([np.concatenate(rows[:half], 0), gap, np.concatenate(rows[half:], 0)], axis=1)
    Image.fromarray(sheet).save(FIGURES / "pilot_objects.jpg", quality=90)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(RESULTS / "pilot" / "metrics.csv")
    spin_ids = pd.read_csv(DATA / "abo_pilot.csv").spin_id.tolist()

    curve = chart_psnr_vs_angle(data)
    near_far = chart_near_far(data)
    example_figure(EXAMPLES, FIGURES / "pilot_examples.jpg")
    for spin_id in spin_ids:
        example_figure((spin_id,), FIGURES / f"example_{spin_id}.jpg")
    objects_sheet(spin_ids)

    ref = data[data.method == "reference"]
    by_gain = ref.groupby(["chroma_gain", "offset_deg"])[["psnr_object", "psnr", "ssim", "similarity"]].mean()
    lines = ["| Viewpoint difference | PSNR object, dB | PSNR full image, dB | SSIM | Matching similarity | PSNR object with official x1.25 boost, dB |",
             "|---|---|---|---|---|---|"]
    for deg in by_gain.loc[1.0].index:
        a, b = by_gain.loc[(1.0, deg)], by_gain.loc[(1.25, deg)]
        lines.append(f"| {deg:.0f}° | {a.psnr_object:.1f} | {a.psnr:.1f} | {a.ssim:.3f} | {a.similarity:.2f} | {b.psnr_object:.1f} |")
    lines += ["", "| Condition | PSNR object, dB | PSNR full image, dB | SSIM |", "|---|---|---|---|"]
    for name, rows in (("Grayscale input (no colour)", data[data.method == "gray"]),
                       ("Automatic colorization, no reference", data[data.method == "automatic"]),
                       ("Reference shows a different object", data[(data.method == "other_object") & (data.chroma_gain == 1.0)]),
                       ("Reference shows the same object (all angles)", ref[ref.chroma_gain == 1.0])):
        lines.append(f"| {name} | {rows.psnr_object.mean():.1f} | {rows.psnr.mean():.1f} | {rows.ssim.mean():.3f} |")
    lines += ["", "| Object | PSNR, reference within 15° | PSNR, reference 90-180° away | Drop, dB |", "|---|---|---|---|"]
    for spin_id, r in near_far.sort_values("drop", ascending=False).iterrows():
        lines.append(f"| {spin_id} | {r.near:.1f} | {r.far:.1f} | {r['drop']:.1f} |")
    gain_cost = by_gain.loc[1.0].psnr_object.mean() - by_gain.loc[1.25].psnr_object.mean()
    lines += ["", f"Mean cost of the official x1.25 chroma boost: {gain_cost:.1f} dB (PSNR object, all angles)."]
    (RESULTS / "pilot" / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
