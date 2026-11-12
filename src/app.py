"""Web interface of the prototype (Gradio).

Usage:
    python src/app.py        then open http://127.0.0.1:7860

The user uploads a grayscale photograph and a colour photograph of the same object taken from
another viewpoint. The page shows the reference-based result next to the automatic colorization
(no reference), and how the reference colours were matched to the target.
"""
from __future__ import annotations

import os

import gradio as gr
import numpy as np
from matplotlib import colormaps
from PIL import Image

from refcolor.exemplar import OFFICIAL_CHROMA_GAIN
from refcolor.images import gray_to_rgb, lightness
from refcolor.paths import ROOT
from refcolor.pipeline import colorize_automatically, colorize_with_reference

EXAMPLES = ROOT / "examples"  # sample pairs from the ABO dataset (CC BY 4.0), see THIRD_PARTY_NOTICES.md


def run(target: np.ndarray | None, reference: np.ndarray | None, vivid: bool):
    if target is None:
        raise gr.Error("Upload the grayscale photograph first.")
    target = np.asarray(Image.fromarray(target).convert("RGB"))
    grayscale = gray_to_rgb(lightness(target))
    automatic = colorize_automatically(target).rgb
    if reference is None:
        return grayscale, None, automatic, None, None, "No reference given: only the automatic result is shown."

    reference = np.asarray(Image.fromarray(reference).convert("RGB"))
    result = colorize_with_reference(target, reference, chroma_gain=OFFICIAL_CHROMA_GAIN if vivid else 1.0)
    confidence = (colormaps["viridis"](np.clip(result.similarity / 0.5, 0, 1))[..., :3] * 255).astype(np.uint8)
    note = (
        f"Mean matching similarity: {result.similarity.mean():.2f}. In the pilot experiment the value was about "
        "0.33 when the reference showed the same view and about 0.19 when it showed a different object."
    )
    return grayscale, result.rgb, automatic, result.warped_rgb, confidence, note


with gr.Blocks(title="Reference-based colorization") as demo:
    gr.Markdown(
        "# Reference-based colorization\n"
        "Colorize a grayscale photograph using a colour photograph of **the same object** "
        "taken from a different viewpoint."
    )
    with gr.Row():
        target_in = gr.Image(label="1. Grayscale photograph", type="numpy", height=300)
        reference_in = gr.Image(label="2. Colour reference (same object, other viewpoint)", type="numpy", height=300)
    with gr.Row():
        vivid_in = gr.Checkbox(label="Vivid colours (x1.25, setting of the original model)", value=False)
        button = gr.Button("Colorize", variant="primary")
    with gr.Row():
        gray_out = gr.Image(label="Input as seen by the model", height=300)
        result_out = gr.Image(label="Result with the reference", height=300)
        automatic_out = gr.Image(label="Automatic result (no reference)", height=300)
    with gr.Accordion("How the reference was used", open=False):
        with gr.Row():
            warped_out = gr.Image(label="Reference colours matched to the photograph (before refinement)", height=260)
            confidence_out = gr.Image(label="Matching confidence (bright = reliable match)", height=260)
    note_out = gr.Markdown()
    gr.Examples(
        examples=[
            [str(EXAMPLES / "stool_grayscale.jpg"), str(EXAMPLES / "stool_reference_90deg.jpg")],
            [str(EXAMPLES / "sofa_grayscale.jpg"), str(EXAMPLES / "sofa_reference_60deg.jpg")],
            [str(EXAMPLES / "table_grayscale.jpg"), str(EXAMPLES / "table_reference_45deg.jpg")],
        ],
        inputs=[target_in, reference_in],
        label="Examples: the same object photographed from two viewpoints",
    )
    button.click(
        run, [target_in, reference_in, vivid_in], [gray_out, result_out, automatic_out, warped_out, confidence_out, note_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7860")))
