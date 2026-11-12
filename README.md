# Reference-Based Grayscale Image Colorization

Term paper, Faculty of Informatics, Vytautas Magnus University
BSc "Informatics Systems", 7th semester, autumn 2026
Author: Oleksandr Klepachevskyi · Supervisor: Monika Briedienė

## Title

Reference-Based Grayscale Image Colorization Using a Color Photograph of the Same Object from a Different Viewpoint

## Goal

To develop and evaluate a reference-based image colorization system that colors a grayscale
photograph using a color reference of the same object from a different viewpoint, focusing on
how viewpoint difference affects result quality.

## Tasks

1. [x] Review existing automatic and reference-based (exemplar-based) image colorization methods.
2. [x] Compare existing colorization tools and services on the market, and available open-source models.
3. [ ] Collect a small test set of image pairs: a grayscale photograph and a color reference of the same object from a different viewpoint.
4. [ ] Build a simple prototype that colorizes a grayscale image using an existing pretrained reference-based model.
5. [ ] Conduct a preliminary analysis of how viewpoint difference affects colorization quality, comparing results with and without a reference image (PSNR, SSIM and visual assessment).
6. [ ] Prepare documentation: literature and market analysis, chosen method, preliminary results, identified challenges, and plan for the bachelor thesis.

## Schedule

| Period | Activity | Result |
|---|---|---|
| Sep 22 – Oct 12 | Literature review and analysis of existing tools and models | Literature review draft; comparison of existing solutions |
| Oct 13 – Oct 26 | Selection of a pretrained model; collection of test image pairs | Chosen model; first version of the test set |
| Oct 27 – Nov 6 | First version of the prototype; first colorization experiments | Working prototype with first results |
| **Nov 7** | **Midterm presentation** | Literature review, prototype and first results |
| Nov 8 – Nov 28 | Preliminary evaluation: with vs. without a reference, different viewpoints | Result tables, visual comparison, identified challenges |
| Nov 29 – Dec 14 | Documentation, presentation slides, rehearsal | Completed term paper; ready for defense |
| Dec (TBA) | Final defense | Presentation + prototype |

## Repository structure

```
src/         experiment and prototype code
data/pairs/  test image pairs (not committed; see data/README.md)
results/     generated colorizations, metric tables (not committed)
notebooks/   exploration and Google Colab notebooks
docs/        literature review and market comparison notes
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hardware: MacBook Air M3 (16 GB). Models that require an NVIDIA GPU are run on free Google Colab.

## Status

Tasks 1-2 drafted; prototype runs locally (CLI `src/colorize.py`, web UI `src/app.py`); pilot results in `docs/pilot-results.md`. Next: own photographs for the test set (task 3).
