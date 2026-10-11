# Report requirements and planned structure

Sources: faculty templates `reportTemplate-bachelor-EN-2021.docx`, `titlePage-IS-EN-2022-term-paper.docx`,
and an example term paper (35 pages) and bachelor thesis (38 pages) from previous years.

## Title page

Name · TITLE in capitals · "Term Paper" · "Informatics Systems study programme, state code 6121BX016" ·
"Study field Informatics" · Supervisor (degree, name, surname / signature, date) ·
"Defended prof. dr. Tomas Krilavičius, Dean of Faculty" · "Kaunas, 2026".

## Front matter

| Part | Term paper | Bachelor thesis |
|---|---|---|
| CONTENT | yes | yes |
| LIST OF ABBREVIATIONS | yes (~20 entries in the example) | yes |
| ABSTRACT (EN, with the author/title/supervisor table) | yes | yes |
| SANTRAUKA (LT) | **not present** in the example term paper | present |

So the Lithuanian summary appears only in the thesis. Confirm with the supervisor before submission.

## Formatting (from the template)

- Times New Roman. Chapter titles: CAPITALS, bold, 16 pt, 12 pt before / 6 pt after, left aligned.
- Section titles: bold, 14 pt, 12 pt before / 6 pt after, left aligned.
- Body: 12 pt, line spacing 1.5, justified, first line indent 1.27 cm, black.
- Citations: **IEEE numeric** style, `[1]`, `[2]`, `[3], [4]`. The example used 42 references.
- Tables: caption **on top** per the template (the example placed them below — follow the template),
  Times New Roman 10, no vertical lines, cite every table in the text ("Table 1 presents ...").
  Reused tables must credit the source in the caption ("Adapted from ... [24]").
- Figures: centered, caption **below**, Times New Roman 10, cite every figure in the text.
- Formulas: centered, numbered on the right, referenced by number.

## Observed conventions in the example term paper

- ~30 pages of content, 35 pages total.
- Chapters are numbered 1..7, where REFERENCES is its own numbered chapter and APPENDIXES is not numbered.
- Chapters 3-5 are topic-specific; the template's generic chapter 3 is meant to be renamed.
- The example term paper had **no implementation**: literature review + architecture analysis +
  comparison + a plan for future evaluation. This term paper also includes a prototype and
  experiments.
- Results and conclusions were written as prose. The template asks for **numbered lists**, so use
  numbered lists (safer, and easier for the committee to follow).
- Appendix contains the signed topic formalization document.

## Planned structure of this term paper

```
TITLE PAGE
CONTENT
LIST OF ABBREVIATIONS            CNN, GAN, GT, LPIPS, PSNR, SSIM, FID, CRPSNR, CIELAB, ...
ABSTRACT
1 INTRODUCTION
  1.1 Problem definition          colors of a specific object cannot be recovered automatically
  1.2 Goal of the paper           (goal from the signed topic formalization)
  1.3 Tasks                       (6 tasks from the signed topic formalization)
2 ANALYSIS OF LITERATURE AND METHODS
  2.1 Literature analysis         -> docs/literature.md (task 1)
  2.2 Methods analysis            CIELAB pipeline, matching + generation, evaluation metrics
  2.3 Comparison of existing solutions   -> docs/market-comparison.md (task 2)
3 TEST SET AND PROTOTYPE
  3.1 Test set of image pairs     collection, viewpoint grouping, ground truth (task 3)
  3.2 Prototype                   chosen pretrained model, architecture of the prototype (task 4)
  3.3 Evaluation methodology      metrics, comparison conditions (task 5)
4 RESULTS AND CONCLUSIONS
  4.1 Results                     numbered list + tables/figures
  4.2 Conclusions                 numbered list
  4.3 Plan for the bachelor thesis
5 REFERENCES                      IEEE, verified against publisher pages
APPENDIXES
  Topic formalization document    (signed scan; not stored in this repository)
```

Target length: 25-35 pages, as in the example.
