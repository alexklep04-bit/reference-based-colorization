# Literature review notes (task 1)

## Reference-based (exemplar-based) colorization

- **Chen et al., "CRColor: Cycle reference learning for exemplar-based image colorization", Neurocomputing 674 (2026) 132932.**
  Newest state of the art. Addresses *color fidelity*: whether the colorized image preserves the
  reference's colors. Key problem they state: there is **no ground truth** for a reference–target
  pair, so fidelity cannot be measured directly. Their workaround is CRPSNR: colorize the target,
  then recolor the grayscale reference using that result, and compare with the original reference.
  Trained on ~1.02M ImageNet images, 250k + 70k iterations, batch 16.
  References are selected automatically from the *same category* (a different object), **not** the
  same object from another viewpoint.

- He et al. (2018), Deep Exemplar-based Colorization — the standard baseline of the field.
- Zhang et al. (2019), Lu et al. (2020), Yin et al. (2021), Zhao et al. (2021), Bai et al. (2022),
  Huang et al. (2022), Yang et al. (2024), Chen et al. (2025) — comparison methods used in CRColor.

## Automatic colorization (no reference)

- Zhang et al. (2016), Colorful Image Colorization — used here as the no-reference baseline.

## Gap this term paper addresses

Existing work evaluates with references that are a *different object of the same class*, because
no ground truth exists for the colorized target. Photographing the **same object** from two
viewpoints provides true ground truth and allows measuring how quality depends on the
viewpoint difference.
