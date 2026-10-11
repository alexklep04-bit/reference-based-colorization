# Test set

Image pairs of the **same object** photographed from **different viewpoints**.

Layout (one folder per object):

```
data/pairs/<object_id>/
    view_000.jpg   reference view (angle 0°)
    view_045.jpg
    view_090.jpg
    view_180.jpg
    meta.json      object category, approximate angles, place, date
```

The target image is produced by converting one view to grayscale, so its original
color version is the exact ground truth for evaluation.

Images are not committed to git (size, and to keep the repository clean). Keep them
locally and back them up separately.

Privacy: blur faces and licence plates before using an image in the paper.
