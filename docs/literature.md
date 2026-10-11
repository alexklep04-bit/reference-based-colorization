# Literature review: automatic and reference-based image colorization

Task 1 of the term paper. Draft for Chapter 2 ("Analysis of literature and methods").
Citation style: IEEE, as required by the faculty report template.

## 1. Scope and review method

**Question.** How do automatic and reference-based (exemplar-based) colorization methods work,
how are they evaluated, and what is known about the influence of the reference image on the result?

**Sources.** Google Scholar, arXiv, IEEE Xplore and ScienceDirect were used, together with the
curated list *Awesome-Image-Colorization* [1] and the survey of Anwar et al. [2]. Starting from
the recent state-of-the-art method CRColor [3], its reference list and its citing papers were
followed in both directions.

**Keywords.** image colorization, exemplar-based colorization, reference-based colorization,
color fidelity, color transfer.

**Inclusion criteria.** Peer-reviewed papers on colorization of natural (photographic) images,
published 2002-2026, with priority given to (a) methods that became standard baselines,
(b) work published in the last three years, and (c) methods with publicly available code, since
the prototype of this term paper must reuse an existing pretrained model.

**Exclusion.** Methods limited to line art, anime or cartoons are covered only briefly, because
their inputs (clean line drawings) differ fundamentally from photographs.

## 2. Problem definition

Colorization predicts the two missing chrominance channels of a grayscale image. Methods
normally work in the CIELAB color space: the grayscale input is the lightness channel *L*, and
the network predicts the *a* and *b* channels, which guarantees that the original luminance is
preserved [2].

The task is **ill-posed**: a grayscale car can plausibly be red, blue or white, so many different
outputs are equally correct [3], [4]. This has two consequences that shape the whole field:

1. Evaluation cannot rely on pixel-wise similarity alone, because a plausible but differently
   colored result is penalised although it is not wrong.
2. Control over the output is a separate research problem, which is why methods are grouped by
   the kind of guidance the user provides.

Four families of guidance are used in the literature [1], [2]: none (automatic), user scribbles,
text descriptions, and a reference image (exemplar-based). This review focuses on the first and
the last, as required by the term paper tasks.

## 3. Automatic colorization

**Classical methods.** Early approaches propagated sparse user scribbles using local similarity
in luminance [5], which produced good results but required manual input for every image.

**Convolutional networks (2015-2017).** The first fully automatic deep methods learned color
statistics from large datasets. Iizuka et al. [6] combined local and global image features.
Larsson et al. [7] predicted per-pixel color histograms. Zhang et al. [4] treated colorization as
classification over quantised color bins and introduced class rebalancing to counter the
dominance of desaturated colors, which is the standard explanation of why regression-based
models produce brownish, washed-out results. These three works, all published in 2016, remain the
reference baselines of the field, and [4] is still commonly used as the no-reference baseline.

**Semantics and adversarial training (2018-2022).** Later work added semantic information or
generative priors: ChromaGAN [8] used adversarial training with semantic class distributions;
Su et al. [9] colorized detected object instances separately before fusing them; Wu et al. [10]
and Kim et al. [11] exploited the color prior stored in a pretrained GAN.

**Transformers and diffusion (2021-2026).** Kumar et al. [12] applied an autoregressive
transformer; ColorFormer [13] and CT2 [14] followed with attention-based architectures.
Since 2023 diffusion models dominate: DDColor [15] uses dual decoders and remains a strong,
openly available automatic baseline, while Control Color [16] and more recent work [17] add
multimodal control. The trend is towards more vivid and more controllable results rather than
towards accuracy with respect to the true colors of the depicted object.

## 4. Reference-based (exemplar-based) colorization

Here a color reference image is supplied, and its colors are transferred to the semantically
corresponding regions of the grayscale target.

**Classical methods.** Welsh et al. [18] introduced the idea, matching pixels by luminance
statistics within a neighbourhood. Irony et al. [19] added segmentation-based matching, and
Gupta et al. [20] matched superpixels using several feature types. These methods degrade quickly
when the reference and target differ in viewpoint, scale or illumination, because matching relies
on low-level features.

**Deep exemplar-based colorization.** He et al. [21] proposed the first end-to-end deep method:
a similarity sub-network aligns reference and target features, and a colorization sub-network
both transfers the matched colors and generates plausible colors where no correspondence exists.
This two-part structure (matching + generation) is still the standard design, and the paper also
provides an automatic reference retrieval algorithm. Because its code and weights are public, it
is the most frequently used baseline and the most realistic candidate for this term paper's
prototype.

**Improving the matching.** Subsequent work concentrated on better correspondence: fast
stylization-based architectures [22], attention mechanisms [23], [24], the fusion of
semantic and color information [25], and semantic-sparse matching [26]. Several methods add an
explicit semantic segmentation or classification network to constrain which regions may exchange
colors [3], [26].

**Color fidelity.** Chen et al. [3] (CRColor, *Neurocomputing*, 2026) address a different
question: whether the output actually preserves the colors of the reference. They state the core
obstacle explicitly - for a reference-target pair there is **no ground truth** colored target
image, so fidelity cannot be measured directly. Their solution is the cycle reference PSNR
(CRPSNR): the colorized target is used as the guidance to recolor the grayscale reference, and
the original reference then serves as ground truth. They further train the network with a cycle
loss and report the best CRPSNR among nine compared methods, at equal image quality (FID 3.73).

**Reduced supervision and cost.** Related work removes the need for paired training data [27] or
reduces computational cost [28], which matters for reproduction on modest hardware.

**Adjacent fields.** The same reference-following idea drives video colorization, where the
previous frame or a reference frame guides the next [29], [30], and line art colorization, where
recent diffusion models follow a reference character sheet closely [31], [32]. These fields are
outside the scope of this work but confirm that fidelity to a reference is an active concern.

## 5. Evaluation metrics and datasets

**Metrics used in the literature** [2], [3], [15]:

| Metric | Measures | Needs ground truth |
|---|---|---|
| PSNR, SSIM | pixel / structural similarity to a ground truth image | yes |
| LPIPS | perceptual similarity using deep features | yes |
| FID | realism: distance between image distributions | no (needs a reference set) |
| Top-1 / Top-5 accuracy | semantic plausibility, via a pretrained classifier | no |
| Colorfulness | saturation and color variety | no |
| Histogram Intersection Similarity (HIS) | color similarity to the reference | no |
| CRPSNR [3] | fidelity to the reference, via a recoloring cycle | no |
| User study | human preference | no |

**Datasets.** ImageNet (training; the ctest10k subset for testing), the MSCOCO validation set and
COCO-Stuff are standard [3], [15]. Anwar et al. [2] additionally published the Natural-Color
Dataset (NCD) of objects on white backgrounds, created because of the lack of ground truth
discussed below.

The dominant evaluation protocol is self-supervised: a color photograph is converted to grayscale
and the original serves as ground truth. This works for *automatic* colorization. For
*reference-based* colorization it does not, because the reference is a different image: the
protocol in [3], [21], [26] selects the reference automatically from the same ImageNet category,
which means it is **a different object of the same class** (another car, not the same car).
Consequently no ground truth exists for the colored target, which is precisely the gap CRPSNR [3]
tries to work around.

## 6. Summary and gap addressed by this work

1. Automatic methods produce plausible but arbitrary colors; they cannot recover the true color
   of a specific object, and recent research optimises realism and controllability rather than
   correctness.
2. Reference-based methods can in principle recover true colors, and their matching quality is
   the main object of study.
3. Evaluation of reference-based methods is hampered by the absence of ground truth, and the
   newest work [3] builds an indirect metric to compensate.
4. In all reviewed work the reference is a *different instance* of the same class. No reviewed
   paper evaluates the case where the reference is **the same physical object photographed from a
   different viewpoint**, and no paper reports how result quality changes as the angle between the
   reference and the target grows.

This term paper addresses point 4. Photographing one object from several angles and converting one
view to grayscale yields a **true ground truth** for the colored target, which makes direct
measurement (PSNR, SSIM) possible in a setting where the literature has only indirect metrics.

## 7. Candidate models with public code (input for tasks 2 and 4)

| Method | Type | Venue | Code |
|---|---|---|---|
| Zhang et al. [4] | automatic baseline | ECCV 2016 | public, widely packaged |
| DDColor [15] | automatic, modern | ICCV 2023 | public |
| He et al. [21] | reference-based baseline | SIGGRAPH 2018 | public |
| Xu et al. [22] | reference-based, fast | CVPR 2020 | public |
| CRColor [3] | reference-based, newest | Neurocomputing 2026 | announced ("will be published") |

## References

[1] M. Mo, "Awesome-Image-Colorization: A collection of deep learning based image and video colorization papers." [Online]. Available: https://github.com/MarkMoHR/Awesome-Image-Colorization

[2] S. Anwar, M. Tahir, C. Li, A. Mian, F. S. Khan, and A. W. Muzaffar, "Image colorization: A survey and dataset," *Information Fusion*, arXiv:2008.10774.

[3] S. Chen, M. Wang, X. Zhang, X. Li, and Y. Zhang, "CRColor: Cycle reference learning for exemplar-based image colorization," *Neurocomputing*, vol. 674, p. 132932, 2026.

[4] R. Zhang, P. Isola, and A. A. Efros, "Colorful image colorization," in *ECCV*, 2016.

[5] A. Levin, D. Lischinski, and Y. Weiss, "Colorization using optimization," *ACM Trans. Graph.*, vol. 23, no. 3, 2004.

[6] S. Iizuka, E. Simo-Serra, and H. Ishikawa, "Let there be color!: Joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification," *ACM Trans. Graph.*, vol. 35, no. 4, 2016.

[7] G. Larsson, M. Maire, and G. Shakhnarovich, "Learning representations for automatic colorization," in *ECCV*, 2016.

[8] P. Vitoria, L. Raad, and C. Ballester, "ChromaGAN: Adversarial picture colorization with semantic class distribution," in *WACV*, 2020.

[9] J.-W. Su, H.-K. Chu, and J.-B. Huang, "Instance-aware image colorization," in *CVPR*, 2020.

[10] Y. Wu, X. Wang, Y. Li, H. Zhang, X. Zhao, and Y. Shan, "Towards vivid and diverse image colorization with generative color prior," in *ICCV*, 2021.

[11] G. Kim, K. Kang, S. Kim, H. Lee, S. Kim, J. Kim, S.-H. Baek, and S. Cho, "BigColor: Colorization using a generative color prior for natural images," in *ECCV*, 2022.

[12] M. Kumar, D. Weissenborn, and N. Kalchbrenner, "Colorization transformer," in *ICLR*, 2021.

[13] X. Ji, B. Jiang, D. Luo, G. Tao, W. Chu, Z. Xie, C. Wang, and Y. Tai, "ColorFormer: Image colorization via color memory assisted hybrid-attention transformer," in *ECCV*, 2022.

[14] S. Weng, J. Sun, Y. Li, S. Li, and B. Shi, "CT2: Colorization transformer via color tokens," in *ECCV*, 2022.

[15] X. Kang, T. Yang, W. Ouyang, P. Ren, L. Li, and X. Xie, "DDColor: Towards photo-realistic image colorization via dual decoders," in *ICCV*, 2023.

[16] Z. Liang, Z. Li, S. Zhou, C. Li, and C. C. Loy, "Control color: Multimodal diffusion-based interactive image colorization," arXiv:2402.10855, 2024.

[17] "ColorFLUX: A structure-color decoupling framework for old photo colorization," in *CVPR*, 2026.

[18] T. Welsh, M. Ashikhmin, and K. Mueller, "Transferring color to greyscale images," *ACM Trans. Graph.*, vol. 21, no. 3, 2002.

[19] R. Irony, D. Cohen-Or, and D. Lischinski, "Colorization by example," in *Eurographics Symposium on Rendering*, 2005.

[20] R. K. Gupta, A. Y.-S. Chia, D. Rajan, E. S. Ng, and H. Zhiyong, "Image colorization using similar images," in *ACM Multimedia*, 2012.

[21] M. He, D. Chen, J. Liao, P. V. Sander, and L. Yuan, "Deep exemplar-based colorization," *ACM Trans. Graph.*, vol. 37, no. 4, 2018.

[22] Z. Xu, T. Wang, F. Fang, Y. Sheng, and G. Zhang, "Stylization-based architecture for fast deep exemplar colorization," in *CVPR*, 2020.

[23] H. Yin, "Yes, 'attention is all you need', for exemplar based colorization," in *ACM Multimedia*, 2021.

[24] M. G. Blanch, I. Khalifeh, N. O'Connor, and M. Mrak, "Attention-based stylisation for exemplar image colourisation," in *IEEE MMSP*, 2021.

[25] P. Lu, J. Yu, X. Peng, Z. Zhao, and X. Wang, "Gray2ColorNet: Transfer more colors from reference image," in *ACM Multimedia*, 2020.

[26] Y. Bai, C. Dong, Z. Chai, A. Wang, Z. Xu, and C. Yuan, "Semantic-sparse colorization network for deep exemplar-based colorization," in *ECCV*, 2022.

[27] W. Wang, "Unsupervised deep exemplar colorization via pyramid dual non-local attention," *IEEE Trans. Image Process.*, 2023.

[28] "Lightweight deep exemplar colorization via semantic attention-guided Laplacian pyramid," *IEEE Trans. Vis. Comput. Graph.*, 2024.

[29] B. Zhang, M. He, J. Liao, P. V. Sander, L. Yuan, A. Bermak, and D. Chen, "Deep exemplar-based video colorization," in *CVPR*, 2019.

[30] S. Iizuka and E. Simo-Serra, "DeepRemaster: Temporal source-reference attention networks for comprehensive video enhancement," *ACM Trans. Graph.*, vol. 38, no. 6, 2019.

[31] Z. Liu et al., "MangaNinja: Line art colorization with precise reference following," in *CVPR*, 2025.

[32] J. Zhu et al., "Cobra: Efficient line art colorization with broader references," in *SIGGRAPH*, 2025.

---

**Note on verification.** Author lists and venues above were taken from the curated list [1], the
survey [2] and the reference list of [3]. Before submission, every entry must be checked against
the publisher's page, and the entries marked without authors ([17], [27], [28]) completed.
