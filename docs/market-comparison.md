# Comparison of existing colorization solutions

Task 2 of the term paper. Draft for section 2.3 ("Comparison of existing solutions").
All information was checked on 12 November 2026; prices and features of commercial tools change often.

Two questions were asked about every solution:

1. How are the colours determined (automatically, by text, by hand, from a reference image)?
2. Can the user supply a colour photograph of **the same object** as a reference?

## 1. Commercial tools and services

| Tool | Type | How colours are determined | Reference image | Price | Notes |
|---|---|---|---|---|---|
| MyHeritage In Color [1] | web service | fully automatic (licensed DeOldify technology) | no | several photos free (watermarked); unlimited with a subscription | aimed at family photographs |
| Palette.fm [2], [3] | web service + API | automatic; can be steered with a text description and colour filters | **yes** - the API has an endpoint "colorize image with reference image" | credits (about 72 USD per year); free low-resolution watermarked preview | closed model; no published evaluation of how accurately reference colours are transferred |
| Adobe Photoshop, Neural Filters "Colorize" [4] | desktop software | automatic, then manual colour "focal points" | no (colours are picked by hand) | Creative Cloud subscription | manual correction reported as unreliable in hands-on tests |
| AKVIS Coloriage AI [5] | desktop software | automatic, or manual strokes | partly: a photo can be opened as a palette to pick colours by hand | paid licence | no automatic transfer between corresponding regions |
| ColorizePhoto.com [6] | web editor | manual | partly: colours are sampled by hand from a reference picture and painted onto the photo | not checked | fully manual workflow |
| General-purpose generative image editors (Google Gemini 2.5 Flash Image, FLUX.1 Kontext and tools built on them) [7] | cloud API, web apps, plug-ins | text instruction, optionally with one or more reference images | yes (as additional input images) | pay per image (about 0.04 USD) | the whole image is generated again, so the original lightness and fine details are not guaranteed to be preserved; no published measurement of colour fidelity |

## 2. Open-source models

Licences were read from the GitHub repositories; requirements from the official README files.

| Model | Year, venue | Type | Licence | Requirements of the official code | Role in this work |
|---|---|---|---|---|---|
| Colorful Image Colorization, Zhang et al. [8] | 2016, ECCV | automatic | BSD-2-Clause | PyTorch; runs on a CPU | **no-reference baseline** |
| DeOldify [9] | 2018 | automatic | MIT (repository archived in 2024) | PyTorch / fast.ai | - |
| DDColor, Kang et al. [10] | 2023, ICCV | automatic | Apache-2.0 | PyTorch | candidate for the bachelor thesis |
| Deep Exemplar-based Colorization, He et al. [11] | 2018, SIGGRAPH | reference-based | MIT | Windows 64-bit, NVIDIA GPU (CUDA 8.0), Caffe; the matching sub-network is C++ code that must be compiled in Visual Studio | cannot be run on the available hardware |
| Deep Exemplar-based Video Colorization, Zhang et al. [12] | 2019, CVPR | reference-based | MIT | PyTorch, NVIDIA GPU + CUDA | **selected for the prototype**; runs on a laptop CPU after re-implementing the inference script |
| Stylization-based fast exemplar colorization, Xu et al. [13] | 2020, CVPR | reference-based | MIT | Ubuntu 16.04, CUDA 9.0 | - |
| Unsupervised exemplar colorization (PDNLA-Net), Wang et al. [14] | 2023, IEEE TIP | reference-based | no licence file | not checked | - |
| Control Color, Liang et al. [15] | 2024, arXiv | diffusion-based, multimodal (text, strokes, exemplar) | S-Lab licence | based on Stable Diffusion; checkpoints on Google Drive | candidate for the bachelor thesis |
| ColorMNet, Yang et al. [16] | 2024, ECCV | reference-based, video | no licence file | not checked | - |
| CRColor, Chen et al. [17] | 2026, Neurocomputing | reference-based | - | code announced in the paper; no public repository found | - |

## 3. Findings

1. **Consumer tools are automatic.** They produce plausible colours, not the true colours of the
   depicted object; corrections are made by hand (text prompts, colour points, brush strokes).
2. **Reference-image input is rare.** It is offered by the Palette.fm API and, indirectly, by
   general-purpose generative image editors. In the remaining tools a reference picture only
   serves as a palette for manual colour picking.
3. **Nobody reports colour fidelity.** No tool publishes how accurately the colours of a reference
   are reproduced, and none considers the case where the reference shows the same object from a
   different viewpoint.
4. **Hardware is the practical barrier for open-source reference-based models.** The standard
   baseline [11] needs Windows, CUDA and compiled C++ code; newer models are built for NVIDIA GPUs.
   The model of Zhang et al. [12] is pure PyTorch, which made it possible to run it on a laptop
   CPU; it is also one of the nine methods compared in the newest study [17], so results obtained
   with it can be related to the literature.

## Sources

[1] MyHeritage, "What is MyHeritage In Color". https://www.myheritage.com/help/en/articles/12852501-what-is-myheritage-in-color
[2] Palette.fm documentation, "Colorize image with reference image". https://docs.palette.fm/extra-resources/api-reference/colorize-image-with-reference-image
[3] Palette.fm documentation, "Subscription plans". https://docs.palette.fm/image-processing/subscription-plans
[4] Adobe, "Neural Filters gallery in Photoshop on desktop". https://helpx.adobe.com/photoshop/using/neural-filters-list-and-faq.html
[5] AKVIS, "Coloriage AI". https://akvis.com/en/coloriage/index.php
[6] ColorizePhoto.com, "How to colorize black and white photos". https://www.colorizephoto.com/how-to-colorize-black-and-white-photos
[7] Example of a tool built on these models: "Photo Colorizer" (accepts up to four reference images). https://www.nanobananai.ai/ai-tools/photo-colorizer
[8] https://github.com/richzhang/colorization
[9] https://github.com/jantic/DeOldify
[10] https://github.com/piddnad/DDColor
[11] https://github.com/msracver/Deep-Exemplar-based-Colorization
[12] https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization
[13] https://github.com/xuzhongyou/Colorization
[14] https://github.com/wd1511/PDNLA-Net
[15] https://github.com/ZhexinLiang/Control-Color
[16] https://github.com/yyang181/colormnet
[17] S. Chen et al., "CRColor: Cycle reference learning for exemplar-based image colorization", Neurocomputing, vol. 674, 2026.
