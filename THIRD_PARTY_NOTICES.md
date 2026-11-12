# Third-party notices

This project uses pretrained models and data created by others.

## Deep Exemplar-based Video Colorization (reference-based model)

B. Zhang, M. He, J. Liao, P. V. Sander, L. Yuan, A. Bermak, D. Chen, CVPR 2019.
https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization

The classes `VGG19Features`, `ResidualBlock`, `WarpNet`, `ColorVidNet` and the colour conversion in
`src/refcolor/nets.py`, and the inference steps in `src/refcolor/exemplar.py`, are ported from this
implementation. The pretrained weights are downloaded from the official release and are not part of
this repository.

```
MIT License

Copyright (c) 2020 zhangmozhe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Colorful Image Colorization (automatic model)

R. Zhang, P. Isola, A. A. Efros, ECCV 2016.
https://github.com/richzhang/colorization

The class `ECCVGenerator` in `src/refcolor/nets.py` is ported from this implementation. The
pretrained weights are downloaded from the official location and are not part of this repository.

```
Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Amazon Berkeley Objects (ABO) dataset

J. Collins, S. Goel, K. Deng, A. Luthra, L. Xu, E. Gundogdu, X. Zhang, T. F. Yago Vicente,
T. Dideriksen, H. Arora, M. Guillaumin, J. Malik, "ABO: Dataset and Benchmarks for Real-World 3D
Object Understanding", CVPR 2022. https://amazon-berkeley-objects.s3.amazonaws.com/index.html

The pilot experiment uses turntable ("spin") images of this dataset, licensed under the Creative
Commons Attribution 4.0 International licence (CC BY 4.0). Credit for the data: Amazon.com. Credit
for building the dataset: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora
(Amazon.com), Jasmine Collins and Jitendra Malik (UC Berkeley). The images are downloaded by
`src/prepare_abo_pilot.py` and are not part of this repository; figures in `docs/figures/` show
resized and processed versions of them.
