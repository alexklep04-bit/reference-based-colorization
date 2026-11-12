"""Network definitions of the two pretrained models used in this project.

The architectures are ported from the official implementations so that the official pretrained
weights can be loaded unchanged (the names and the order of the layers must stay identical):

* VGG19Features, WarpNet, ColorVidNet -- B. Zhang et al., "Deep Exemplar-based Video Colorization",
  CVPR 2019. https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization (MIT licence)
* ECCVGenerator -- R. Zhang, P. Isola, A. A. Efros, "Colorful Image Colorization", ECCV 2016.
  https://github.com/richzhang/colorization (BSD 2-Clause licence)

See THIRD_PARTY_NOTICES.md for the licence texts.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = sys.float_info.epsilon


# --------------------------------------------------------------------------------------------
# Reference-based model (Zhang et al., 2019)
# --------------------------------------------------------------------------------------------
def vgg_preprocess(rgb: torch.Tensor) -> torch.Tensor:
    """RGB in [0, 1] -> mean-subtracted BGR in [0, 255], as expected by the Caffe VGG-19 weights."""
    bgr = torch.cat((rgb[:, 2:3], rgb[:, 1:2], rgb[:, 0:1]), dim=1)
    mean = torch.tensor([0.40760392, 0.45795686, 0.48501961]).type_as(bgr).view(1, 3, 1, 1)
    return (bgr - mean) * 255


def feature_normalize(feature: torch.Tensor) -> torch.Tensor:
    """Scale every feature vector to unit length (L2 norm over the channel dimension)."""
    return feature / (torch.norm(feature, 2, 1, keepdim=True) + EPS)


class VGG19Features(nn.Module):
    """VGG-19 convolutional layers; returns the activations requested in out_keys."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, rgb: torch.Tensor, out_keys: list[str]) -> list[torch.Tensor]:
        out = {}
        x = vgg_preprocess(rgb)
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["r21"] = F.relu(self.conv2_1(self.pool(out["r12"])))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["r31"] = F.relu(self.conv3_1(self.pool(out["r22"])))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["r41"] = F.relu(self.conv4_1(self.pool(out["r34"])))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["r51"] = F.relu(self.conv5_1(self.pool(out["r44"])))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        return [out[key] for key in out_keys]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.InstanceNorm2d(channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1)
        self.bn2 = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prelu(self.bn1(self.conv1(self.padding1(x))))
        out = self.bn2(self.conv2(self.padding2(out)))
        return self.prelu(out + x)


class WarpNet(nn.Module):
    """Correspondence network: matches target and reference features and warps the reference colours."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        fc = self.feature_channel
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, fc, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(fc),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, fc, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(fc),
            nn.PReLU(),
        )
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, fc, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(fc),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, fc, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(fc),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.layer = nn.Sequential(ResidualBlock(fc * 4), ResidualBlock(fc * 4), ResidualBlock(fc * 4))
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.upsampling = nn.Upsample(scale_factor=4)

    def _embed(self, relu2_1, relu3_1, relu4_1, relu5_1, projection: nn.Conv2d) -> torch.Tensor:
        """Fuse four VGG levels into one zero-mean, unit-length descriptor per location."""
        f2, f3, f4, f5 = self.layer2_1(relu2_1), self.layer3_1(relu3_1), self.layer4_1(relu4_1), self.layer5_1(relu5_1)
        if f5.shape[2:] != f2.shape[2:]:
            f5 = F.pad(f5, (0, 0, 1, 1), "replicate")
        fused = self.layer(torch.cat((f2, f3, f4, f5), 1))
        emb = projection(fused).view(fused.shape[0], self.inter_channels, -1)
        emb = emb - emb.mean(dim=-1, keepdim=True)
        return emb / (torch.norm(emb, 2, 1, keepdim=True) + EPS)

    def forward(self, ref_lab, target_feats, ref_feats, temperature: float):
        """ref_lab: reference image in Lab; *_feats: normalised relu2_1..relu5_1 activations.

        Returns the reference Lab values warped onto the target grid and the similarity map
        (cosine similarity of the best match for every target location).
        """
        batch, channel, height, width = ref_lab.shape
        fh, fw = height // 4, width // 4

        theta = self._embed(*target_feats, self.theta)  # B x 256 x N
        phi = self._embed(*ref_feats, self.phi)  # B x 256 x N
        corr = torch.matmul(theta.permute(0, 2, 1), phi)  # B x N x N, cosine similarities

        similarity = corr.max(dim=-1, keepdim=True)[0].view(batch, 1, fh, fw)
        weights = F.softmax(corr / temperature, dim=-1)

        ref_small = F.avg_pool2d(ref_lab, 4).view(batch, channel, -1).permute(0, 2, 1)  # B x N x 3
        warped = torch.matmul(weights, ref_small).permute(0, 2, 1).contiguous().view(batch, channel, fh, fw)
        return self.upsampling(warped), self.upsampling(similarity)


class ColorVidNet(nn.Module):
    """Colorization network: refines the warped colours; U-Net-like with dilated convolutions."""

    def __init__(self, in_channels: int = 7) -> None:
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.InstanceNorm2d(128)
        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: target lightness (1), warped ab (2), similarity (1), previous frame Lab (3) -> ab."""
        relu = F.relu
        c1 = relu(self.conv1_2(relu(self.conv1_1(x))))
        c1n = self.conv1_2norm(c1)
        c2 = relu(self.conv2_2(relu(self.conv2_1(self.conv1_2norm_ss(c1n)))))
        c2n = self.conv2_2norm(c2)
        c3 = relu(self.conv3_3(relu(self.conv3_2(relu(self.conv3_1(self.conv2_2norm_ss(c2n)))))))
        c3n = self.conv3_3norm(c3)
        c4 = relu(self.conv4_3(relu(self.conv4_2(relu(self.conv4_1(self.conv3_3norm_ss(c3n)))))))
        c5 = relu(self.conv5_3(relu(self.conv5_2(relu(self.conv5_1(self.conv4_3norm(c4)))))))
        c6 = relu(self.conv6_3(relu(self.conv6_2(relu(self.conv6_1(self.conv5_3norm(c5)))))))
        c7 = relu(self.conv7_3(relu(self.conv7_2(relu(self.conv7_1(self.conv6_3norm(c6)))))))
        c8 = relu(self.conv8_1(self.conv7_3norm(c7)) + self.conv3_3_short(c3n))
        c8 = relu(self.conv8_3(relu(self.conv8_2(c8))))
        c9 = relu(self.conv9_1(self.conv8_3norm(c8)) + self.conv2_2_short(c2n))
        c9 = relu(self.conv9_2(c9))
        c10 = relu(self.conv10_1(self.conv9_2norm(c9)) + self.conv1_2_short(c1n))
        c10 = F.leaky_relu(self.conv10_2(c10), 0.2)
        return torch.tanh(self.conv10_ab(c10)) * 128


def lab_to_rgb_tensor(lab: torch.Tensor) -> torch.Tensor:
    """Differentiable CIELAB -> sRGB conversion for N x 3 x H x W tensors (L in [0, 100])."""
    rgb_from_xyz = torch.tensor(
        [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
    ).type_as(lab)
    x = lab.permute(0, 2, 3, 1)  # N x H x W x 3
    fy = (x[..., 0:1] + 16.0) / 116.0
    fx = x[..., 1:2] / 500.0 + fy
    fz = (fy - x[..., 2:3] / 200.0).clamp(min=0)
    f = torch.cat((fx, fy, fz), dim=3)
    xyz = torch.where(f > 0.2068966, f**3, (f - 16.0 / 116.0) / 7.787)
    xyz = xyz * torch.tensor([0.95047, 1.0, 1.08883]).type_as(lab)
    rgb = torch.matmul(xyz, rgb_from_xyz)
    rgb = torch.where(rgb > 0.0031308, 1.055 * rgb.clamp(min=1e-12) ** (1 / 2.4) - 0.055, rgb * 12.92)
    return rgb.clamp(0, 1).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------------------------
# Automatic model (Zhang et al., 2016)
# --------------------------------------------------------------------------------------------
class ECCVGenerator(nn.Module):
    """Predicts a distribution over 313 quantised ab colours per location and decodes it to ab."""

    L_CENT, L_NORM, AB_NORM = 50.0, 100.0, 110.0

    def __init__(self) -> None:
        super().__init__()

        def block(channels: list[tuple[int, int]], stride_last: int = 1, dilation: int = 1, norm: bool = True):
            layers = []
            for i, (cin, cout) in enumerate(channels):
                stride = stride_last if i == len(channels) - 1 else 1
                layers += [nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation), nn.ReLU(True)]
            if norm:
                layers.append(nn.BatchNorm2d(channels[-1][1]))
            return layers

        self.model1 = nn.Sequential(*block([(1, 64), (64, 64)], stride_last=2))
        self.model2 = nn.Sequential(*block([(64, 128), (128, 128)], stride_last=2))
        self.model3 = nn.Sequential(*block([(128, 256), (256, 256), (256, 256)], stride_last=2))
        self.model4 = nn.Sequential(*block([(256, 512), (512, 512), (512, 512)]))
        self.model5 = nn.Sequential(*block([(512, 512)] * 3, dilation=2))
        self.model6 = nn.Sequential(*block([(512, 512)] * 3, dilation=2))
        self.model7 = nn.Sequential(*block([(512, 512)] * 3))
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            *block([(256, 256), (256, 256)], norm=False),
            nn.Conv2d(256, 313, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, lightness: torch.Tensor) -> torch.Tensor:
        """lightness: N x 1 x H x W, L in [0, 100] -> ab at the same resolution."""
        x = (lightness - self.L_CENT) / self.L_NORM
        for stage in (self.model1, self.model2, self.model3, self.model4, self.model5, self.model6, self.model7, self.model8):
            x = stage(x)
        return self.upsample4(self.model_out(self.softmax(x))) * self.AB_NORM
