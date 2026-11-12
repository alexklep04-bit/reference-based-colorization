"""Locations of the project folders."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"

# Official weights of Zhang et al., "Deep Exemplar-based Video Colorization", CVPR 2019
EXEMPLAR_WEIGHTS = {
    "vgg": WEIGHTS / "exemplar" / "vgg19_conv.pth",
    "warp": WEIGHTS / "exemplar" / "nonlocal_net_iter_76000.pth",
    "color": WEIGHTS / "exemplar" / "colornet_iter_76000.pth",
}
# Official weights of Zhang et al., "Colorful Image Colorization", ECCV 2016
AUTOMATIC_WEIGHTS_URL = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"
