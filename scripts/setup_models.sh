#!/usr/bin/env bash
# Download the official pretrained weights of the reference-based model:
# B. Zhang et al., "Deep Exemplar-based Video Colorization", CVPR 2019 (MIT licence).
# Only the three files needed for inference are kept (about 320 MB of a 770 MB archive).
# The weights of the automatic model (Zhang et al., ECCV 2016) are downloaded automatically on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/releases/download/v1.0/colorization_checkpoint.zip"
DEST="weights/exemplar"
mkdir -p "$DEST"

if [ ! -f "$DEST/colornet_iter_76000.pth" ] || [ ! -f "$DEST/nonlocal_net_iter_76000.pth" ] || [ ! -f "$DEST/vgg19_conv.pth" ]; then
    tmp="$(mktemp -d)"
    echo "Downloading the checkpoint archive (about 770 MB)..."
    curl -L --fail -o "$tmp/checkpoint.zip" "$URL"
    unzip -j -o "$tmp/checkpoint.zip" \
        "data/vgg19_conv.pth" \
        "checkpoints/video_moredata_l1/nonlocal_net_iter_76000.pth" \
        "checkpoints/video_moredata_l1/colornet_iter_76000.pth" \
        -d "$DEST"
    rm -rf "$tmp"
fi

echo "Verifying checksums..."
(cd "$DEST" && shasum -a 256 -c ../../scripts/weights.sha256)
