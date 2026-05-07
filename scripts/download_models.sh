#!/usr/bin/env bash
# Fetch the public ONNX checkpoints used by this reference implementation.
#
# These are the SCRFD-10GF detector and ArcFace ResNet50 (w600k_r50) embedder
# distributed by InsightFace as part of the buffalo_l model pack:
#   https://github.com/deepinsight/insightface/tree/master/python-package
#
# The script is idempotent; run it again to refresh missing files only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/models/onnx"
mkdir -p "${DEST}"

# Replace these URLs with the source you trust. Many users mirror buffalo_l on
# their own object storage; we leave the official InsightFace release as the
# default but encourage verifying the SHA256 sums after download.
SCRFD_URL="${SCRFD_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip}"

if [[ -f "${DEST}/scrfd_10g.onnx" && -f "${DEST}/arcface_r50.onnx" ]]; then
    echo "models already present in ${DEST}; nothing to do"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "downloading buffalo_l.zip..."
curl -fL "${SCRFD_URL}" -o "${WORK}/buffalo_l.zip"

echo "extracting..."
unzip -q "${WORK}/buffalo_l.zip" -d "${WORK}"

# buffalo_l ships with five files; we only need two.
cp "${WORK}/buffalo_l/det_10g.onnx" "${DEST}/scrfd_10g.onnx"
cp "${WORK}/buffalo_l/w600k_r50.onnx" "${DEST}/arcface_r50.onnx"

echo "done:"
ls -lh "${DEST}/scrfd_10g.onnx" "${DEST}/arcface_r50.onnx"
