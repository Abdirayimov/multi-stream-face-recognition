#!/usr/bin/env bash
# Build TensorRT engines from the ONNX checkpoints in models/onnx.
#
# Engines are FP16 by default; pass `--int8` to enable INT8 calibration (requires
# a calibration cache; see docs/building.md).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${ROOT}/models/onnx"
ENG_DIR="${ROOT}/models/engines"
mkdir -p "${ENG_DIR}"

PRECISION="--fp16"
if [[ "${1:-}" == "--int8" ]]; then
    PRECISION="--int8"
fi

if [[ -f "${ONNX_DIR}/scrfd_10g.onnx" ]]; then
    echo "building SCRFD engine (${PRECISION})..."
    trtexec \
        --onnx="${ONNX_DIR}/scrfd_10g.onnx" \
        --saveEngine="${ENG_DIR}/scrfd_10g_fp16.engine" \
        ${PRECISION} \
        --workspace=4096
fi

if [[ -f "${ONNX_DIR}/arcface_r50.onnx" ]]; then
    echo "building ArcFace engine (${PRECISION})..."
    trtexec \
        --onnx="${ONNX_DIR}/arcface_r50.onnx" \
        --saveEngine="${ENG_DIR}/arcface_r50_fp16.engine" \
        ${PRECISION} \
        --minShapes=input.1:1x3x112x112 \
        --optShapes=input.1:32x3x112x112 \
        --maxShapes=input.1:64x3x112x112 \
        --workspace=4096
fi

echo "done:"
ls -lh "${ENG_DIR}/"
