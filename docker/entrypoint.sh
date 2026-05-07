#!/usr/bin/env bash
set -euo pipefail

# Build TensorRT engines on first run if they are missing.
ENGINES_DIR="${ENGINES_DIR:-/app/models/engines}"
ONNX_DIR="${ONNX_DIR:-/app/models/onnx}"

if [[ ! -f "${ENGINES_DIR}/scrfd_10g_fp16.engine" ]]; then
    if [[ -f "${ONNX_DIR}/scrfd_10g.onnx" ]]; then
        echo "[entrypoint] building SCRFD engine..."
        mkdir -p "${ENGINES_DIR}"
        trtexec \
            --onnx="${ONNX_DIR}/scrfd_10g.onnx" \
            --saveEngine="${ENGINES_DIR}/scrfd_10g_fp16.engine" \
            --fp16 \
            --workspace=4096
    else
        echo "[entrypoint] WARNING: no SCRFD ONNX found at ${ONNX_DIR}/scrfd_10g.onnx"
    fi
fi

if [[ ! -f "${ENGINES_DIR}/arcface_r50_fp16.engine" ]]; then
    if [[ -f "${ONNX_DIR}/arcface_r50.onnx" ]]; then
        echo "[entrypoint] building ArcFace engine..."
        trtexec \
            --onnx="${ONNX_DIR}/arcface_r50.onnx" \
            --saveEngine="${ENGINES_DIR}/arcface_r50_fp16.engine" \
            --fp16 \
            --minShapes=input.1:1x3x112x112 \
            --optShapes=input.1:32x3x112x112 \
            --maxShapes=input.1:64x3x112x112 \
            --workspace=4096
    else
        echo "[entrypoint] WARNING: no ArcFace ONNX found at ${ONNX_DIR}/arcface_r50.onnx"
    fi
fi

exec face_server "$@"
