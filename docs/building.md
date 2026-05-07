# Building

This page covers two routes: a native build on Ubuntu 22.04 with a
working CUDA / TensorRT / DeepStream installation, and a container build
that handles the toolchain for you.

## Native build (Ubuntu 22.04)

Prerequisites:

```bash
sudo apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    libeigen3-dev libspdlog-dev libyaml-cpp-dev \
    libfaiss-dev \
    libopencv-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

CUDA 12.x, TensorRT 8.6+, and DeepStream 7.x or 8.x must be installed
separately from NVIDIA. Set `TensorRT_ROOT` and `DeepStream_ROOT` if
they are not at their default locations:

```bash
export TensorRT_ROOT=/usr/local/tensorrt
export DeepStream_ROOT=/opt/nvidia/deepstream/deepstream
```

Configure and build:

```bash
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TOOLS=ON \
      -DBUILD_TESTS=OFF
cmake --build build -j
```

You will get three executables under `build/`:

- `face_server` — the main pipeline binary
- `face_enroll` — bulk index builder
- `face_benchmark` — per-stage latency probe

## Container build

The Docker recipe uses the official `nvcr.io/nvidia/deepstream` devel
image as a base, so you only need the NVIDIA Container Toolkit on the
host:

```bash
docker compose up --build
```

The image is multi-stage: the builder stage compiles everything against
the DeepStream devel image, and the runtime stage copies just the
binaries and configs into a smaller runtime image.

## Building TensorRT engines

The repo never commits `.engine` files (they are CUDA/TensorRT version
specific and would balloon the repo). Build them from the public ONNX
checkpoints:

```bash
./scripts/download_models.sh   # buffalo_l from InsightFace
./scripts/build_engines.sh     # FP16 by default
```

The container's `entrypoint.sh` will also build engines on first run if
they are missing.

## Common issues

**`fatal error: NvInfer.h: No such file or directory`** — TensorRT
headers not on the include path. Set `TensorRT_ROOT` or pass
`-DTensorRT_ROOT=/path/to/tensorrt` to CMake.

**`Could NOT find FAISS`** — install `libfaiss-dev` (Ubuntu 22.04+) or
build FAISS from source with `-DFAISS_ENABLE_GPU=ON`. Older FAISS builds
without GPU support will not work.

**`enqueueV3 failed`** — usually means the engine's expected batch shape
is not what the runtime is pushing. Check that
`pipeline.batch_size`, `detection.engine_path`'s built batch profile,
and the `batch-size` field in `configs/pgie_scrfd.txt` all agree.
