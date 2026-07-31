# syntax=docker/dockerfile:1.6
#
# Two-stage build:
#   builder  - DeepStream devel image, compiles face_pipeline + tools
#   runtime  - DeepStream runtime image, just the binaries + configs

ARG DS_VERSION=8.0
ARG DS_BASE=nvcr.io/nvidia/deepstream:${DS_VERSION}-gc-triton-devel
ARG DS_RUNTIME_BASE=nvcr.io/nvidia/deepstream:${DS_VERSION}-gc-triton-devel

# ----------------------------------------------------------------------------
# Stage 1: builder
# ----------------------------------------------------------------------------
FROM ${DS_BASE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        pkg-config \
        libeigen3-dev \
        libspdlog-dev \
        libyaml-cpp-dev \
        libgtest-dev \
        libfaiss-dev \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ ./cmake/
COPY include/ ./include/
COPY src/ ./src/
COPY tools/ ./tools/

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TOOLS=ON \
        -DFACE_PIPELINE_BUILD_TESTS=OFF \
    && cmake --build build --parallel \
    && cmake --install build --prefix /opt/face_pipeline

# ----------------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------------
FROM ${DS_RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libspdlog1 \
        libyaml-cpp0.7 \
        libfaiss \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/face_pipeline /opt/face_pipeline
COPY configs/ /app/configs/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PATH="/opt/face_pipeline/bin:${PATH}"
WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--config", "/app/configs/system_config.yaml", "--pgie", "/app/configs/pgie_scrfd.txt"]
