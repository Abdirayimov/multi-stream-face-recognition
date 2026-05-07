#!/usr/bin/env bash
# Generate synthetic 30-second 720p MP4s using ffmpeg's testsrc filter.
# Used by docker-compose.yml when no real RTSP cameras are available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/synthetic"
mkdir -p "${DEST}"

N="${1:-4}"
DURATION="${2:-30}"

for i in $(seq -f "%02g" 1 "${N}"); do
    OUT="${DEST}/stream_${i}.mp4"
    if [[ -f "${OUT}" ]]; then
        echo "exists: ${OUT}"
        continue
    fi
    echo "generating: ${OUT}"
    ffmpeg -y -hide_banner -loglevel error \
        -f lavfi -i "testsrc=duration=${DURATION}:size=1280x720:rate=30" \
        -c:v libx264 -pix_fmt yuv420p -preset veryfast \
        "${OUT}"
done

echo "done. streams in ${DEST}:"
ls -lh "${DEST}/"
