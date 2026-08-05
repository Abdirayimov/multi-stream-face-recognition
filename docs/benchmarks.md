# Benchmarks

The numbers in `README.md` are indicative; this page tells you how to
reproduce them and what the moving parts are.

## Hardware reference

The reference numbers were measured on:

- GPU: NVIDIA RTX 3060 (12 GB)
- CUDA: 12.x
- TensorRT: 8.6+
- DeepStream: 8.0
- Driver: 545+

Scaling to other hardware is not published here because it was not
measured. Re-run `face_benchmark` on the card you care about rather than
applying a multiplier to these figures.

## What `face_benchmark` measures

```bash
./build/face_benchmark --config configs/system_config.yaml
```

This runs a sweep over FAISS index sizes (1K, 10K, 50K), with batch
queries of 32 random L2-normalised vectors and `top_k=5`. It reports
p50 / p95 / p99 latency and a derived QPS. The vectors are random
Gaussians, so the recall numbers are meaningless; the latency numbers
are not.

## Reproducing the detector / encoder numbers

`face_benchmark` deliberately does not exercise the TRT engines because
that requires a working GPU + built engines. To get those numbers:

1. Build engines via `scripts/build_engines.sh`.
2. Use NVIDIA's own `trtexec`:

   ```bash
   trtexec --loadEngine=models/engines/scrfd_10g_fp16.engine \
           --batch=8 --warmUp=200 --iterations=2000

   trtexec --loadEngine=models/engines/arcface_r50_fp16.engine \
           --shapes=input.1:32x3x112x112 --warmUp=200 --iterations=2000
   ```

   `trtexec` reports throughput and latency percentiles directly.

## End-to-end FPS

End-to-end numbers depend on too many variables (input resolution, face
count per frame, decode load) for a single benchmark to be meaningful.
A practical recipe:

1. Generate four synthetic streams: `./scripts/generate_test_streams.sh 4`
2. Run with a wall-clock timer or `gst-launch -v` style pad probes.
3. Watch `nvidia-smi dmon -s u` to confirm you are GPU-bound rather
   than decode-bound.

If you are decode-bound, lower the source resolution or increase the
muxer width/height in `system_config.yaml` to match.

## Notes on the FAISS heuristic

The `nlist` default is `min(4 * sqrt(n), n / 39)`, which is a common
rule of thumb but not optimal for every distribution. For n in
[1K, 100K] the default is fine; for n > 1M consider pre-training the
quantizer on a representative subsample and bumping `nprobe` to keep
recall stable.
