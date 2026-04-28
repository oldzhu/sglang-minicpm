#!/usr/bin/env bash
# run_w4fp8_spike.sh — build and run the W4-FP8 SM120 spike on fcloud
# Upload bench_w4fp8_sm120.cu and this script to /root/, then `bash run_w4fp8_spike.sh`.

set -euo pipefail

SRC="${SRC:-bench_w4fp8_sm120.cu}"
OUT="${OUT:-bench_w4fp8}"
ARCH="${ARCH:-sm_120}"
NVCC="${NVCC:-nvcc}"

echo "=== nvcc version ==="
$NVCC --version
echo "=== device ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo "=== build ==="
$NVCC -arch=$ARCH -O3 -std=c++17 -lineinfo -Xptxas -v \
      "$SRC" -lcudart -o "$OUT" 2>&1 | tee build.log

echo "=== run (default M=16384 N=14336 K=4096) ==="
./"$OUT" 2>&1 | tee run_default.log

echo "=== run (small M=2048 N=4096 K=4096) ==="
./"$OUT" 2048 4096 4096 2>&1 | tee run_small.log

echo "=== run (decode-shape M=1 N=14336 K=4096 — bandwidth-bound) ==="
./"$OUT" 1 14336 4096 2>&1 | tee run_decode.log

echo "=== run (medium M=8192 N=8192 K=8192) ==="
./"$OUT" 8192 8192 8192 2>&1 | tee run_medium.log

echo "=== done ==="
echo "Logs: build.log run_default.log run_small.log run_decode.log run_medium.log"
