#!/usr/bin/env bash
# set -euo pipefail  # Disabled: crashes close the terminal before stack trace is visible

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_WHL="${SCRIPT_DIR}/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"
shopt -s nullglob
GPTQMODEL_WHEELS=("${SCRIPT_DIR}"/gptqmodel-5.7.0-*.whl)
TRANSFORMERS_WHEELS=("${SCRIPT_DIR}"/transformers-4.57.1-*.whl)
TORCHAO_WHEELS=("${SCRIPT_DIR}"/torchao-0.9.0-*.whl)
SGL_KERNEL_WHEELS=("${SCRIPT_DIR}"/sgl_kernel-*.whl "${SCRIPT_DIR}"/sgl-kernel-*.whl)
shopt -u nullglob

echo "[prepare_env] start $(date '+%F %T')"

uv pip install --no-deps -e ./sglang/python

if [[ "${#GPTQMODEL_WHEELS[@]}" -ne 1 ]]; then
	echo "[prepare_env] expected exactly one gptqmodel wheel in ${SCRIPT_DIR}, found ${#GPTQMODEL_WHEELS[@]}" >&2
	printf '  %s\n' "${GPTQMODEL_WHEELS[@]}" >&2
	exit 1
fi

if [[ "${#TRANSFORMERS_WHEELS[@]}" -ne 1 ]]; then
	echo "[prepare_env] expected exactly one transformers wheel in ${SCRIPT_DIR}, found ${#TRANSFORMERS_WHEELS[@]}" >&2
	printf '  %s\n' "${TRANSFORMERS_WHEELS[@]}" >&2
	exit 1
fi

if [[ "${#TORCHAO_WHEELS[@]}" -ne 1 ]]; then
	echo "[prepare_env] expected exactly one torchao wheel in ${SCRIPT_DIR}, found ${#TORCHAO_WHEELS[@]}" >&2
	printf '  %s\n' "${TORCHAO_WHEELS[@]}" >&2
	exit 1
fi

echo "[prepare_env] installing gptqmodel wheel: ${GPTQMODEL_WHEELS[0]}"
echo "[prepare_env] installing transformers wheel: ${TRANSFORMERS_WHEELS[0]}"
uv pip install --force-reinstall --no-deps --no-build-isolation "${TRANSFORMERS_WHEELS[0]}" -v

if [[ ! -f "${FLASH_ATTN_WHL}" ]]; then
	echo "[prepare_env] missing flash-attn wheel: ${FLASH_ATTN_WHL}" >&2
	exit 1
fi

uv pip install "${FLASH_ATTN_WHL}" --no-build-isolation -v

if [[ "${#SGL_KERNEL_WHEELS[@]}" -ne 1 ]]; then
	echo "[prepare_env] expected exactly one sgl-kernel wheel in ${SCRIPT_DIR}, found ${#SGL_KERNEL_WHEELS[@]}" >&2
	printf '  %s\n' "${SGL_KERNEL_WHEELS[@]}" >&2
	exit 1
fi

echo "[prepare_env] installing sgl-kernel wheel: ${SGL_KERNEL_WHEELS[0]}"
uv pip install --force-reinstall --no-deps "${SGL_KERNEL_WHEELS[0]}" -v

uv pip uninstall -y torchao || true
echo "[prepare_env] installing torchao wheel: ${TORCHAO_WHEELS[0]}"
uv pip install --force-reinstall --no-deps --no-build-isolation "${TORCHAO_WHEELS[0]}" -v

echo "[prepare_env] installing lightweight gptqmodel dependency: logbar"
uv pip install --force-reinstall logbar -v

echo "[prepare_env] installing lightweight gptqmodel dependency: accelerate"
uv pip install --force-reinstall --no-deps accelerate -v

echo "[prepare_env] pinning lightweight dependency: huggingface-hub==0.34.4"
uv pip install --force-reinstall "huggingface-hub==0.34.4" -v

echo "[prepare_env] installing lightweight gptqmodel dependency: threadpoolctl"
uv pip install --force-reinstall threadpoolctl -v

echo "[prepare_env] installing lightweight gptqmodel dependency: tokenicer"
uv pip install --force-reinstall tokenicer -v

echo "[prepare_env] installing lightweight gptqmodel dependency: pypcre"
uv pip install --force-reinstall pypcre -v

echo "[prepare_env] installing lightweight gptqmodel dependency: device-smi"
uv pip install --force-reinstall device-smi -v

uv pip uninstall -y gptqmodel || true
echo "[prepare_env] installing gptqmodel wheel: ${GPTQMODEL_WHEELS[0]}"
uv pip install --force-reinstall --no-deps --no-build-isolation "${GPTQMODEL_WHEELS[0]}" -v

python3 - <<'PY'
import importlib
import json

for name in ["torch", "gptqmodel", "transformers", "torchao"]:
	module = importlib.import_module(name)
	print(
		f"[prepare_env] pinned_dependency {json.dumps({'module': name, 'version': getattr(module, '__version__', 'unknown'), 'file': getattr(module, '__file__', None)}, ensure_ascii=False, sort_keys=True)}"
	)
PY

export SOAR_QUANT_MODE="${SOAR_QUANT_MODE:-gptq}"
QUANT_MODE="${SOAR_QUANT_MODE}"

export SOAR_GPTQ_CALIBRATION_FILE="${SOAR_GPTQ_CALIBRATION_FILE:-$(pwd)/perf_public_set.jsonl}"
export SOAR_GPTQ_CALIBRATION_SAMPLES="${SOAR_GPTQ_CALIBRATION_SAMPLES:-90}"
export SOAR_GPTQ_CALIBRATION_SAMPLING="${SOAR_GPTQ_CALIBRATION_SAMPLING:-stratified}"
export SOAR_GPTQ_CALIBRATION_TASK_INCLUDE="${SOAR_GPTQ_CALIBRATION_TASK_INCLUDE:-qa,mcq,cwe}"
export SOAR_GPTQ_CALIBRATION_SEED="${SOAR_GPTQ_CALIBRATION_SEED:-20260320}"
export SOAR_GPTQ_CALIBRATION_TASK_BALANCE="${SOAR_GPTQ_CALIBRATION_TASK_BALANCE:-1}"
export SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS="${SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS:-1}"
export SOAR_GPTQ_BATCH_SIZE="${SOAR_GPTQ_BATCH_SIZE:-1}"
export SOAR_GPTQ_BITS="${SOAR_GPTQ_BITS:-4}"
export SOAR_GPTQ_GROUP_SIZE="${SOAR_GPTQ_GROUP_SIZE:-128}"
export SOAR_GPTQ_MIXED_PRECISION_PRESET="${SOAR_GPTQ_MIXED_PRECISION_PRESET:-sparse_qkv_w8}"
export SOAR_GPTQ_O_PROJ_BITS="${SOAR_GPTQ_O_PROJ_BITS:-8}"
export SOAR_GPTQ_O_PROJ_GROUP_SIZE="${SOAR_GPTQ_O_PROJ_GROUP_SIZE:-128}"
export SOAR_GPTQ_SPARSE_QKV_BITS="${SOAR_GPTQ_SPARSE_QKV_BITS:-8}"
export SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE="${SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE:-128}"
export SOAR_GPTQ_SPARSE_LAYER_IDS="${SOAR_GPTQ_SPARSE_LAYER_IDS:-}"
export SOAR_GPTQ_ATTN_IMPL="${SOAR_GPTQ_ATTN_IMPL:-flash_attention_2}"
export SOAR_GPTQ_FORCE_DENSE="${SOAR_GPTQ_FORCE_DENSE:-1}"
export SOAR_GPTQ_DAMP_PERCENT="${SOAR_GPTQ_DAMP_PERCENT:-0.05}"
export SOAR_GPTQ_MSE="${SOAR_GPTQ_MSE:-0.0}"
export SOAR_TRUST_REMOTE_CODE="${SOAR_TRUST_REMOTE_CODE:-true}"
export SOAR_GPTQ_LAYER_AWARE="${SOAR_GPTQ_LAYER_AWARE:-1}"
export SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG="${SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG:-1}"
export SOAR_GPTQ_INCLUDE_MODULES="${SOAR_GPTQ_INCLUDE_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
export SOAR_GPTQ_EXCLUDE_MODULES="${SOAR_GPTQ_EXCLUDE_MODULES:-self_attn.o_gate,self_attn.z_proj}"
export SOAR_ENABLE_FUSED_QK_NORM_ROPE="${SOAR_ENABLE_FUSED_QK_NORM_ROPE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6}"

export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
export SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO="${SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO:-1}"
export SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE="${SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE:-1}"
export SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD="${SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD:-128}"
export SGLANG_FLA_CHUNK_SIZE="${SGLANG_FLA_CHUNK_SIZE:-64}"

# SOAR W4A8 #1: opt-in switch to route std-attn QKV/O + MLP linears through
# the cutlass FP8 blockwise GEMM (SM120 QMMA, 296 TF) instead of BF16 Marlin
# (148 TF). Lightning attention stays on the BF16 Marlin path. Default off
# so the baseline build is unchanged. See
# docs/soar_2026_changes/PROPOSAL_iteration_W4A8_001.{en,zh}.md.
export SOAR_W4A8_FP8_GEMM="${SOAR_W4A8_FP8_GEMM:-0}"

# SOAR CHANGE_0131: opt-in MXFP4 KV cache (--kv-cache-dtype fp4_e2m1).
# Default off (FP8 e5m2 baseline). Set SOAR_FP4_KV_CACHE=1 to enable.
# See docs/soar_2026_changes/CHANGE_0131_nvfp4_kv_p2_plumbing.{en,zh}.md.
export SOAR_FP4_KV_CACHE="${SOAR_FP4_KV_CACHE:-0}"
if [[ "$SOAR_FP4_KV_CACHE" == "1" || "$SOAR_FP4_KV_CACHE" == "true" || "$SOAR_FP4_KV_CACHE" == "TRUE" ]]; then
	KV_CACHE_DTYPE_ARG="fp4_e2m1"
else
	KV_CACHE_DTYPE_ARG="fp8_e5m2"
fi

# SOAR Round 13d: opt-in switch to drop --force-dense-minicpm so the model
# runs in native sparse mode (8 minicpm4 sparse-attention layers + 24
# lightning-attn layers). Default off = current dense submission baseline.
# Set SOAR_SPARSE_MODE=1 to enable native sparse path for retesting.
export SOAR_SPARSE_MODE="${SOAR_SPARSE_MODE:-0}"
if [[ "$SOAR_SPARSE_MODE" == "1" || "$SOAR_SPARSE_MODE" == "true" || "$SOAR_SPARSE_MODE" == "TRUE" ]]; then
	FORCE_DENSE_ARG=""
	# Sparse path is incompatible with --enable-torch-compile (CUDA graph
	# capture calls torch.cuda.get_rng_state() which fails during capture
	# in the sparse attention kernels). Drop torch compile in sparse mode.
	TORCH_COMPILE_ARGS=""
else
	FORCE_DENSE_ARG=" --force-dense-minicpm"
	TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs 8"
fi

if [[ "$QUANT_MODE" == "gptq" ]]; then
	FUSED_QK_NORM_ROPE_ARG=""
	if [[ "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "1" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "true" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "TRUE" ]]; then
		FUSED_QK_NORM_ROPE_ARG=" --enable-fused-qk-norm-rope"
	fi
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 24 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --dense-as-sparse --quantization gptq_marlin${FORCE_DENSE_ARG} --kv-cache-dtype ${KV_CACHE_DTYPE_ARG}${FUSED_QK_NORM_ROPE_ARG}${TORCH_COMPILE_ARGS} --enable-mixed-chunk"
elif [[ "$QUANT_MODE" == "fp8_blockwise" ]]; then
	# FP8 blockwise: pre-quantized offline weights (N,K) float8_e4m3fn + blockwise scales
	# Uses SM120 UMMA kernel (fp8_blockwise_scaled_mm) via weight.t() col-major zero-copy
	# No --dense-as-sparse (model is not sparse), no --enable-torch-compile (initial gate test)
	FUSED_QK_NORM_ROPE_ARG=""
	if [[ "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "1" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "true" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "TRUE" ]]; then
		FUSED_QK_NORM_ROPE_ARG=" --enable-fused-qk-norm-rope"
	fi
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --max-running-requests 24 --mem-fraction-static 0.84 --schedule-conservativeness 0.8 --quantization fp8_blockwise --force-dense-minicpm --kv-cache-dtype fp8_e5m2${FUSED_QK_NORM_ROPE_ARG} --enable-torch-compile --torch-compile-max-bs 8 --enable-mixed-chunk"
fi

# export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"

echo "[prepare_env] SOAR_QUANT_MODE=${QUANT_MODE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_FILE=${SOAR_GPTQ_CALIBRATION_FILE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_SAMPLES=${SOAR_GPTQ_CALIBRATION_SAMPLES}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_SAMPLING=${SOAR_GPTQ_CALIBRATION_SAMPLING}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=${SOAR_GPTQ_CALIBRATION_TASK_INCLUDE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_SEED=${SOAR_GPTQ_CALIBRATION_SEED}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_TASK_BALANCE=${SOAR_GPTQ_CALIBRATION_TASK_BALANCE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=${SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS}"
echo "[prepare_env] SOAR_GPTQ_BATCH_SIZE=${SOAR_GPTQ_BATCH_SIZE}"
echo "[prepare_env] SOAR_GPTQ_MIXED_PRECISION_PRESET=${SOAR_GPTQ_MIXED_PRECISION_PRESET}"
echo "[prepare_env] SOAR_GPTQ_O_PROJ_BITS=${SOAR_GPTQ_O_PROJ_BITS}"
echo "[prepare_env] SOAR_GPTQ_O_PROJ_GROUP_SIZE=${SOAR_GPTQ_O_PROJ_GROUP_SIZE}"
echo "[prepare_env] SOAR_GPTQ_SPARSE_QKV_BITS=${SOAR_GPTQ_SPARSE_QKV_BITS}"
echo "[prepare_env] SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE=${SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE}"
echo "[prepare_env] SOAR_GPTQ_SPARSE_LAYER_IDS=${SOAR_GPTQ_SPARSE_LAYER_IDS}"
echo "[prepare_env] SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG=${SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG}"
echo "[prepare_env] SOAR_GPTQ_FORCE_DENSE=${SOAR_GPTQ_FORCE_DENSE}"
echo "[prepare_env] SOAR_ENABLE_FUSED_QK_NORM_ROPE=${SOAR_ENABLE_FUSED_QK_NORM_ROPE}"
echo "[prepare_env] SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=${SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO}"
echo "[prepare_env] SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=${SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE}"
echo "[prepare_env] SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=${SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD}"
echo "[prepare_env] SGLANG_FLA_CHUNK_SIZE=${SGLANG_FLA_CHUNK_SIZE}"
echo "[prepare_env] SOAR_W4A8_FP8_GEMM=${SOAR_W4A8_FP8_GEMM}"
echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
