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

# === v21 default: enable Tier 1 long-context server args by default ===
# v21 packaging ships with SOAR_TIER1_LONG_CONTEXT=1 so the official launcher
# (which sources this file with no env overrides) picks up:
#   --chunked-prefill-size 65536 / --max-prefill-tokens 65536
#   --prefill-max-requests 4 / --schedule-conservativeness 0.8
# Validated 2026-05-04 (Tier1-A vs Tier1-B): zero local short-context
# regression; accuracy 78.73% identical to v20 baseline (norm 98.42%, C=0.96);
# expected upside on official long-context speed set (68% in 32K-512K range).
# To roll back to v20 byte-equivalent: export SOAR_TIER1_LONG_CONTEXT=0.
: "${SOAR_TIER1_LONG_CONTEXT:=1}"
export SOAR_TIER1_LONG_CONTEXT

# === Phase A (PROPOSAL_phase_a_nvfp4_baseline_design_20260504): NVFP4 weight
# quantization profile selector. Single env switch controls preprocess pipeline
# AND server quantization flag. Default 'gptq' = byte-equivalent to v22.
#   gptq      = current v22 baseline (sparse_qkv_w8 GPTQ W4A16, gptq_marlin loader)
#   nvfp4     = uniform NVFP4 weights via nvidia-modelopt + sglang modelopt_fp4 loader
#   nvfp4_fos = NVFP4 with FourOverSix adaptive M=6/M=4 [Phase B]
export SOAR_QUANT_PROFILE="${SOAR_QUANT_PROFILE:-gptq}"
case "$SOAR_QUANT_PROFILE" in
  gptq|nvfp4|nvfp4_fos) ;;
  *)
    echo "[prepare_env] ERROR: SOAR_QUANT_PROFILE='${SOAR_QUANT_PROFILE}' invalid (expected gptq|nvfp4|nvfp4_fos)" >&2
    exit 1
    ;;
esac
# Phase B: nvfp4_fos profile turns on the FourOverSix scale-selection patch
# inside preprocess_model.py's run_nvfp4_quantization. Server-side args are
# identical to nvfp4 (--quantization modelopt_fp4); only the on-disk weights
# differ.
if [[ "$SOAR_QUANT_PROFILE" == "nvfp4_fos" ]]; then
	export SOAR_NVFP4_FOUR_OVER_SIX="${SOAR_NVFP4_FOUR_OVER_SIX:-1}"
	# Phase B iter 2: longer calibration context for long-context model.
	# Default in preprocess_model.py is 4096 — too short to exercise
	# attention activation distribution on 32k–128k samples. Bump to 16384.
	export SOAR_NVFP4_MAX_CALIB_SEQ_LEN="${SOAR_NVFP4_MAX_CALIB_SEQ_LEN:-16384}"
	# Phase B iter 2: switch to conservative scheduling (Test 12 family)
	# to eliminate the run-to-run accuracy variance observed in iter 1
	# (75.98% vs 70.27% on the same ckpt — runaway-think on mcq under
	# aggressive scheduling). chunk=32K, prefill-max-req=1, sched-cons=1.0,
	# torch-compile-max-bs=8. Note: this overrides the default
	# SOAR_TIER1_LONG_CONTEXT=1 set above.
	# Iter 3 (NVFP4-FOS-3): respect caller-provided SOAR_TIER1_LONG_CONTEXT so we
	# can A/B iter-1 scheduling (Tier1, tcmb=24) against iter-2 (conservative,
	# tcmb=8) on the SAME ckpt to isolate scheduling vs calibration as the
	# variance source.
	export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"
	export SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
fi
# Note: SOAR_QUANT_MODE is intentionally left at its default "gptq" even when
# profile is nvfp4* \u2014 that way the gptq server-arg branch below still fires
# (it builds the canonical Tier1/dense/torch_compile arg set), and Phase A only
# swaps the --quantization flag inside that branch via QUANT_FLAG_ARG.
# preprocess_model.py reads SOAR_QUANT_PROFILE directly to pick the
# quantization function (gptq vs nvfp4); see run_nvfp4_quantization.

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

# === Phase A: install nvidia-modelopt (NVFP4 export) only when profile demands it.
# All installs use --no-deps to avoid touching our pinned torch/transformers/
# huggingface-hub. The transitive deps below are the minimum required for
# `import modelopt.torch.quantization` to work; each is also --no-deps so
# nothing else gets upgraded.
if [[ "$SOAR_QUANT_PROFILE" == "nvfp4" || "$SOAR_QUANT_PROFILE" == "nvfp4_fos" ]]; then
	echo "[prepare_env] SOAR_QUANT_PROFILE=${SOAR_QUANT_PROFILE} -> installing nvidia-modelopt (--no-deps)"
	# IMPORTANT: modelopt 0.31.0 is incompatible with torch 2.9 (imports torch.onnx._type_utils
	# which was removed in torch 2.8+). modelopt 0.43.0 is the first release compatible with
	# torch 2.9. The matching nvidia-modelopt-core (Cython kernels) tops out at 0.33.1 — that
	# combination is the verified-working set on fcloud (2026-05-04).
	# All installs use --no-deps so our pinned torch/transformers/huggingface-hub/etc are
	# untouched. Transitive deps actually exercised by `import modelopt.torch.quantization`:
	# cppimport, pulp, onnx, pydantic, rich, torchprofile (numpy/scipy/safetensors/tqdm/regex
	# are already in base image).
	uv pip install --no-deps "nvidia-modelopt==0.43.0" -v || {
		echo "[prepare_env] ERROR: nvidia-modelopt install failed — Phase A requires modelopt; rerun with SOAR_QUANT_PROFILE=gptq or fix install." >&2
		exit 1
	}
	uv pip install --no-deps "nvidia-modelopt-core[cu12]==0.33.1" -v || {
		echo "[prepare_env] ERROR: nvidia-modelopt-core install failed" >&2
		exit 1
	}
	uv pip install --no-deps "cppimport" "pulp" "onnx" "pydantic" "rich" "torchprofile" -v || {
		echo "[prepare_env] ERROR: modelopt transitive deps install failed" >&2
		exit 1
	}
	python3 - <<'PY'
import importlib, json
try:
    import modelopt
    import modelopt.torch.quantization as mtq
    print(f"[prepare_env] pinned_dependency {json.dumps({'module': 'modelopt', 'version': getattr(modelopt, '__version__', 'unknown'), 'file': getattr(modelopt, '__file__', None)}, ensure_ascii=False, sort_keys=True)}")
    assert hasattr(mtq, 'NVFP4_DEFAULT_CFG'), 'NVFP4_DEFAULT_CFG missing in installed modelopt — wrong version?'
except Exception as e:
    print(f"[prepare_env] ERROR: modelopt import-time check failed: {e!r}")
    raise SystemExit(1)
PY
fi

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

# v20: default attention backend to stock flashinfer (Round 13f-1 / Round 13f-4
# variance quantification: equivalent acc to minicpm_flashinfer baseline within
# ±1pt local noise band, +9% S1 / +8% S8 / +6% Smax speed gain).
# Override via SOAR_BACKEND_VARIANT=minicpm_flashinfer to revert to v18-line
# defaults for A/B testing.
export SOAR_BACKEND_VARIANT="${SOAR_BACKEND_VARIANT:-flashinfer}"

# CHANGE_0140 (Round 14.1): patch chat_template at preprocess time to disable
# enable_thinking for mcq prompts (detected by literal substring
# "LETTER is one of ABCD"). Targets mcq runaway-thinking failure mode.
# Set SOAR_DISABLE_MCQ_THINKING=0 to A/B against the unpatched template.
export SOAR_DISABLE_MCQ_THINKING="${SOAR_DISABLE_MCQ_THINKING:-1}"

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
	# PROPOSAL #2-A (v22): env-gated torch-compile-max-bs sweep. Default 24
	# extends compiled CUDA graph coverage to the full Smax range
	# (max-running-requests=24), eliminating eager-mode fallback for
	# bs in [9,24]. Validated on fcloud 2026-05-04 (commit 09af88b14):
	# Smax 33.62s -> 32.54s (-3.2%), acc 79.11% (stable). Set
	# SOAR_TORCH_COMPILE_MAX_BS=8 to roll back to v21 byte-equivalent.
	SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-24}"
	TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs ${SOAR_TORCH_COMPILE_MAX_BS}"
fi

if [[ "$QUANT_MODE" == "gptq" ]]; then
	FUSED_QK_NORM_ROPE_ARG=""
	if [[ "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "1" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "true" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "TRUE" ]]; then
		FUSED_QK_NORM_ROPE_ARG=" --enable-fused-qk-norm-rope"
	fi
	# CHANGE_0136: when SOAR_SPARSE_DENSE_LEN is set, the per-request
	# dense/sparse routing threshold is overridden at backend init time.
	# In that mode --force-dense-minicpm must be dropped (otherwise the
	# threshold has no effect and every request runs dense). The threshold
	# is read at runtime via os.environ inside MiniCPMAttentionBackend.
	# Also drop --enable-torch-compile: the sparse path is incompatible
	# with cudagraph capture under torch.compile (raises
	# CUDAGeneratorImpl::current_seed during CUDA graph capture from
	# hybrid_linear_attn_backend._can_use_fast_state_io). This matches the
	# SOAR_SPARSE_MODE=1 branch above.
	# Also drop --dense-as-sparse: it forces dense_len=0 inside the
	# backend ctor BEFORE our env override can take effect (the override
	# is gated by `not self.dense_as_sparse`); leaving it on would route
	# every request to the sparse path regardless of SOAR_SPARSE_DENSE_LEN.
	SDL_DENSE_AS_SPARSE_OVERRIDE=""
	if [[ -n "$SOAR_SPARSE_DENSE_LEN" ]]; then
		export SOAR_SPARSE_DENSE_LEN
		FORCE_DENSE_ARG=""
		TORCH_COMPILE_ARGS=""
		SDL_DENSE_AS_SPARSE_OVERRIDE="drop"
		echo "[prepare_env] SOAR_SPARSE_DENSE_LEN=${SOAR_SPARSE_DENSE_LEN} -> dropping --force-dense-minicpm, --dense-as-sparse, and --enable-torch-compile; per-request routing controlled by env override"
	fi
	# Round 13f-1 smoketest: SOAR_BACKEND_VARIANT=flashinfer swaps the
	# minicpm_flashinfer backend (with optional --force-dense-minicpm) for
	# the stock --attention-backend flashinfer (the official-site default).
	# Default behaviour unchanged when the env var is unset.
	BACKEND_ARG=" --attention-backend minicpm_flashinfer"
	if [[ "$SOAR_BACKEND_VARIANT" == "flashinfer" ]]; then
		BACKEND_ARG=" --attention-backend flashinfer"
		# Round 13f-2: by default keep dropping --force-dense-minicpm (Round 13f-1 behaviour).
		# Set SOAR_BACKEND_KEEP_FORCE_DENSE=1 to retain --force-dense-minicpm so
		# model_config exposes has_sparse_attention=False / sparse_layer_ids=[];
		# isolates whether those flag-gated scheduler/KV-pool paths own the
		# 2.4pt acc regression seen in Round 13f-1. See
		# docs/soar_2026_changes/PROPOSAL_round13f2_flashinfer_keep_force_dense.en.md
		if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
			# Keep FORCE_DENSE_ARG as set above (" --force-dense-minicpm")
			:
		else
			FORCE_DENSE_ARG=""
		fi
		# Round 13f-3: also gate --dense-as-sparse on KEEP_FORCE_DENSE so we can
		# isolate whether --dense-as-sparse is the missing acc lever. With
		# KEEP_FORCE_DENSE=1 the flashinfer branch becomes flag-equivalent to
		# Test 12 (which internally rewrites minicpm_flashinfer→flashinfer).
		if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
			DENSE_AS_SPARSE_ARG=" --dense-as-sparse"
		else
			DENSE_AS_SPARSE_ARG=""
		fi
		echo "[prepare_env] SOAR_BACKEND_VARIANT=flashinfer KEEP_FORCE_DENSE=${SOAR_BACKEND_KEEP_FORCE_DENSE:-0} -> stock flashinfer, FORCE_DENSE_ARG='${FORCE_DENSE_ARG}', DENSE_AS_SPARSE_ARG='${DENSE_AS_SPARSE_ARG}'"
	else
		if [[ "$SDL_DENSE_AS_SPARSE_OVERRIDE" == "drop" ]]; then
			DENSE_AS_SPARSE_ARG=""
		else
			DENSE_AS_SPARSE_ARG=" --dense-as-sparse"
		fi
	fi
	# PROPOSAL_tier1_long_context_retest_20260430: opt-in env switch to retest
	# the catalog Tier 1 best config (prefill-max-req=4, sched-cons=0.8,
	# chunk=65536) on the new long-context speed dataset. Defaults to v20
	# shipped values when SOAR_TIER1_LONG_CONTEXT is unset/0.
	if [[ "$SOAR_TIER1_LONG_CONTEXT" == "1" || "$SOAR_TIER1_LONG_CONTEXT" == "true" || "$SOAR_TIER1_LONG_CONTEXT" == "TRUE" ]]; then
		TIER1_CHUNK_SIZE="65536"
		TIER1_PREFILL_MAX_REQ="4"
		TIER1_SCHED_CONS="0.8"
		echo "[prepare_env] SOAR_TIER1_LONG_CONTEXT=1 -> chunk=${TIER1_CHUNK_SIZE}, prefill-max-req=${TIER1_PREFILL_MAX_REQ}, sched-cons=${TIER1_SCHED_CONS}"
	else
		TIER1_CHUNK_SIZE="32768"
		TIER1_PREFILL_MAX_REQ="1"
		TIER1_SCHED_CONS="1.0"
	fi
	# Phase A: when SOAR_QUANT_PROFILE selects an NVFP4 variant, swap the loader
	# flag from gptq_marlin to modelopt_fp4. All other args (Tier1, dense, KV,
	# torch_compile, mixed-chunk) stay identical to v22 baseline.
	QUANT_FLAG_ARG=" --quantization gptq_marlin"
	if [[ "$SOAR_QUANT_PROFILE" == "nvfp4" || "$SOAR_QUANT_PROFILE" == "nvfp4_fos" ]]; then
		QUANT_FLAG_ARG=" --quantization modelopt_fp4"
	fi
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache${BACKEND_ARG} --chunked-prefill-size ${TIER1_CHUNK_SIZE} --max-prefill-tokens ${TIER1_CHUNK_SIZE} --prefill-max-requests ${TIER1_PREFILL_MAX_REQ} --max-running-requests 24 --mem-fraction-static 0.84 --schedule-conservativeness ${TIER1_SCHED_CONS}${DENSE_AS_SPARSE_ARG}${QUANT_FLAG_ARG}${FORCE_DENSE_ARG} --kv-cache-dtype ${KV_CACHE_DTYPE_ARG}${FUSED_QK_NORM_ROPE_ARG}${TORCH_COMPILE_ARGS} --enable-mixed-chunk"
elif [[ "$QUANT_MODE" == "fp8_blockwise" ]]; then
	# FP8 blockwise: pre-quantized offline weights (N,K) float8_e4m3fn + blockwise scales
	# Uses SM120 UMMA kernel (fp8_blockwise_scaled_mm) via weight.t() col-major zero-copy
	# No --dense-as-sparse (model is not sparse), no --enable-torch-compile (initial gate test)
	FUSED_QK_NORM_ROPE_ARG=""
	if [[ "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "1" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "true" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "TRUE" ]]; then
		FUSED_QK_NORM_ROPE_ARG=" --enable-fused-qk-norm-rope"
	fi
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --max-running-requests 24 --mem-fraction-static 0.84 --schedule-conservativeness 0.8 --quantization fp8_blockwise --force-dense-minicpm --kv-cache-dtype fp8_e5m2${FUSED_QK_NORM_ROPE_ARG} --enable-torch-compile --torch-compile-max-bs 8 --enable-mixed-chunk"
elif [[ "$QUANT_MODE" == "noquant" ]]; then
	# SOAR Round 13e Test 1: BF16 (no quantization) + native sparse + FP8 KV.
	# Used to isolate whether the GPTQ + sparse + FP8 KV regression observed
	# in Round 13d is GPTQ-specific or a global sparse-path regression on HEAD.
	# - No --quantization (BF16 weights from MiniCPM-SALA-Copy / MiniCPM-SALA)
	# - No --force-dense-minicpm (run the 8 sparse-attention layers natively)
	# - No --enable-torch-compile (incompatible with sparse attn cudagraph capture)
	# - Smaller --max-running-requests / mem-fraction (BF16 weights are ~3-4x larger
	#   than 4-bit GPTQ weights, less HBM left for KV/activations).
	FUSED_QK_NORM_ROPE_ARG=""
	if [[ "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "1" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "true" || "$SOAR_ENABLE_FUSED_QK_NORM_ROPE" == "TRUE" ]]; then
		FUSED_QK_NORM_ROPE_ARG=" --enable-fused-qk-norm-rope"
	fi
	# NOTE: --dense-as-sparse intentionally removed (Round 13e analysis):
	#   - Under flashinfer backend it's a no-op (custom MiniCPM backend not loaded).
	#   - Under minicpm_flashinfer it forces requests with seq_len < hf_config.sparse_dense_len
	#     (default 512) through the expensive sparse top-k+sparse-FA path, which is
	#     slower than the dense FA branch they would otherwise take. Letting the
	#     model-config dense_len threshold route short requests to dense and long
	#     requests to sparse matches the mixed architecture's design intent.
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 8 --mem-fraction-static 0.78 --schedule-conservativeness 1.0 --kv-cache-dtype fp8_e5m2${FUSED_QK_NORM_ROPE_ARG} --enable-mixed-chunk"
fi

# === SOAR 2026 Phase R1a (CHANGE_0153): Medusa speculative decoding opt-in.
# Default 0 → no behavior change vs v22 baseline. When 1, append the
# verify-tree args. R1a only registers scaffolding (worker raises
# NotImplementedError until R1b lands), so SOAR_SPEC_MEDUSA=1 will fail loudly
# on server start until R1b — intentional, so accidental enablement is
# impossible during normal benchmarks.
export SOAR_SPEC_MEDUSA="${SOAR_SPEC_MEDUSA:-0}"
export SOAR_SPEC_MEDUSA_HEADS="${SOAR_SPEC_MEDUSA_HEADS:-1}"
if [[ "$SOAR_SPEC_MEDUSA" == "1" || "$SOAR_SPEC_MEDUSA" == "true" || "$SOAR_SPEC_MEDUSA" == "TRUE" ]]; then
	NUM_DRAFT_TOKENS=$(( SOAR_SPEC_MEDUSA_HEADS + 1 ))
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --speculative-algorithm MEDUSA --speculative-num-medusa-heads ${SOAR_SPEC_MEDUSA_HEADS} --speculative-num-draft-tokens ${NUM_DRAFT_TOKENS}"
	# CHANGE_0155 R1b Stage 2 §14: cuda-graph + torch.compile are now
	# re-enabled by default for the Medusa pass-through path. Earlier
	# revisions stripped them after the "shape [1] doesn't match broadcast
	# shape [7]" crash, but that crash had the same root cause as bug #2
	# (stale prefill metadata leaking into the first decode through
	# schedule_batch.prepare_for_decode's early-return on spec_algorithm
	# != NONE). MedusaWorker.forward_batch_generation now (a) re-runs
	# prepare_for_decode, and (b) permanently flips
	# batch.spec_algorithm = NONE before get_model_worker_batch(), so the
	# resulting ForwardBatch matches the captured num_tokens_per_bs=1
	# normal-decode graphs. For MEDUSA, cuda_graph_runner.get_spec_info
	# returns None (no eagle/standalone/ngram branch), so capture-time
	# graphs are byte-identical to non-spec decode graphs.
	#
	# SOAR_SPEC_MEDUSA_EAGER=1 forces eager mode (no cuda-graph, no
	# torch.compile) as a rollback toggle if a regression surfaces.
	export SOAR_SPEC_MEDUSA_EAGER="${SOAR_SPEC_MEDUSA_EAGER:-0}"
	if [[ "$SOAR_SPEC_MEDUSA_EAGER" == "1" || "$SOAR_SPEC_MEDUSA_EAGER" == "true" || "$SOAR_SPEC_MEDUSA_EAGER" == "TRUE" ]]; then
		export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS//--enable-torch-compile/}"
		export SGLANG_SERVER_ARGS="$(echo "$SGLANG_SERVER_ARGS" | sed -E 's/--torch-compile-max-bs [0-9]+//g')"
		export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --disable-cuda-graph"
	fi
fi

# === SOAR 2026 (RESEARCH_speculative_decoding_survey_001 §5.2): NGRAM opt-in.
# Free-insurance speculative path; zero training, zero submission size, zero
# GLA-fork concern (n-gram draft has no recurrent state). Default 0 keeps
# behavior byte-identical to v22 baseline. Tunable knobs use sglang's existing
# server-arg defaults; override per-test if needed.
export SOAR_SPEC_NGRAM="${SOAR_SPEC_NGRAM:-0}"
if [[ "$SOAR_SPEC_NGRAM" == "1" || "$SOAR_SPEC_NGRAM" == "true" || "$SOAR_SPEC_NGRAM" == "TRUE" ]]; then
	if [[ "$SOAR_SPEC_MEDUSA" == "1" || "$SOAR_SPEC_MEDUSA" == "true" || "$SOAR_SPEC_MEDUSA" == "TRUE" ]]; then
		echo "[prepare_env] WARNING: SOAR_SPEC_NGRAM and SOAR_SPEC_MEDUSA both set; using MEDUSA (mutually exclusive in sglang)." >&2
	else
		export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS} --speculative-algorithm NGRAM"
	fi
fi

# export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"

echo "[prepare_env] SOAR_QUANT_PROFILE=${SOAR_QUANT_PROFILE}"
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
echo "[prepare_env] SOAR_SPARSE_DENSE_LEN=${SOAR_SPARSE_DENSE_LEN:-<unset>}"
echo "[prepare_env] SOAR_BACKEND_VARIANT=${SOAR_BACKEND_VARIANT:-<unset>}"
echo "[prepare_env] SOAR_SPEC_MEDUSA=${SOAR_SPEC_MEDUSA}"
echo "[prepare_env] SOAR_SPEC_MEDUSA_HEADS=${SOAR_SPEC_MEDUSA_HEADS}"
echo "[prepare_env] SOAR_SPEC_NGRAM=${SOAR_SPEC_NGRAM}"
echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
