# Chat log — Round 10: NVFP4 KV P1 survey + W4-FP8 spike scaffold (2026-04-28 12:00)

Continuation of W4A8 deep-dive session.

## User asks
"While I am reviewing the explain document, please do NVFP4 KV proposal (P1 survey first) and 1-day W4-FP8 spike."

## Agent actions

### 1. NVFP4 KV P1 survey (offline, no fcloud)
Searched all `nvfp4|fp4|e2m1` and `kv_cache_dtype` references in `python/sglang/srt/`.

**Findings**:
- ~80% of plumbing exists in tree: server arg `--kv-cache-dtype fp4_e2m1`, `model_runner.configure_kv_cache_dtype`, `MHATokenToKVPoolFP4`, `KVFP4QuantizeUtil` (MXFP4 quant/dequant via `@torch.compile`).
- **Important labeling correction**: sglang's `fp4_e2m1` is **MXFP4** (16-elem block, e8m0 uint8 scale), not NVFP4 (e4m3 fp32 scale). For our use this is fine — MXFP4 is hardware-friendly on SM120 and saves more memory.
- Real gaps are 100% in our custom **`minicpm_backend.py`**: 4 gates currently keyed off `kv_cache_dtype_str.startswith("fp8")` need to also fire for `fp4_e2m1`; plus `set_kv_buffer` k_scale interaction needs to pass `None` for FP4.
- Memory: MXFP4 KV is **~44% smaller than FP8** (1152 vs 2048 bytes/token/layer for our shape).
- Updated P2 plan: ~3 days (smoke-test with `--force-dense-minicpm` first to isolate plumbing from sparse-attention gates).

**Doc**: [SURVEY_NVFP4_KV_P1_20260428_1130.{en,zh}.md](../SURVEY_NVFP4_KV_P1_20260428_1130.en.md)

**3 open questions for user before P2**:
1. Is fcloud's PyTorch ≥ 2.8 (needs `torch.float4_e2m1fn_x2`)?
2. Approve P2 starting from `--force-dense-minicpm` smoke?
3. Accuracy regression tolerance (MXFP4 is more aggressive than NVFP4)?

### 2. W4-FP8 1-day spike (scaffold authored, awaiting fcloud)
Authored:
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu) — minimal hand-rolled W4 → FP8 dequant + `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` in inline PTX. Sized so gridDim×K_LOOP_ITERS produces ≈M·N·K FMAs per launch. Times via cudaEvents over 100 iters and prints TFLOPS + verdict (GREEN/YELLOW/RED) against 281 TF FP8 ceiling.
- [spike_w4fp8/run_w4fp8_spike.sh](../spike_w4fp8/run_w4fp8_spike.sh) — build with `nvcc -arch=sm_120 -O3 -std=c++17`, runs 4 shape variants (default, small, decode, medium-square).
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.{en,zh}.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md) — result template with verdict thresholds and decision tree.

**Spike scaffold caveats** (documented in result template):
- Synthetic data — measures throughput only, not correctness.
- No TMA/swizzle/pipelining — measured TFLOPS is a **lower bound** on tuned-kernel achievable.
- Per-warp single-tile inner loop matched to total FMAs — fair cost proxy, doesn't model L2 pressure.
- Dequant chain is the **most expensive** version (bf16 mul + bf16→FP8 cvt). A tuned impl could close 10–20%.

### Awaiting user
- **Approval to upload spike .cu + .sh to fcloud and run** (~1 fcloud-hour, no production impact).
- **Decision on NVFP4 KV P2 start gate** (3 questions above).

## Outcomes
- All P1 deliverables ready offline (no fcloud needed).
- W4-FP8 spike code ready to upload (no fcloud started yet — per cost-saving rule, will ask user before any fcloud action).
- Both items advanced one full step without burning fcloud time.

## Cross-references
- [SURVEY_NVFP4_KV_P1_20260428_1130.en.md](../SURVEY_NVFP4_KV_P1_20260428_1130.en.md)
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.en.md)
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu)
- [spike_w4fp8/run_w4fp8_spike.sh](../spike_w4fp8/run_w4fp8_spike.sh)
- [PROPOSAL_NVFP4_KV_CACHE_20260427_1730.en.md](../PROPOSAL_NVFP4_KV_CACHE_20260427_1730.en.md)
- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md](../PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.en.md)
