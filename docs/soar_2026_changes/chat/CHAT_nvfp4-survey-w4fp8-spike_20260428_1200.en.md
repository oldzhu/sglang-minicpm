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

---

## Round 11 — fcloud execution of W4-FP8 spike (2026-04-28 13:00)

### User asks
1. "Approve dense-only smoke first? ok"
2. "could u explain what is starts with --force-dense-minicpm smoke test to isolate plumbing from sparse gates. and why we need isolate plumbing from sparse gates?"
3. "Accuracy regression tolerance? >=75%"
4. "I started fcloud and you can 1. check if pytorch >> 2.8  2. test W4-FP8 spike scaffold"

### Agent answers / actions
- **Q1 explained**: `--force-dense-minicpm` flag in `prepare_env.sh` SGLANG_SERVER_ARGS routes attention through standard FlashAttention (skipping MiniCPM SALA sparse path). MXFP4 KV touches 2 independent code paths: (1) plumbing — FP4 pool, quantize/dequantize, FA descale; (2) sparse-attention — top-k scoring, compressed-K bridge, infllmv2, scratch buffers. Enabling both at once makes bug attribution impossible. Dense-only first → ~3 plumbing gates, fast iteration. Once green, re-enable sparse as separate iteration.
- **Q3 recorded**: NVFP4 KV P2 minimum accuracy threshold = 75% (vs 79.29% baseline → max 4.3pp regression).
- **fcloud env check**: torch=2.9.1+cu128 (✅ has `float4_e2m1fn_x2`), nvcc 12.8 supports `-arch=sm_120`, RTX 6000D compute_cap 12.0. All prerequisites met for both spike and NVFP4 KV.
- **Spike scaffold required 2 fixes** during build:
  1. `int4_t` is not a real type → replaced with `int` + manual 4-bit sign-extend.
  2. `cvt.rn.satfinite.e4m3x2.bf16x2` PTX form does NOT exist on SM120 → ptxas rejected. Switched to supported `cvt.rn.satfinite.e4m3x2.f32` (2 fp32 → 1 fp8x2). Dequant chain now: int4 unpack → fp32 sub-zp → fp32 fmul scale → cvt to e4m3x2.
- **Spike results** (4 shapes, 100 iters, cudaEvent timing):

| Shape | TFLOPS | % FP8 ceiling | Verdict |
|---|---|---|---|
| 16384×14336×4096 | 156.4 | 55.7% | RED |
| 2048×4096×4096 | 155.8 | 55.4% | RED |
| 1×14336×4096 (decode) | 32.6 | 11.6% | RED (BW-bound) |
| 8192×8192×8192 | 156.4 | 55.6% | RED |

  ptxas: 23 registers, 0 spills, 1 barrier — kernel is compute-peak limited. The dequant chain costs ~43% of cycles in the unfused fp32 path.
- **fcloud shutdown executed** immediately after collecting results (cost-saving rule).

### Outcomes
- **W4-FP8 dense parked (RED)**. A tuned production kernel could plausibly reach 200–240 TF (75–85%) but the 3–4 week engineering cost for ~5–12% end-to-end gain on prefill-only paths is unfavorable vs. NVFP4 KV (~3 days, ~44% KV memory savings, helps S∞ directly).
- **NVFP4 KV cache promoted to P2**. Will start dense-only smoke after user go-ahead.
- Catalog updated to mark W4-FP8 RED and NVFP4 KV P2-NEXT.

### Cross-references
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.en.md) (final, replaces template)
- [SURVEY_NVFP4_KV_P1_20260428_1130.en.md](../SURVEY_NVFP4_KV_P1_20260428_1130.en.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md) (W4-FP8 → RED, NVFP4 KV → P2-NEXT)
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu) (with cvt fix)
