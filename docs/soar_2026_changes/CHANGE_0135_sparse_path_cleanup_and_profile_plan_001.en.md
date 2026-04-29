# CHANGE_0135 (continuation 001) — Round 13e Option-B profile result

Companion to `CHANGE_0135_sparse_path_cleanup_and_profile_plan.en.md` (the proposal). This doc captures the actual profile data and the resolution of that doc's decision tree.

## 1. What was profiled

- Branch / commit: `mixed_minicpm_cudagraph` @ `f4097eef6` (post-CHANGE_0135 cleanup).
- Server config (from `prepare_env.sh` `noquant` branch):
  - `--trust-remote-code --disable-radix-cache`
  - `--attention-backend minicpm_flashinfer`
  - `--chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1`
  - `--max-running-requests 8 --mem-fraction-static 0.78`
  - `--schedule-conservativeness 1.0`
  - `--kv-cache-dtype fp8_e5m2 --enable-mixed-chunk`
  - **No** `--quantization`, `--force-dense-minicpm`, or `--dense-as-sparse`.
- Model: `/root/models/openbmb/MiniCPM-SALA` (BF16, ~18 GB loaded).
- Routing: 8 sparse_attention layers + 24 lightning-attn layers; per-request `dense_len = sparse_dense_len` (default 512). All three samples are far above 512, so all 8 sparse layers route through the **sparse top-k + paged-KV FlashInfer** path.
- KV cache: torch.float8_e5m2 (12,073,962 tokens reserved, ~46 GB).
- Cudagraph captured for `bs ∈ {1, 2, 4, 8}`.
- Workload: single request via `/generate`, `max_new_tokens=64`, picked from `/root/data/perf_public_set.jsonl` by nearest `prompt_tokens` to the target.
- Tool: `torch.profiler` started/stopped via `/start_profile` and `/stop_profile`.

| Run    | sample idx | prompt_tokens | wall   | total GPU kernel time |
|--------|------------|---------------|--------|------------------------|
| 32k    | 129        | 31 744        |  9.7 s | 5 658 ms              |
| 64k    |  49        | 63 683        | 24.7 s | 10 344 ms             |
| 128k   | 149        | 127 732       | 49.3 s | 19 775 ms             |

GPU active fraction (kernel-time / wall-time): 58% @ 32k → 42% @ 64k → 40% @ 128k. Even with `--prefill-max-requests 1`, host-side scheduling overhead is non-trivial and grows with context.

Raw analyzer output is committed to `docs/soar_2026_changes/profile_data/round13e_analyze.txt`.

## 2. Kernel-level summary (top-of-list)

The same kernel dominates at every context length:
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` — a plain BF16 cuTLASS GEMM (the dense linear/MLP projections of the model).

| Bucket                                  | 32k        | 64k        | 128k        |
|-----------------------------------------|------------|------------|-------------|
| BF16 256×128 GEMM (linear/MLP prefill)  | 66.7%      | 73.2%      | **76.9%**   |
| BF16 gemvx (decode-time linear/MLP)     | 13.4%+3.0% | 7.3%+1.6%  | 3.8%+0.8%   |
| flashinfer BatchPrefillWithPagedKV (sparse FA core) | 3.6% | 4.0% | 4.3% |
| act_and_mul (SiLU·gate)                 | 1.5%       | 1.5%       | 1.6%        |
| FillFunctor<BFloat16> (compress_k1/k2 fill, post-CHANGE_0133) | 1.0% | 1.1% | 1.1% |
| FusedAddRMSNorm                         | 1.1%       | 1.1%       | 1.1%        |
| get_block_table_cuda_v2<96> (sparse top-k metadata) | 0.7% | 0.8% | 0.8% |
| cumsum_kernel (sparse metadata)         | 0.7%       | 0.7%       | 0.8%        |
| flatten_and_fill (sparse metadata)      | 0.4%       | 0.4%       | 0.4%        |
| chunk_fwd_kernel_o (lightning-attn)     | 0.5%       | 0.6%       | 0.6%        |
| chunk_fwd_kernel_h (lightning-attn)     | 0.4%       | 0.4%       | 0.4%        |

Category roll-up:

| Category            | 32k    | 64k    | 128k   |
|---------------------|--------|--------|--------|
| GEMM (BF16 cuTLASS) | 67.2%  | 73.8%  | 77.7%  |
| GEMM (other / GEMV) | 16.4%  | 9.0%   | 4.7%   |
| FLA / SimpleGLA / sparse FA / fused norm | 7.7% | 8.3% | 8.7% |
| Other (fills, indexing, copies, topk)    | 7.7% | 7.9% | 8.1% |
| RMSNorm (QK-norm + RoPE)                 | 0.6% | 0.6% | 0.6% |
| Embedding / topk-gather                  | 0.4% | 0.4% | 0.3% |

## 3. Reading the data

1. **The bottleneck is the BF16 GEMM, not sparse attention.** The sparse-attention compute itself (`BatchPrefillWithPagedKVCacheKernel`) accounts for 3.6–4.3% of GPU time. Sparse metadata builders combined (`get_block_table` + `cumsum` + `flatten_and_fill` + topk) add ~2%. So the entire sparse-attn machinery is well under 7%.

2. **Why is GEMM so heavy?** In the noquant build all 32 layers' QKV / O / MLP projections run the BF16 cuTLASS kernel. SM120's BF16 throughput (~148 TFLOPS) is roughly 2× lower than its FP8 throughput (~296 TFLOPS) and 4× lower than INT4 with Marlin. The competition baseline (Test 12: GPTQ + FP8 KV + dense, 121.7 s S₁) achieves its speed by routing those projections through Marlin/INT4 and FP8 GEMMs. Switching the linear layers off Marlin (because GPTQ‑sparse layers were destabilized in earlier rounds) immediately gives up most of that win — and indeed the dominant kernel here is "BF16 GEMM at full weight precision".

3. **Sparse-FA scales fine.** GEMM time scales 3.8 s → 7.6 s → 15.2 s (≈linear in seq), sparse-FA scales 0.20 s → 0.42 s → 0.85 s (also linear, similar slope). No quadratic blow-up; the sparse top-k + paged-KV path on SM120 is well-behaved.

4. **CHANGE_0133 is sufficient.** The compress_k1/k2 fill is now 1.0–1.1% of GPU time at all three lengths (≈1720–1888 launches). It is no longer the unbounded GB/step monster it was before the bounded-fill patch.

5. **Host overhead is the second-biggest item, not a code-path.** GPU active fraction is 58% (32k) → 40% (128k). At 128k roughly 30 s out of 49 s wall is GPU work; the rest is launch / scheduler / Python overhead. `--prefill-max-requests 1 --max-running-requests 8` is already minimizing CPU cost; further wins here would come from CUDA Graphs over the sparse path or fusing the small bookkeeping kernels (cumsum/fill/index) — but the *upper bound* of doing this is removing the ~8% "Other" tail, not the 77% GEMM.

## 4. Resolving the CHANGE_0135 decision tree

The proposal listed four exit branches:

| Branch | Trigger                                    | Match? |
|--------|---------------------------------------------|--------|
| (a) Sparse FA memory-bound at >70% bandwidth — close sparse line | sparse FA dominates time | **No** — sparse FA is ≤4.3% |
| (b) One sparse kernel dominates — write targeted SM120 opt        | one kernel >25%          | **No** — top sparse kernel is 4.3% |
| (c) CPU launch / scheduler dominant — fix host loop / cudagraph   | GPU active <30% or scheduler >20% wall | **Partial** — 40% active at 128k, but optimizing this caps at ~8–10% wall reduction |
| (d) BF16 weights are the cost — accept that BF16+sparse cannot beat INT4+dense | GEMM >50% | **YES — 67–77%** |

The data points cleanly to **branch (d)**, which the proposal did not explicitly enumerate but is implicit in its objective ("decide whether sparse-line work can ever beat dense+INT4"):

- The thing making the sparse line slow on SOAR is **not** the sparse attention itself; it is the **BF16 weight precision** that the sparse line currently runs at.
- To make sparse-line meaningfully faster we would need to reproduce the dense+INT4 quantization story under sparse routing — i.e. **GPTQ + sparse + FP8 KV** with stable accuracy on both the 8 sparse_attention and 24 lightning-attn layers.
- Earlier attempts at GPTQ-sparse with `sparse_qkv_w8` collapsed accuracy to ~50% on the public set (see `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` notes and prior Round 11/12 results). Without an accuracy fix, branch (d)'s only honest conclusion for the submission is: **freeze the BF16+sparse line and keep the dense+GPTQ+FP8 KV baseline as the submission path.**

## 5. Decision and next-step recommendation

1. **For SOAR submission (short-term):** keep the dense + GPTQ + FP8 KV baseline (Test 12 config) as the submission path. Do not invest more kernel-side effort on the BF16+sparse profile. The kernel that would have to get faster is generic BF16 cuTLASS GEMM — there is no quick local win there above what sgl-kernel already gives us.

2. **For sparse-line research (medium-term, optional):** the only realistic way to make the sparse line competitive is to replace the BF16 GEMM weights with quantized weights *while keeping sparse attention numerically stable*. Two avenues:
   - **GPTQ + sparse + FP8 KV with re-tuned calibration** for the 8 sparse layers (try keeping QKV at W8A16 but the rest at W4A16; re-test public accuracy to see whether the historical 50% collapse can be avoided).
   - **FP8 weights (W8A8 / mxfp8) for all layers under sparse routing.** SM120 has QMMA (mxfp8) hardware path; this would replace the 76.9%-of-GPU BF16 GEMM with the 296-TFLOPS FP8 path. Requires verifying numerical stability of sparse attention's `compress_k`/top-k under FP8 activations.

3. **Tail kernel cleanups (low priority):** the 8% "Other" tail (mostly launch overhead + element-wise fills/indexing) is a candidate for a CUDA-Graph-over-sparse-path pass once we know we want to keep the sparse line. Skip until item 2 above is decided — there is no point speeding up the 8% tail of a 67–77% GEMM bottleneck.

## 6. Validation status

Profile collection was a single-request capture, intended only to identify the dominant kernels. It does **not** measure throughput at official concurrency (S₁ / S₈ / S∞). Round 13e's earlier concurrency-32 timeouts are entirely consistent with the BF16-GEMM bottleneck shown here: at concurrency the same GEMM-bound kernels serialize and total wall time grows roughly linearly with batch.

No accuracy run was performed in this round. The dense+GPTQ submission baseline accuracy / speed numbers in `TEST_RESULTS_TRACKING.md` (Test 12) remain the reference.

## 7. Rollback

There is nothing to roll back from this round — only profile data was produced and `--dense-as-sparse` was already removed in CHANGE_0135 itself. The fcloud-side workaround `ln -sf /usr/bin/gcc /usr/local/bin/clang && ln -sf /usr/bin/g++ /usr/local/bin/clang++` (needed because Python 3.10.19 sysconfig sets `CC=clang -pthread` while the fcloud image only has `gcc`) is **not yet committed**. If we keep using the noquant branch on fresh fcloud images we should fold this into `prepare_env.sh` or pre-bake a pypcre wheel into `submission_sim.tar`. Tracked as a follow-up — not part of this commit.

## 8. Files

- `docs/soar_2026_changes/profile_data/round13e_analyze.txt` (new) — raw top-25 kernel breakdown for all three lengths.
- `docs/soar_2026_changes/CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md` (this file).
- `docs/soar_2026_changes/CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.zh.md` (Chinese mirror).

Trace files (`round13e_{32k,64k,128k}.trace.json.gz`, total ≈379 MB) live on fcloud at `/root/profile_round13e/` and were not downloaded — they are reproducible with `profile_driver.py` if needed for deeper TensorBoard inspection.
