# CHANGE_0133 — Sparse decode `compress_k1/k2` buffer over-fill (proposal)

## Status: PROPOSAL (awaiting approval)

## Background

Round 13d/13e attempted to retest the GPTQ + FP8 KV + native-sparse path on
HEAD. Both attempts showed a catastrophic decode-time slowdown on long-context
samples (per-request decode 20–50× slower than Test 8b on commit
`9d3ecd168` from Apr 9 2026). Round 13d at concurrency=32 timed out at 1h
with ~76/150 done. Round 13e (BF16 + sparse + FP8 KV, no GPTQ) at
concurrency=32 reproduced the same symptom: server CPU was pinned at 100%
on the scheduler thread, decode throughput stayed flat around 320 tok/s,
many requests hit the eval-harness 3000 s read-timeout.

User asked to capture the live call stack to identify whether the
scheduler was in a dead loop or a slow path.

## Diagnosis steps

1. **py-spy / gdb / `cat /proc/PID/stack` were all blocked** — the fcloud
   container does not grant `CAP_SYS_PTRACE` (`CapEff: 0xa80425fb`,
   bit 19 cleared). `/proc/sys/kernel/yama/ptrace_scope = 1`, read-only.
2. **`faulthandler.enable()` is called by sglang scheduler** at
   [scheduler.py:2907](../../python/sglang/srt/managers/scheduler.py#L2907).
   This means `SIGABRT` dumps a per-thread Python traceback to stderr
   before terminating the process — a tracebackable kill.
3. Sent `kill -ABRT 4023` to the spinning scheduler PID. Server log
   captured the traceback before the process exited.

## Captured traceback

```
Fatal Python error: Aborted

Current thread 0x00007ff192ec9740 (most recent call first):
  File ".../sglang/python/sglang/srt/layers/attention/minicpm_backend.py",
       line 1761 in init_forward_metadata_replay_cuda_graph
  File ".../sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
       line 1284 in init_forward_metadata_replay_cuda_graph
  File ".../sglang/python/sglang/srt/model_executor/cuda_graph_runner.py",
       line 821 in replay_prepare
  File ".../sglang/python/sglang/srt/model_executor/cuda_graph_runner.py",
       line 847 in replay
  File ".../sglang/python/sglang/srt/model_executor/model_runner.py",
       line 2251 in _forward_raw
  File ".../sglang/python/sglang/srt/model_executor/model_runner.py",
       line 2210 in forward
  File ".../sglang/python/sglang/srt/managers/tp_worker.py",
       line 448 in forward_batch_generation
  File ".../sglang/python/sglang/srt/managers/scheduler.py",
       line 2243 in run_batch
  File ".../sglang/python/sglang/srt/managers/scheduler.py",
       line 1144 in event_loop_overlap
  ...
```

The scheduler is **not in a dead loop**. It is on the cudagraph-replay
prep critical path, doing real work each decode step — but the work is
catastrophically over-sized.

## Root cause

[`python/sglang/srt/layers/attention/minicpm_backend.py:1822-1823`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L1822-L1823),
inside `init_forward_metadata_replay_cuda_graph`, immediately after
`torch.cuda.synchronize()`:

```python
self.decode_cuda_graph_metadata["compress_k1"][
    :forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :
].fill_(float('-inf'))
self.decode_cuda_graph_metadata["compress_k2"][
    :forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :
].fill_(float('-inf'))
```

These two lines were added in commit
[`45c159187`](https://github.com/oldzhu/sglang-minicpm/commit/45c159187)
("[Fix] fix -inf to compress_k1, compress_k2 buffer for cudagraph",
Feb 10 2026, +2/-0 lines).

### Why this is catastrophic on long context

- `self.max_context_len = 524 288` (the static cap registered with the
  pool, **not** the live max sequence length of the current batch).
- `forward_batch.batch_size = 8` for our test (max-running-requests=8).
- `self.k1_kernel_stride = 16` (sparse_kernel_stride from the model
  config).
- Number of rows zeroed per decode step (k1):
  `8 × 524 288 / 16 = 262 144`.
- Each row is `[num_compress_heads × head_dim]` bf16 ≈ 4–8 KB.
- **Total bytes zeroed per decode step ≈ 1–2 GB for k1, again for k2.**
- `torch.cuda.synchronize()` is called immediately above (L1813 area),
  so the fill is not overlapped with anything; every decode step pays
  the full cost.

### Why it didn't kill Test 8b on Apr 9

Test 8b (commit `9d3ecd168`) already had this code, but ran a different
evaluation profile with much shorter contexts and lower concurrency.
The cost scales linearly with `max_context_len` (fixed) and
`batch_size`, but the *relative* cost vs the per-step useful work
explodes when:

- live `max_seq_len << max_context_len` (so the fill wastes
  `(max_context_len/max_seq_len)×` work that is never read), and
- per-step useful work shrinks because decode of long-context dominates
  attention I/O time.

For our long-context eval (32K–128K prompts, concurrency=32 → bs=8 with
24 queued), the live decode-time `max_seq_len` is the full prefilled
prompt length (32K–128K), so the fill is **4× (at 128K) to 16×
(at 32K) over-sized** every decode step. Note: `--chunked-prefill-size
32768` only caps the *prefill* chunk size; it does **not** cap
`seq_lens_cpu.max()` at decode time — by then all prefill chunks have
been written into the KV cache and `seq_lens` reflects the full prompt.

### Relationship to the Round 13d torch.compile crash (independent issue)

Round 13d also hit a different error before this fill could even run:
`RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA
graph capture` (raised inside `torch._dynamo` when sparse-attn kernels
or their decompositions touched RNG state during cudagraph capture).
That crash is **unrelated to the over-fill**: the fill is a plain
`tensor.fill_(-inf)` and uses no RNG. Dropping `--enable-torch-compile`
(commit `613ea54e4`) only sidestepped the RNG/cudagraph-capture
incompatibility; it had no effect on the fill cost.

Consequence: **fixing CHANGE_0133 will not by itself re-enable
torch.compile in sparse mode.** If we later want torch.compile back on
the sparse path, that requires a separate investigation — locate the
RNG-touching site inside the captured region (likely a dropout-style
helper that grabs RNG even when `p=0`, or a kernel calling
`torch.cuda.get_rng_state()`) and either lift it outside capture or
replace it with a deterministic non-Generator path.

## Proposed fix (single-file, low risk)

Use the live `max_len` (already computed at L1742 as
`seq_lens_cpu.max().item()`) and the captured cudagraph `bs` (already a
local at L1722) instead of `forward_batch.batch_size` and
`self.max_context_len`.

### Before (HEAD, L1822-L1823)

```python
self.decode_cuda_graph_metadata["compress_k1"][
    :forward_batch.batch_size * self.max_context_len // self.k1_kernel_stride, :, :
].fill_(float('-inf'))
self.decode_cuda_graph_metadata["compress_k2"][
    :forward_batch.batch_size * self.max_context_len // self.k2_kernel_stride, :, :
].fill_(float('-inf'))
```

### After (proposed)

```python
fill_rows_k1 = bs * (
    (max_len + self.k1_kernel_stride - 1) // self.k1_kernel_stride
)
fill_rows_k2 = bs * (
    (max_len + self.k2_kernel_stride - 1) // self.k2_kernel_stride
)
self.decode_cuda_graph_metadata["compress_k1"][:fill_rows_k1].fill_(
    float('-inf')
)
self.decode_cuda_graph_metadata["compress_k2"][:fill_rows_k2].fill_(
    float('-inf')
)
```

### Correctness reasoning

- The fill-with-`-inf` is a *defensive masking* of the compress buffer
  before the kernel writes the live compress entries; any row that the
  kernel does not subsequently overwrite remains `-inf` so it does not
  contaminate softmax.
- The compress kernel only **writes** rows in the range
  `[0, bs × ceil(seq_len_i / kernel_stride))` for each batch entry
  `i`, where `seq_len_i ≤ max_len`. Rows beyond `max_len/stride` are
  *never read* by the subsequent attention because the cu_seqlens
  pointers (`metadata.k1.cu_seqlens`, `metadata.k2.cu_seqlens` updated
  at L1825-L1826) bound the sparse top-k scoring to the live lengths.
- Therefore zeroing only `[0, bs × ceil(max_len/stride))` is *sufficient
  and safe*: the masked region exactly covers everything the kernel
  may touch in this batch.
- Using the cudagraph-captured `bs` (instead of
  `forward_batch.batch_size`) is also a correctness win: in cudagraph
  replay the captured tensors are sized at `bs`, not the live batch
  size; using `forward_batch.batch_size` was already
  capture-vs-replay-mismatch-prone (though both are 8 for
  max-running-requests=8 with no padding).

### Expected speedup

Bytes zeroed per decode step shrink by `max_context_len/max_len`:

| Live max_len | Reduction |
|--------------|-----------|
|  32 768      | **16×**   |
|  65 536      |  8×       |
| 131 072      |  4×       |
| 262 144      |  2×       |
| 524 288      |  1× (no change at the cap) |

For our public-set workload (`max_seq_len_k` observed ≈ 32K-128K during
Round 13e), this is **4–16× less GPU memory traffic per decode step**
on the cudagraph replay critical path.

## Validation plan

1. Apply the patch on `mixed_minicpm_cudagraph`.
2. Push to `minicpm-src`.
3. Sync to fcloud (no wheel rebuild — pure Python edit).
4. Restart server with `SOAR_QUANT_MODE=noquant` (BF16 + native sparse
   + FP8 KV).
5. Re-run accuracy at `--concurrency 32`.
6. **Pass criterion**: completes within the 1h harness budget AND
   accuracy ≥ 78% (Round 13d's stalled run hit 76% before the timeout
   while still partially complete; with the over-fill removed and
   long-context decode no longer bottlenecked on per-step memory
   traffic, we expect higher accuracy because more long-context
   samples will finish within the per-request 3000 s timeout).
7. If accuracy passes, run the speed-only benchmarks
   (`speed s1/s8/smax`) to compare against the dense baseline.

## Rollback

```bash
git checkout python/sglang/srt/layers/attention/minicpm_backend.py
```

(The fix is a 4-line change in one function.)

## Risks

- **None to functional correctness** (mask region exactly covers the
  region the kernel can touch).
- **None to dense path** — this code path is only reached when
  `--force-dense-minicpm` is **off** (i.e., the sparse mode), so the
  current submission baseline is untouched.
- Possible, **small** risk: if any later kernel iterates rows beyond
  `bs × ceil(max_len/stride)` without consulting cu_seqlens, it would
  read stale (non-`-inf`) entries left over from a previous larger
  batch. We will validate by running the full accuracy eval before
  using this path for any submission.

## Next steps after approval

1. Patch + push.
2. Restart fcloud, run accuracy + speed.
3. Document results in
   `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` (Test 35).
4. If sparse path becomes viable, evaluate whether to redo GPTQ with the
   same fix in place (Round 13e Test 2 — GPTQ + sparse + FP8 KV).
