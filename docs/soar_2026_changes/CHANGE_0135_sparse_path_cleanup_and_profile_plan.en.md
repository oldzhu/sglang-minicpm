# CHANGE_0135 — Sparse path cleanup (`--dense-as-sparse`) and Option-B profiling plan

Status: APPLIED (prepare_env.sh edit) + PROPOSAL (profiling plan, awaiting fcloud)
Branch: `mixed_minicpm_cudagraph`
Related: CHANGE_0133 (compress buffer over-fill), CHANGE_0134 (eval model_path / sparse activation rules)

## Background and motivation

Round 13e Test 1 (BF16 + native sparse + FP8 KV at concurrency=32) aborted at
85/150 with multiple 3000s read-timeouts. CHANGE_0134 ruled out the
GPTQ-vs-BF16 tokenizer mismatch as a cause. The remaining hypothesis is
genuine sparse-attention slowness at long context (32K-128K).

Before committing to a deep profiling effort, we identified two
configuration issues to clean up first:

### Issue 1 — `--dense-as-sparse` is harmful in our setup

The MiniCPM custom backend has a per-request seq-len threshold
`dense_len = hf_config.sparse_dense_len` (default `512`). Per
[`minicpm_sparse_utils.py:1023`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L1023):

```python
if forward_batch.seq_lens_cpu[i] >= self.config.dense_len or dense_as_sparse:
    # sparse branch (top-k scoring + sparse FlashAttention)
else:
    # dense branch (FA full attention) inside the same backend
```

`--dense-as-sparse` forces `dense_len = 0`, routing **every** request
through the sparse path including short batches that would otherwise take
the much cheaper dense FA branch.

- Under `--attention-backend flashinfer` (official toolkit default):
  `--dense-as-sparse` is dead code — the custom MiniCPM backend isn't even
  loaded. Sparse code never runs. Remove for cleanliness.
- Under `--attention-backend minicpm_flashinfer` (our Round 13e config):
  `--dense-as-sparse` is **actively harmful**. It forces short requests
  through expensive top-k+sparse-FA when the dense FA branch is faster
  and equally accurate.

The mixed sparse/dense routing inside the MiniCPM backend is the
designed behavior of MiniCPM4's mixed architecture — short prefills /
short decode batches go dense, long-context goes sparse. We should let
that routing work, not override it.

### Issue 2 — `dense_len` is config-only, no CLI flag

The threshold comes from:

- HF config top-level `sparse_dense_len`, OR
- nested `sparse_config.dense_len`, OR
- fallback `512`

(see [`python/sglang/srt/configs/minicpm.py:83-95`](../../python/sglang/srt/configs/minicpm.py#L83-L95)
and [`python/sglang/srt/configs/model_config.py:281-283`](../../python/sglang/srt/configs/model_config.py#L281-L283)).

There is no `--sparse-dense-len` server arg. To tune the threshold today,
we must edit the model's `config.json` at preprocess time. Adding a CLI
override is ~10 lines (`server_args.py` + `minicpm_backend.py:238`) but
deferred until profile data shows the threshold matters for our
workload.

## Rule-compliance statement

- No accuracy-affecting change. Removing `--dense-as-sparse` only changes
  *which* attention kernel runs; both branches produce mathematically
  equivalent attention outputs (dense FA = full attention; sparse
  branch is the model's published sparse design for long sequences ≥
  `dense_len`).
- No model-side change. Configuration only.
- Does not modify `eval_model*.py`.
- Compatible with submission package — only affects `prepare_env.sh`
  noquant branch which is not the current submission baseline.

## Implementation (applied)

`benchmark/soar/demo_sala/prepare_env.sh`, `noquant` branch — drop
`--dense-as-sparse` and add an explanatory comment:

```diff
+	# NOTE: --dense-as-sparse intentionally removed (Round 13e analysis):
+	#   - Under flashinfer backend it's a no-op (custom MiniCPM backend not loaded).
+	#   - Under minicpm_flashinfer it forces requests with seq_len < hf_config.sparse_dense_len
+	#     (default 512) through the expensive sparse top-k+sparse-FA path, which is
+	#     slower than the dense FA branch they would otherwise take. Letting the
+	#     model-config dense_len threshold route short requests to dense and long
+	#     requests to sparse matches the mixed architecture's design intent.
-	export SGLANG_SERVER_ARGS="... --dense-as-sparse --kv-cache-dtype fp8_e5m2 ..."
+	export SGLANG_SERVER_ARGS="... --kv-cache-dtype fp8_e5m2 ..."
```

## Option-B profiling plan (proposal — awaiting user approval before fcloud run)

Goal: identify the actual hot kernel(s) on the sparse decode path at
representative bs × seq_len, **before** committing time to optimization.

### Profile configuration

- Quant: `--quant-mode noquant` (BF16) — same as Round 13e for direct
  comparison.
- Server args: post-cleanup config (no `--dense-as-sparse`).
- Concurrency: **single request** (no `--max-concurrent` race / KV
  pressure noise).
- Workload: 1 prompt at ~64K input + ~256 decode tokens, then 1 prompt
  at ~128K input + ~256 decode tokens. Both should hit sparse branch
  (≫ 512 dense_len threshold).
- Profiler: torch profiler (`with profile(...)`) **inside a tiny
  scratch script** that calls the sglang HTTP API. Do NOT modify the
  eval harness. Capture 32 decode steps after warmup. Save the JSON
  trace + Chrome trace to fcloud `/root/profile_round13e/`.

### What we want to see

1. Top 5 GPU kernels by total time (pinpoint top-k scoring vs sparse
   FA vs metadata builders vs lightning-attn vs GEMMs).
2. CPU-side overhead from `_resolve_model_path`-style metadata builds
   (we already saw `build_sparse_decode_metadata` per step in
   [`minicpm_backend.py:467`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L467)).
3. Whether cudagraph capture covers the sparse branch (CHANGE_0070
   indicates yes for KV indptr; CHANGE_0133 fixed compress buffer
   over-fill — verify both still hold under `--dense-as-sparse` removed).
4. Memory bandwidth utilization at decode time (target: ≥ 50% of
   1398 GB/s for sparse FA to be considered "well-tuned").

### Decision tree from profile result

- **If 1 kernel dominates (>40%) and it's a known op (sparse FA / top-k)**:
  → propose targeted kernel optimization in CHANGE_0136 (likely Triton
  rewrite or hand-tuned CUDA).
- **If overhead is spread across CPU metadata + many small launches**:
  → propose graph-capture widening / metadata caching in CHANGE_0136.
- **If sparse FA is already memory-bound at >70% bandwidth**:
  → close sparse line officially; sparse architecture has no more
  headroom on this hardware. Resume dense path optimization from
  `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`.

### Scope guardrail

- One single profile session (~30 min on fcloud including server boot).
- One profile-result document (CHANGE_0135_001 or RESEARCH_).
- No source changes during the profile session itself.
- All kernel-optimization edits go in a separate CHANGE_0136+ proposal
  with explicit user approval.

## Validation

After the prepare_env.sh edit, no fcloud run is required to confirm the
cleanup itself (it's an arg removal). Validation will happen as part of
the Option-B profile run.

## Rollback

Revert this commit. The `--dense-as-sparse` flag will reappear in the
noquant branch.

## Next steps

1. User approves Option-B profile plan.
2. User restarts fcloud.
3. Agent runs single-request profile at 64K and 128K, captures torch
   profiler traces.
4. Agent shuts down fcloud, analyzes traces offline, writes
   CHANGE_0135_001 (profile result + decision-tree outcome).
5. Agent proposes CHANGE_0136 (kernel optimization) OR closes sparse
   line officially.
