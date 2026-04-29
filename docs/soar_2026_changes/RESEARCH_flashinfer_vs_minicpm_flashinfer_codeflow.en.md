# RESEARCH — `flashinfer` vs `minicpm_flashinfer` Backend Code-Flow Comparison

**Date**: 2026-04-29
**Trigger**: Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`) showed a real
~7% speed gain but lost 2.4 acc points (76.91% vs Test 12 79.29% → C=0). User
asked: *is this just incompatibility with SALA's mixed sparse/dense layers, or
is there a knob to recover accuracy and turn this into the next baseline?*

This document maps the actual code paths of the two backend strings end-to-end
so we can pinpoint the behavioral delta that costs 2.4 acc points.

## TL;DR

1. The two strings register **different backend classes**, not just kernel toggles.
2. `--attention-backend minicpm_flashinfer` →
   `attention_registry.create_minicpm_flashinfer_backend` → instantiates
   `MiniCPMSparseBackend` (sparse-aware, custom).
3. `--attention-backend flashinfer` → instantiates the stock
   `FlashInferAttnBackend` (no sparse routing, no compress KV cache).
4. With `--force-dense-minicpm`, **both** strings end up using
   `FlashInferAttnBackend` because `server_args.py:1525` rewrites
   `minicpm_flashinfer → flashinfer`. **This is exactly Test 12 baseline.**
5. So Test 12 (79.29%) and Round 13f-1 (76.91%) **both run stock FlashInfer
   for std-attn**. The difference between them is **not** the std-attn kernel.
   It is in the *peripheral* config knobs that flip together with
   `--force-dense-minicpm`. The most likely accuracy-relevant ones:
   - `model_config.has_sparse_attention` flips True (no force) ↔ False (force).
   - `model_config.sparse_layer_ids` flips populated ↔ empty.
   - Lightning-mixer `recurrent_threshold` flips 64 (no force) ↔ 128 (force).
   - `--dense-as-sparse` is independently dropped in Round 13f-1.

## Backend registration (where the divergence starts)

`python/sglang/srt/layers/attention/attention_registry.py`:

```python
@register_attention_backend("minicpm_flashattn")     # line 180
def create_minicpm_flashattn_backend(runner):
    from sglang.srt.layers.attention.minicpm_backend import MiniCPMSparseBackend
    return MiniCPMSparseBackend(runner)

@register_attention_backend("minicpm_flashinfer")    # line 190
def create_minicpm_flashinfer_backend(runner):
    from sglang.srt.layers.attention.minicpm_backend import MiniCPMSparseBackend
    return MiniCPMSparseBackend(runner)
```

```python
@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    return FlashInferAttnBackend(runner)
```

Inside `MiniCPMSparseBackend.__init__` the **string** is read again to choose
the dense FA implementation between flash_attn and flashinfer:

```python
elif attention_backend == "minicpm_flashinfer":      # minicpm_backend.py:366
    self.use_flashinfer = True
```

So the two `minicpm_*` strings differ from each other only in the kernel
choice **inside** the sparse backend. The `flashinfer` (without `minicpm_`)
string is structurally different — a separate backend class.

## How the model wires it up

`MiniCPMSALAForCausalLM` is a hybrid model:

| Layer family | Construction (minicpm.py) | Backend used at runtime |
|--------------|---------------------------|-------------------------|
| Std attention (`MiniCPMAttention`, line 246) | `RadixAttention(...)` | `forward_batch.attn_backend.full_attn_backend.forward_extend / forward_decode` |
| Lightning mixer (`MiniCPMLightningMixer`, line 569) | Direct CUDA kernel via `SimpleGLAAttnBackend` | `forward_batch.attn_backend.linear_attn_backend.*` |

`forward_batch.attn_backend` is `HybridLinearAttnBackend`, constructed by
`attn_backend_wrapper` in `attention_registry.py`. It holds two children:

- `full_attn_backend = full_attn_backend` ← passed in by the wrapper, this is
  whatever `attention_backend` registered (so either `MiniCPMSparseBackend` or
  `FlashInferAttnBackend`).
- `linear_attn_backend = SimpleGLAAttnBackend(...)`.

The model itself (`minicpm.py:279`) is **agnostic**: it only calls
`self.attn(q, k, v, forward_batch)`. Whether that std-attn layer is a
"sparse" layer in SALA's sense is **decided by the backend**, by inspecting
`model_runner.model_config.sparse_layer_ids` at construction time.

## End-to-end forward path

### Path A — `--attention-backend minicpm_flashinfer`, no `--force-dense-minicpm`

```
Scheduler → ModelRunner.forward
  → MiniCPMSALAForCausalLM.forward
      → MiniCPMDecoderLayer.forward
          ├── self_attn = MiniCPMAttention (std)        ─┐
          │     RadixAttention.forward                   │
          │       forward_batch.attn_backend.forward     │
          │         = HybridLinearAttnBackend.forward    │
          │             dispatch by layer_type           │
          │               → full_attn_backend.forward    │
          │                  = MiniCPMSparseBackend      │
          │                       per-request decision:  │
          │                         seq_lens >= dense_len?
          │                           yes → top-k sparse FA  ◄──── sparse routing
          │                                  + compress k1/k2
          │                                  + sparse_page_table
          │                           no  → dense FA fallback     ◄──── full attn
          │                                  via flashinfer kernel
          │                                  (sparse_page_table copy → BUG #0137)
          │
          └── self_attn = MiniCPMLightningMixer          ─┐
                forward_batch.attn_backend.linear_attn… ─┘
```

Key features only present on this path:
- `compress_k1/k2` cache (mean-pooled K) used to compute top-k page selection.
- `sparse_page_table` per request.
- `cudagraph` is captured per (decode bs, head_group_num, sparse_topk*block_size)
  shape — separate from stock flashinfer's capture.
- Exposed on HEAD by latent bugs (CHANGE_0133, CHANGE_0137 above).

### Path B — `--attention-backend flashinfer` (Round 13f-1)

```
Scheduler → ModelRunner.forward
  → MiniCPMSALAForCausalLM.forward
      → MiniCPMDecoderLayer.forward
          ├── self_attn = MiniCPMAttention (std)        ─┐
          │     RadixAttention.forward                   │
          │       forward_batch.attn_backend.forward     │
          │         = HybridLinearAttnBackend.forward    │
          │             → full_attn_backend.forward      │
          │                  = stock FlashInferAttnBackend
          │                     prefill: BatchPrefillWith*KVCacheKernel
          │                     decode:  BatchDecodeWithPagedKVCacheKernel
          │                     **no sparse routing, no top-k**
          │                     **every layer = full dense attention**
          │
          └── self_attn = MiniCPMLightningMixer          ─┐
                forward_batch.attn_backend.linear_attn… ─┘
```

Key features (relative to Path A):
- `MiniCPMSparseBackend` not loaded; `compress_k1/k2` cache not allocated.
- `sparse_page_table` does not exist.
- All std-attn layers run as **full** dense attention. SALA's sparse-trained
  layers see un-masked attention scores (no top-k filter).
- This is mathematically a **superset** of top-k sparse attention.
  Information-theoretically the layer has *more* context, not less. So why
  does accuracy drop? See discussion below.

### Path C — `--attention-backend minicpm_flashinfer --force-dense-minicpm` (Test 12 baseline)

```
server_args.py:1521-1525:
    if force_dense_minicpm and attention_backend == "minicpm_flashinfer":
        attention_backend = "flashinfer"

→ effectively becomes Path B at runtime.
```

PLUS:
- `model_config.has_sparse_attention` → `False`.
- `model_config.sparse_layer_ids` → `[]`.
- `default_recurrent_threshold = 128` (vs 64 in Path B).

So **Path B and Path C use the same std-attn kernel** (stock flashinfer's
`BatchPrefillWith*KVCacheKernel`), but **differ** in:
1. Lightning mixer recurrent vs chunk-mode threshold (64 vs 128).
2. Whether the model_config exposes `sparse_layer_ids`. (The attn backend
   doesn't read it once we're on stock flashinfer; but other code paths
   might — e.g., KV cache layout, weight loader.)
3. Whether `--dense-as-sparse` is in `SGLANG_SERVER_ARGS`. Test 12 keeps
   it; Round 13f-1 drops it. With stock flashinfer this flag is unused by
   the attn path but is still parsed by `server_args` and may flow to other
   subsystems.

## Why Path B (Round 13f-1) loses 2.4 acc points despite running the same
kernel as Path C (Test 12)

The std-attn kernel is identical. The accuracy delta therefore must come from
the peripheral knobs that flip with `force_dense_minicpm`:

### Hypothesis 1 (most likely) — Lightning-mixer recurrent threshold

`hybrid_linear_attn_backend.py:1484`:
```python
default_recurrent_threshold = 128 if self.force_dense_minicpm else 64
```

The lightning mixer has two implementations:
- **chunk** mode (used when seq_len ≥ threshold) — chunked SimpleGLA, faster
  on long seqs but accumulates state via chunked-causal kernel.
- **recurrent** mode (used when seq_len < threshold) — element-wise
  recurrent state update, exact match to training math.

Path B threshold = 64, Path C threshold = 128. Many eval prompts have
extend_seq_len in [64, 128) (e.g., short mcq prompts ~60-100 tokens, qa
short answers, decode-step batches with running > 64). On Path B these run
**chunk** mode; on Path C they run **recurrent** mode. The two are
mathematically equivalent in float32 but **diverge in low precision**
because chunk-mode's blockwise reductions accumulate FP errors differently.

This is the **strongest candidate** for a 2.4 acc-point drop on a 150-sample
benchmark, particularly on tasks like cwe (which lost most: 82.33% on Path B
vs ≥87% expected) and qa (56.67% vs 63.33% on Test 20). cwe-style retrieval
is highly sensitive to attention-state precision.

### Hypothesis 2 — `has_sparse_attention=True` side-effects

If `has_sparse_attention` is True, the model loader / KV cache might:
- Allocate a different KV layout for sparse layers (e.g., extra room for
  compress_k1/k2). With stock flashinfer not consuming compress, the cache
  is just unused — should be harmless.
- Trigger a different rope / scaling path. Need to grep model code for
  `sparse_layer_ids` usage at construction time.

This is a **secondary** candidate — could be confirmed/ruled out cheaply by
forcing `has_sparse_attention=False` while leaving `attention_backend=flashinfer`.

### Hypothesis 3 — `--dense-as-sparse` removal exposes a config drift

`--dense-as-sparse` flag is parsed by `server_args` and influences
`MiniCPMSparseBackend.__init__` (forces dense_len=0). Stock flashinfer
doesn't use this. Removing it should be a no-op on Path B. **Probably not
the culprit.**

### Hypothesis 4 — KV-cache pool layout differs

When `attention_backend=flashinfer` (Path B and Path C both), the KV pool
type is determined by `kv_cache_dtype=fp8_e5m2` and the backend's preferred
layout. The two paths choose the same pool. **Probably not the culprit.**

## Suggested experiments (cheap, parallelizable)

The user explicitly wants Round 13f-1 evolved into a viable submission
baseline. The path forward is to find the smallest configuration delta on
top of Path B that recovers Test-12-class accuracy while keeping the speed gain.

| # | Experiment | What it tests | Expected outcome |
|---|------------|---------------|------------------|
| **A** | Path B + `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128` | Hypothesis 1 | If acc rises ≥1.5pt → confirms lightning threshold is dominant; pursue this knob. |
| **B** | Path B + `--force-dense-minicpm` BUT keep stock flashinfer string | Combines all force-dense side-effects (config + threshold) without rewriting backend twice | If acc reaches Test 12 level → speed gain is real and unrelated to sparse routing; cut a new submission baseline. |
| **C** | Path C with `--max-running-requests` reduced to 16 (closer to Round 13f-1 effective concurrency) | Rules out scheduler-side effects | Should not move acc; confirms the kernel/threshold answer. |
| **D** | Path B with `--enable-torch-compile --torch-compile-max-bs 8` re-enabled | Round 13f-1 had compile dropped; Test 12 keeps it. Compile path may stabilize numerics. | If compile recovers acc → numerical stability is the issue. |

Recommended first run: **experiment B**. If `--force-dense-minicpm` + stock
flashinfer (which is internally what Test 12 already does — they're the
**same** server config) reproduces 79.29%, then Round 13f-1's "speed gain" is
illusory (same backend, same kernel) and the 7% delta must come from one of
{compile, lightning threshold, scheduler}. If we replicate that under the
`flashinfer` string explicitly, we likely already are at the next baseline.

If experiment B shows speed regresses back to Test 12 numbers, then the
gain in Round 13f-1 came specifically from **dropping** `--force-dense-minicpm`,
which means stock flashinfer was running with `has_sparse_attention=True`.
That state is what costs 2.4 acc points and what we'd need to characterize.

## Answer to the original question

> *Are u sure if we set the backend to flashinfer, then it can't handle
> minicpm mixed layers (sparse and lightning)?*

- **Lightning layers**: handled correctly on **both** paths via
  `SimpleGLAAttnBackend` — the std-attn backend is irrelevant for them.
- **Sparse-attention std layers**: stock flashinfer **does compute them**,
  just as **dense full attention** (a superset of top-k sparse). It will
  not crash. It will produce numerically valid outputs. Accuracy may or
  may not match training, depending on whether SALA's fine-tuning is robust
  to seeing un-masked attention. Empirically Round 13f-1 lost 2.4 pts —
  likely **not** due to sparse-vs-dense computation per se but due to the
  collateral config flips (lightning threshold, `has_sparse_attention`
  flag) that change between Path B and Path C.
- **Bottom line**: stock flashinfer is **functionally compatible** with
  SALA. It is **not currently configured to be accuracy-equivalent** to
  the minicpm_flashinfer + force-dense path. The fix is config alignment,
  not kernel replacement. The four experiments above will isolate which
  knob owns the 2.4 acc-pt regression.

## Cross-references

- Test row: `R13f1-flashinfer` in TEST_RESULTS_TRACKING.md.
- Round 13f-1 chat: [chat/CHAT_round13f_change0136_validation_20260429_1410.en.md](chat/CHAT_round13f_change0136_validation_20260429_1410.en.md).
- Off-by-one bug surfaced by CHANGE_0136 (Path A only): [CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md](CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md).
- Code anchors:
  - [python/sglang/srt/layers/attention/attention_registry.py](../../python/sglang/srt/layers/attention/attention_registry.py#L180-L220)
  - [python/sglang/srt/server_args.py](../../python/sglang/srt/server_args.py#L1521-L1525)
  - [python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py](../../python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1484)
  - [python/sglang/srt/layers/attention/minicpm_backend.py](../../python/sglang/srt/layers/attention/minicpm_backend.py#L366)
