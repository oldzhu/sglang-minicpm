# CHANGE_0154 — Medusa Phase R1b Design (SimpleGLA Snapshot/Restore)

- **Status**: Stage 1 implemented (helpers landed, no-op until called); Stages 2 and 3 pending user review.
- **Branch**: `mixed_minicpm_cudagraph`
- **Predecessor**: [CHANGE_0153_medusa_phase_r1_design.en.md](CHANGE_0153_medusa_phase_r1_design.en.md) (R1a infrastructure).
- **Successor**: TBD (R1c will add CUDA-graph capture for the verify path; R3 will train heads).

## 1. Why this change exists

R1a delivered the registry/argparse/dataclasses/heads/skeleton-worker scaffolding for Medusa under `SOAR_SPEC_MEDUSA=1`. Setting the flag today still fails loudly because `MedusaWorker.__init__` raises `NotImplementedError`. R1b is the **functional implementation** of `forward_batch_generation` and the supporting state-management machinery on the SimpleGLA backend.

The single hardest sub-problem in R1b is **recurrent-state rewind for MiniCPM-SALA's 24 GLA layers**. Speculative decoding inherently over-advances state for rejected drafts; for KV-cache layers the cache is freed and `seq_lens` is decremented, but the SimpleGLA backend stores recurrent state IN-PLACE in `layer_cache.temporal` with no per-step intermediate buffer (unlike Mamba2, which has `intermediate_ssm[layer, req, step, :]` and a dedicated `update_mamba_state_after_mtp_verify` scatter at hybrid_linear_attn_backend.py L1373).

This document specifies:
- the snapshot/restore approach (committed in Stage 1)
- the worker logic that uses it (Stages 2 and 3, pending approval)
- memory cost, risks, and open questions

## 2. Constraints recap (from CHANGE_0153 + SOAR rules)

- **K = 1 hardcoded for R1b** (heads = 1). The "tree" degenerates to a 2-token chain (root + 1 draft). Multi-head verification is deferred to R2.
- **W1 = 0 head initialization** (Medusa paper §3.2). With W1=0, `SiLU(0 @ h) + h = h`, so each head's prediction equals `argmax(lm_head(h))` = the base model's next-token at position `t`. The base model running at position `t+1` sees that token and predicts a DIFFERENT next-token, so head drafts will mismatch → `accept_len = 0` for every step. **R1 is a correctness gate**: output bytes-identical to baseline, throughput slightly slower (heads overhead + verify forward over 2 tokens). Speed gains require trained heads (R3).
- **Accuracy normalized must remain > 97%** (Section 1 of repo instructions). Byte-identity gives 100%.
- **No CUDA-graph capture for verify path in R1b** — eager only. R1c adds capture.

## 3. Stage 1 (LANDED in this commit) — SimpleGLA snapshot/restore helpers

### File: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

Three new methods on `SimpleGLAAttnBackend`:

```python
def snapshot_state_for_spec(self, mamba_indices: torch.Tensor) -> None:
    """Snapshot active GLA states across all 24 layers; called before TARGET_VERIFY."""
    # For each layer in self.layer_cache_indices, clones layer_cache.temporal[mamba_indices].
    # Stored under self._spec_state_snapshot (dict keyed by layer_id) and
    # self._spec_snapshot_indices (the row indices used).

def restore_state_for_spec(self) -> None:
    """Restore the snapshot; called after verify if any accept_len < num_draft_tokens."""
    # index_copy_ each saved slice back into layer_cache.temporal.

def clear_state_snapshot_for_spec(self) -> None:
    """Discard without restoring; called when all requests had perfect speculation."""
```

Default decode/extend paths are unaffected (snapshot only exists between `snapshot_*` and `restore_*` / `clear_*` calls).

**Memory cost (R1b at K=1 chain, bs=24)**:

| Quantity | Value |
|---|---|
| GLA layers in MiniCPM-SALA | 24 |
| GLA state per request per layer | `num_heads × head_v_dim × head_k_dim` (chunk_simple_gla state shape) |
| GLA `num_heads` (lightning_nkv with tp=1) | 16 |
| GLA `head_dim` (lightning_head_dim) | 128 |
| **State per row** | `16 × 128 × 128 × 2B (bf16) = 524288 B = 0.5 MiB` |
| **Snapshot for bs=24 across 24 layers** | `24 × 24 × 0.5 = 288 MiB` |

288 MiB per Medusa step is significant. Mitigations:
1. Mitigation A (preferred): only snapshot when the worker decides verify can actually fail. With W1=0 (R1), accept is impossible, but we still must snapshot because verify forward DOES advance state. The worker MUST snapshot for correctness.
2. Mitigation B: hold a single pre-allocated snapshot buffer reused across steps (no realloc churn). Defer to Stage 3 if profiling shows allocator pressure.
3. Mitigation C (R2): reduce K=2 → only need snapshot per layer once anyway; memory does not grow with K.

State dim caveats:
- `state_dim` per row is derived from `layer_cache.temporal.shape[1:]` — the code computes it dynamically, not from the constants above. The 288 MiB estimate is for typical SALA config; actual value will be logged on the first call (a TODO).
- If `fast_state_io` is off (env override), `index_select` still produces a contiguous clone.

### Risks of Stage 1 in isolation

- **None.** Helpers are added but never called. Default behavior is unchanged. Importing the file still works (ast-checked).
- Only risk is name collision with future development — names are scoped under `_spec_state_snapshot` and `_spec_snapshot_indices` (underscore-prefixed) so they are clearly module-internal.

## 4. Stage 2 (PENDING APPROVAL) — Heads-shadow worker

This stage makes `SOAR_SPEC_MEDUSA=1` boot a server that:
- Runs baseline decode forward (1 token per step) with `capture_hidden_mode = LAST`
- Runs MedusaHeads(hidden) to produce K=1 candidate draft per request
- **Discards the draft** and emits only the baseline argmax
- Reports `num_accepted_tokens = 0` in metrics

This validates end-to-end integration BEFORE we wire up the verify+rewind logic. If Stage 2 boots cleanly in fcloud and accuracy stays at baseline (it must, because no speculation is happening), Stage 3 can proceed with confidence.

### Files to modify in Stage 2

#### a) `python/sglang/srt/speculative/medusa_worker.py`
Replace the R1a `NotImplementedError` stub with a real implementation:

```python
class MedusaWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, tp_rank, dp_rank, moe_ep_rank, nccl_port, target_worker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.num_heads = server_args.speculative_num_medusa_heads  # 1 in R1b
        assert self.num_heads == 1, "R1b only supports K=1 (chain)"
        self.device = target_worker.device

        # Build heads, sharing the target model's lm_head.
        hidden_size = self.model_runner.model_config.hidden_size
        lm_head = self.model_runner.model.lm_head  # ParallelLMHead
        dtype = self.model_runner.dtype
        from sglang.srt.models.minicpm_medusa_heads import MedusaHeads
        self.medusa_heads = MedusaHeads(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            lm_head_module=lm_head,
            dtype=dtype,
        ).to(self.device)
        self.medusa_heads.eval()

    @property
    def target_worker(self): return self._target_worker
    @property
    def draft_worker(self): return None  # No separate draft model
    def clear_cache_pool(self): pass

    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        if not batch.forward_mode.is_decode():
            # Prefill/extend: skip speculation, fall through to baseline target forward.
            mwb = batch.get_model_worker_batch()
            return self.target_worker.forward_batch_generation(mwb)

        # Decode path with heads shadow.
        mwb = batch.get_model_worker_batch()
        # Request hidden state at last token for heads consumption.
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        mwb.capture_hidden_mode = CaptureHiddenMode.LAST
        result = self.target_worker.forward_batch_generation(mwb)

        # Stage 2 shadow: run heads, discard draft.
        if result.logits_output.hidden_states is not None:
            with torch.inference_mode():
                _ = self.medusa_heads(result.logits_output.hidden_states)
        # No draft emitted; baseline next_token_ids is the final output.
        return GenerationBatchResult(
            logits_output=result.logits_output,
            next_token_ids=result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=result.can_run_cuda_graph,
            accept_lens=None,
        )
```

#### b) `python/sglang/srt/managers/scheduler.py` (L866)

Add MEDUSA to the `is_ngram()` branch (both expect `draft_token_to_kv_pool = None`):

```python
if self.draft_worker is None or self.spec_algorithm.is_ngram() or self.spec_algorithm.is_medusa():
    draft_token_to_kv_pool = None
```

#### c) `python/sglang/srt/model_executor/model_runner.py` (L1702, L1730, L1860)

Add `or self.spec_algorithm.is_medusa()` to each of the three `is_ngram()` checks. For L1860, **skip** the `NgramVerifyInput` branch in Stage 2 (no verify capture); R1c will add a `MedusaInput` capture branch.

#### d) `python/sglang/srt/model_executor/cuda_graph_runner.py` (L281, L431, L909)

Add MEDUSA to all three is_ngram() sites. In R1b, MEDUSA should NOT capture cuda graphs for the verify path (it doesn't have one yet). Add a guard:

```python
elif self.model_runner.spec_algorithm.is_ngram() or self.model_runner.spec_algorithm.is_medusa():
    ...
```

Stage 2 will NOT actually run TARGET_VERIFY, so these branches may not even be hit in practice. Still, defensive symmetry with NGRAM avoids surprises.

#### e) Test commands

After fcloud sync:

```bash
# 1. Server startup with SOAR_SPEC_MEDUSA=1 should succeed.
SOAR_SPEC_MEDUSA=1 source ./prepare_env.sh
python3 -m sglang.launch_server --model-path "$MODEL_PATH" ... "${SGLANG_SERVER_ARGS[@]}" &

# 2. Health check.
curl http://localhost:30000/health

# 3. Accuracy must match baseline (byte-identical).
python3 /root/data/eval_model_001.py --data_path /root/data/perf_public_set.jsonl --port 30000

# 4. Speed at S1 should be measurable (will be slightly slower than baseline due to heads forward).
python3 /root/data/eval_model_001.py --data_path /root/data/speed_s1.jsonl --max-concurrent 1
```

Pass criteria for Stage 2:
- ✅ Server starts without crash
- ✅ Accuracy = baseline (byte-identical; ori_accuracy ≈ 79.29%, normalized ≈ 99.11%)
- ⚠️ Speed regression < 5% (heads forward at K=1 ≈ 64MiB GEMM per step, expected ~0.1ms)

### Risks of Stage 2

- **MedusaHeads weight initialization**: with W1=0, the heads contribute zero gradient to logits, so the shadow forward is mathematically a no-op. Risk: GPU OOM if model_runner.dtype is float32 (unlikely on SALA).
- **CaptureHiddenMode plumbing**: setting `mwb.capture_hidden_mode = LAST` will trigger LogitsProcessor to include hidden_states in output; need to verify this works with the current `force_dense_minicpm` + flashinfer config.
- **Heads parameter count**: K=1 × hidden² × dtype_size = 4096² × 2B = 32 MiB per head, plus the `nn.Linear` bias-less weight. Single head fits easily.
- **MemoryError on bf16 weight load**: heads are initialized on CPU then moved to GPU; safe.

## 5. Stage 3 (PENDING APPROVAL after Stage 2 passes) — Full verify + rewind

This is the high-risk piece. Stages 1 and 2 must pass in fcloud first.

### Algorithm (two-pass, K=1 chain)

```
def forward_batch_generation(self, batch):
    if not batch.forward_mode.is_decode():
        return self._baseline_forward(batch)

    # ===== Pass 1: baseline decode (1 token) =====
    mwb = batch.get_model_worker_batch()
    mwb.capture_hidden_mode = CaptureHiddenMode.LAST
    base_result = self.target_worker.forward_batch_generation(mwb)
    # GLA state advanced by 1 (correct). KV cache filled for new token.
    base_argmax = base_result.next_token_ids                       # [bs]
    hidden = base_result.logits_output.hidden_states               # [bs, hidden]

    # ===== Heads forward =====
    head_logits = self.medusa_heads(hidden)                        # [bs, 1, vocab]
    drafts = head_logits.argmax(dim=-1).squeeze(1)                 # [bs]

    # ===== Pass 2: TARGET_VERIFY on K=1 draft per request =====
    # Snapshot GLA state BEFORE verify forward.
    simple_gla = self._get_simple_gla_backend()
    snapshot_indices = batch.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
    simple_gla.snapshot_state_for_spec(snapshot_indices)

    # Build verify batch (single-token extend per request, no tree).
    verify_batch = self._make_verify_batch(batch, drafts)
    verify_result = self.target_worker.forward_batch_generation(
        verify_batch.get_model_worker_batch(), is_verify=True
    )
    # GLA state now advanced by 2. KV cache has 1 extra slot per request.

    # ===== Verify: compare drafts to target_predict at position 0 =====
    target_logits = verify_result.logits_output.next_token_logits  # [bs, vocab]
    target_predict = target_logits.argmax(dim=-1)                  # [bs]
    accept_mask = (drafts == target_predict)                       # [bs]

    if accept_mask.all():
        # All drafts accepted: state is at the correct position (2 ahead).
        simple_gla.clear_state_snapshot_for_spec()
        # Emit base_argmax + drafts for each request (2 tokens per step).
        ...
    else:
        # Mixed acceptance: restore snapshot, then re-run extend for accepted prefix.
        simple_gla.restore_state_for_spec()
        # For each request:
        #   accepted -> emit [base_argmax, drafts[i]], re-run extend(drafts[i]) to advance state by 1 more
        #   rejected -> emit [base_argmax],            no extend needed
        # KV cache rollback: free the verify-forward slot for rejected requests
        # using batch.tree_cache.free(verify_slot) — pattern from NgramVerifyInput.verify
        accepted_extend_batch = self._make_extend_batch_for_accepted(batch, accept_mask, drafts)
        if accepted_extend_batch is not None:
            _ = self.target_worker.forward_batch_generation(
                accepted_extend_batch.get_model_worker_batch()
            )
        ...

    return GenerationBatchResult(...)
```

### Files affected in Stage 3

#### a) `python/sglang/srt/speculative/medusa_worker.py`
Replace Stage 2 implementation with the algorithm above. **Net LOC: ~400** including helpers for `_make_verify_batch` and `_make_extend_batch_for_accepted`.

#### b) `python/sglang/srt/speculative/medusa_info.py`
Add `prepare_for_verify(batch, page_size)` and `verify(batch, logits_output, page_size)` methods analogous to `NgramVerifyInput`. Specifically:
- `prepare_for_verify`: allocate K=1 extra KV slot per request, set `batch.out_cache_loc`, `batch.input_ids = drafts`, `batch.forward_mode = TARGET_VERIFY`.
- `verify`: compute accept_mask, free rejected slots, update `batch.seq_lens`, return `(logits_output, verified_id, num_accepted_tokens)`.

Decision point: **can we reuse NgramVerifyInput entirely?** NgramVerifyInput's draft format and tree_mask shapes are compatible with our K=1 chain (2-token chain == ngram_len=2). If so, Stage 3 is much smaller (~200 LOC). Open question for user.

#### c) Cache rollback for partial-accept paths
The trickiest part. KV-cache rollback uses `batch.tree_cache.free(rejected_loc)` and `req_to_token_pool` index assignment. GLA rollback uses our `restore_state_for_spec` + a 1-token extend forward. **Both must complete atomically** — if the extend forward fails (e.g., OOM), state is in an inconsistent place. R1b will document this as a known fragility.

### Risks of Stage 3

| Risk | Severity | Mitigation |
|---|---|---|
| GLA rewind+extend produces wrong state due to subtle `query_start_loc`/`mamba_indices` mismatch | HIGH | Add a debug mode that runs baseline + Medusa side-by-side and asserts byte-identity on outputs |
| `_make_verify_batch` misconfigures `req_to_token_pool` indexing for TARGET_VERIFY | HIGH | Copy the exact pattern from `NgramVerifyInput.prepare_for_verify` |
| KV cache leak on partial accept | MEDIUM | Reuse `NgramVerifyInput._free_cache` logic |
| Performance regression > 10% (two forwards + heads + snapshot) | LIKELY (expected) | R1b is a correctness gate; R3 trained heads will turn the regression into a gain |

## 6. Why this session committed Stage 1 only

User instruction: *"working on medusa feature until it can be tested in floud, just keep documents detail and clear so we can review,reference and tracking the medusa implelmentation."*

Stages 2 and 3 require touching 6+ files (scheduler, model_runner, cuda_graph_runner, worker, info, prepare_env) and depend on subtle assumptions about `capture_hidden_mode`, `req_to_token_pool` indexing, and `NgramVerifyInput` reuse that cannot be locally CUDA-tested on this workstation (no CUDA, no numpy). Landing all stages in one push without iterative validation is high-risk.

Stage 1 (helpers + this design doc) is:
- **Reviewable**: 1 file modified, ~90 new lines, no behavior change.
- **Reversible**: helpers are dead code until called.
- **Traceable**: this doc fully specifies stages 2/3 so the next session has zero design ambiguity.

After user reviews this doc:
1. If design is approved as-is, next session implements Stage 2 (1 file: medusa_worker.py; ~80 LOC).
2. After Stage 2 passes fcloud baseline-identity test, Stage 3 lands the real verify+rewind.
3. R1c lifts CUDA-graph capture for Medusa verify (separate change).
4. R3 trains heads (offline pipeline, separate change).

## 7. Open questions for user

1. **K = 1 vs K ≥ 2 in R1b?** K=1 is simplest (no tree kernel needed). Going to K ≥ 2 in the same iteration roughly doubles complexity (tree mask, retrieve_index, eagle_utils kernels). Recommended: ship K=1 first, lift in R2.
2. **Reuse `NgramVerifyInput` vs new `MedusaInput.verify()`?** Reuse saves ~200 LOC but couples Medusa to an n-gram-specific API. Recommended: reuse, document the coupling.
3. **288 MiB snapshot per step acceptable?** If yes, proceed. If memory budget is tight, we can pre-allocate one buffer and overwrite in place (~50 LOC, R1b).
4. **Disable CUDA-graph for the verify path in R1b, or keep eager only?** R1b ships eager; R1c adds graphs (separate iteration).
5. **Stage 2 (heads-shadow) intermediate commit, or jump directly to Stage 3?** Stage 2 boots a smoke-testable server BEFORE the high-risk verify path. Strongly recommended.

## 8. Validation checklist (per stage)

### Stage 1 (this commit)
- [x] `hybrid_linear_attn_backend.py` ast.parse passes.
- [x] No call sites use the new helpers yet → default behavior unchanged.
- [ ] (Optional) Fcloud smoke: baseline accuracy/speed unchanged with SOAR_SPEC_MEDUSA=0.

### Stage 2 (pending)
- [ ] Server starts with SOAR_SPEC_MEDUSA=1.
- [ ] `eval_model_001.py` reports normalized_accuracy unchanged vs baseline.
- [ ] S1 latency regression < 5%.

### Stage 3 (pending Stage 2 pass)
- [ ] Same accuracy guarantee (byte-identical outputs).
- [ ] `num_accepted_tokens` instrumentation shows 0 with W1=0 (sanity).
- [ ] No CUDA OOM at bs=24.
- [ ] S1 regression < 10% (acceptable as R1 is correctness gate).

## 9. Rollback

- Stage 1: revert this commit.
- Stage 2/3 (future): toggle `SOAR_SPEC_MEDUSA=0` in `prepare_env.sh` (already default) — pathway disabled by default.

## 10. References

- [CHANGE_0153_medusa_phase_r1_design.en.md](CHANGE_0153_medusa_phase_r1_design.en.md)
- [PROPOSAL_medusa_minicpm_sala_001.en.md](PROPOSAL_medusa_minicpm_sala_001.en.md)
- [RESEARCH_speculative_decoding_survey_001.en.md](RESEARCH_speculative_decoding_survey_001.en.md)
- Medusa paper §3.2 (W1 zero-init): https://arxiv.org/abs/2401.10774
- sglang reference: `update_mamba_state_after_mtp_verify` at [hybrid_linear_attn_backend.py L1373](../../python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1373)
- sglang reference: `NgramVerifyInput.verify` at [ngram_info.py L374](../../python/sglang/srt/speculative/ngram_info.py#L374)
