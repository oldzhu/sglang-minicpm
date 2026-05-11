# PROPOSAL — Medusa Phase R1b Stage 3: real verify + rewind

**Status**: proposal, awaiting validation in session.
**Branch**: `mixed_minicpm_cudagraph`.
**Predecessors**: [CHANGE_0153 (R1 design)](CHANGE_0153_medusa_phase_r1_design.en.md), [CHANGE_0154 (R1a scaffolding)](CHANGE_0154_medusa_phase_r1a_scaffolding.en.md), [CHANGE_0155 (R1b Stage 2 pass-through)](CHANGE_0155_medusa_phase_r1b_stage2.en.md).

## 1. Background & motivation

After Stage 2 (commit `46553947b` → `1e5cd15a8` → `469e5815f`), `MedusaWorker` is a pure pass-through: the worker is wired, heads are allocated, the scheduler dispatch works, and runtime is bit-equivalent to the v22 baseline (acc 80.11%, S1 118.28s ≈ Test 12). The submission tarball `minicpm_sala_submit_v23.tar.gz` validates this on official hardware.

Stage 3's task is to make the worker actually produce + verify draft tokens, while preserving the zero-init head invariant: at K=1 with `W1=0` heads, every draft token equals the target's argmax, so the verify always accepts and the model output is **byte-identical** to baseline.

## 2. Rule-compliance statement

- Submission size unchanged (heads were already counted in Stage 2; only host code changes).
- On-site quantization unchanged.
- Output byte-identity preserved at zero-init heads → normalized accuracy ≥ baseline (expect ≥ 99% → C=1.0).
- No forbidden tricks — same eval harness, same `--max-concurrent`, no prefix-cache re-enable.

## 3. Goal & expected gain ceiling

| Stage | Scope | Expected accuracy | Expected speed |
|-------|-------|-------------------|----------------|
| **3a** (this proposal, eager) | Heads + verify wired; cuda-graph OFF for verify | byte-identical (acc ≈ 80.11%, C=1.0) | likely **slower** than baseline (eager + verify overhead) — correctness only |
| **3b** (follow-up, cuda-graph) | Same logic, cuda-graph captures TARGET_VERIFY | acc unchanged | **best case S1 −30~40%** (1 forward emits 2 tokens at 100% accept, minus head + verify overhead). Realistic gain probably **−10~20%** until heads are trained. |

**Theoretical ceiling** at K=1, 100% accept: ~1.7-1.8x decode speedup. **Practical ceiling at this stage**: limited by verify overhead and head forward cost. Real submission gain awaits head training (CHANGE_0154 §6, deferred).

## 4. Existing infra we reuse

`ngram_worker.py` + `ngram_info.py` already implement the entire verify pipeline for linear K-token chains on this exact server config (`--attention-backend minicpm_flashinfer`). Specifically:

| Component | NGRAM file | We reuse? |
|-----------|------------|-----------|
| `NgramVerifyInput.prepare_for_verify` (alloc verify KV slots, set `out_cache_loc`, expand `seq_lens`) | `ngram_info.py:74` | Yes, generic — copy or import. |
| `NgramVerifyInput.verify` (greedy accept walk, KV rewind, `next_token_ids` extract) | `ngram_info.py:374` | Yes. |
| `generate_attn_arg_prefill` (flashinfer KV-indices, custom_mask for verify) | `ngram_info.py:124` | Yes. |
| `minicpm_backend.py:521` `is_target_verify()` branch | already exists | Yes, validates backend compatibility. |
| `cuda_graph_runner.py:286` TARGET_VERIFY capture path | already exists for eagle | reuse for Stage 3b only. |

**Conclusion**: we don't reinvent verify; we plug Medusa head forward in where NGRAM does cache lookup.

## 5. Implementation plan (Stage 3a)

### 5.1 Hidden-state capture

Medusa heads consume the **previous step's last hidden state** to produce the next step's draft. So before any verify can run, we need to (a) set `spec_info.capture_hidden_mode = CaptureHiddenMode.LAST` on the previous forward, and (b) plumb the returned `hidden_states` back to the worker.

For the **first** decode of a sequence (right after extend), no previous hidden state exists yet. We fall back to pass-through (no draft, no verify) for that single step, then start specing from step #2.

### 5.2 Draft generation

```python
# In MedusaWorker:
def _generate_draft(self, prev_hidden: torch.Tensor) -> torch.Tensor:
    # prev_hidden: (bs, hidden_size) — last hidden of each seq from prev step
    # MedusaHeads.forward returns (bs, K, vocab)
    logits = self.medusa_heads(prev_hidden)
    draft_tokens = logits.argmax(dim=-1)  # (bs, K)
    return draft_tokens  # int64
```

### 5.3 Verify input construction

For K=1: trivial linear chain — each "tree" is a single draft token following the just-sampled target token. We use:

- `draft_token` = `[t_prev_target, d_1]` per seq (length 2)
- `tree_mask` = identity-shaped causal mask for the 2-token chain
- `positions` = `[seq_len-1, seq_len]`
- Pass to `NgramVerifyInput` (rename irrelevant — it's a `SpecInputType.NGRAM_VERIFY` for now; can either re-tag or just reuse the type).

### 5.4 Worker forward flow (Stage 3a)

```
def forward_batch_generation(batch):
    if batch.forward_mode.is_extend():
        # First-time prefill — pass through with LAST capture so we have hidden for step 2.
        ... target_worker.forward_batch_generation(batch) with capture_hidden_mode=LAST
        save self.last_hidden[req_id] = h_per_seq
        return result (no spec)

    if batch.forward_mode.is_decode():
        if no cached hidden for any req in batch:
            # Cold start — pass through, capture hidden for next step
            ... pass-through with LAST capture
            return result

        else:
            # Hot path — draft, verify, rewind
            drafts = self._generate_draft(self.last_hidden_for_batch(batch))
            self._build_verify_input(batch, drafts)
            batch.forward_mode = TARGET_VERIFY
            batch.spec_info.capture_hidden_mode = LAST  # for NEXT step's draft
            res = target_worker.forward_batch_generation(batch_mwb, is_verify=True)
            logits_output, next_token_ids, num_accepted = verify_input.verify(...)
            # Update self.last_hidden from logits_output.hidden_states (per accepted index)
            return GenerationBatchResult(...)
```

### 5.5 Correctness check (zero-init invariant)

With `W1 = 0`:
- `MedusaHead.forward(h) = SiLU(0) + h = h`
- `MedusaHeads.forward(h)[k] = lm_head(h)` = same logits as target's lm_head on `h`
- `argmax(MedusaHeads(h)) = argmax(target_lm_head(h)) = t_prev_target` (which is also the next-token argmax)

So `d_1 = t_prev_target_argmax_of_step_N == target_argmax_at_position_N` — verify always accepts, produces same `next_token_ids` as pass-through. **Byte-identical output.**

## 6. Risk assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Hidden state capture in eager mode silently broken | low | NGRAM-style `is_verify=True` already exercises hidden capture for NGRAM; we follow same wiring. |
| `NgramVerifyInput` reuse causes type-check failures elsewhere | low | `SpecInputType.NGRAM_VERIFY` is duck-typed downstream; if anything breaks we add a `MedusaVerifyInput` alias subclass. |
| `minicpm_flashinfer` TARGET_VERIFY branch broken on hybrid GLA layers | medium | NGRAM has been working historically on this exact model — but we never ran it on the FP8 KV + dense path. **Mitigation**: bring up Stage 3a in EAGER mode (already supported via `SOAR_SPEC_MEDUSA_EAGER=1` toggle from CHANGE_0155 §14). |
| KV rewind interacts badly with SimpleGLA recurrent state | medium-high | CHANGE_0153 §3 originally flagged this. For Stage 3a, since we always accept at zero-init, **no rewind ever fires**, so we sidestep the issue. Stage 3b/3c (post-training heads) would need real validation. |
| Cuda-graph capture for TARGET_VERIFY on minicpm_flashinfer broken | medium | Defer to Stage 3b; Stage 3a runs eager. |

## 7. Validation plan

### Stage 3a sanity (local)
1. Server boots with `SOAR_SPEC_MEDUSA=1 SOAR_SPEC_MEDUSA_EAGER=1`.
2. Health check passes.
3. 3-sample accuracy smoke test → all answers identical to v22 baseline (byte-equality, since zero-init heads + greedy decode).

### Stage 3a fcloud
1. `sync` → `restart-server` → `wait-server` → `accuracy` full run.
2. **Pass criterion**: `ori_accuracy ≥ 80.0%` (within local noise of Stage 2 cuda-graph result 80.11%).
3. If pass, S1 quick (no S8/Smax needed — speed expected to be slower in eager).

### Stage 3b (deferred to next iteration if 3a passes)
1. Disable EAGER toggle → cuda-graph back on.
2. Validate cuda-graph captures TARGET_VERIFY shape.
3. Full accuracy + S1/S8/Smax run.

## 8. Files to touch

| File | Change |
|------|--------|
| `python/sglang/srt/speculative/medusa_worker.py` | Replace Stage 2 pass-through with Stage 3a draft+verify flow. Add hidden-state cache. |
| `python/sglang/srt/models/minicpm_medusa_heads.py` | No change (forward already correct). |
| `benchmark/soar/demo_sala/prepare_env.sh` | No change initially — Stage 3a runs with `SOAR_SPEC_MEDUSA_EAGER=1` (manual export). After 3b, will flip back to default cuda-graph. |
| `docs/soar_2026_changes/CHANGE_0156_medusa_phase_r1b_stage3.{en,zh}.md` | Created on entry to Stage 3 (this proposal becomes its §1-7). |

## 9. Rollback

- Stage 3a: `SOAR_SPEC_MEDUSA=0` → MedusaWorker not instantiated → behavior = v22 baseline. Always safe.
- Worker-level: if Stage 3a logic itself breaks, revert `medusa_worker.py` to commit `46553947b` (Stage 2 pass-through).

## 10. Next steps after 3a

If 3a passes (correctness validated):
- **Stage 3b**: re-enable cuda-graph for TARGET_VERIFY + DECODE; measure real speed gain ceiling.
- **Stage 3c** (deferred, may require model retraining): initialize heads from `lm_head` (CHANGE_0154 §6) or train heads on eval distribution — push acceptance rate above the 100%-on-zero-init invariant (which only holds because heads predict identity to target).

## 11. Backend probe correction (2026-05-11)

During Stage 3a code-up I read `minicpm_backend.py:521` and saw `NotImplementedError` on `is_target_verify()`, and briefly concluded Stage 3 was blocked. **That conclusion was wrong.** User pointed out that `prepare_env.sh` line 199 sets `SOAR_BACKEND_VARIANT=flashinfer` by default, which rewrites the launch arg to `--attention-backend flashinfer` (the stock backend, not the custom `minicpm_flashinfer`). The `NotImplementedError` only fires if someone explicitly overrides `SOAR_BACKEND_VARIANT=minicpm_flashinfer`.

**Stock flashinfer fully supports `TARGET_VERIFY`** — it is the canonical backend for sglang's eagle and ngram paths. So Stage 3 is not blocked; the original §1-10 plan stands and the §6 risk "minicpm_flashinfer TARGET_VERIFY untested on hybrid GLA" simply does not apply because we don't use that backend by default.

### Lesson logged

Reading a source file's `NotImplementedError` without first confirming which backend is actually used at runtime is a false-alarm pattern. Always check `prepare_env.sh` defaults before concluding a code path is blocked.

Proceeding with Stage 3a implementation against stock flashinfer.

## 12. NGRAM probe finding (2026-05-11) — Stage 3b cuda-graph blocker located in advance

Before writing any Medusa Stage 3a code, we ran a zero-code-change NGRAM speculative probe on fcloud to validate that our exact config (stock flashinfer + GPTQ + FP8 KV + dense + mixed-chunk + torch.compile + 16-bs cuda-graph buckets) can host any TARGET_VERIFY workload end-to-end.

### Probe outcome

Server boot reached cuda-graph capture and **crashed on the first verify-shape bucket**:

```
File "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 515, in _capture_metadata
    if forward_mode.is_target_verify() and spec_info.topk > 1:
AttributeError: 'NgramVerifyInput' object has no attribute 'topk'
```

Same check at line 575 in `_replay_metadata` uses `spec_info.topk` and `spec_info.draft_token_num` unconditionally.

### Why this matters for Medusa

- `MedusaInput` (`python/sglang/srt/speculative/medusa_info.py`) has **neither `topk` nor `draft_token_num`** attributes — confirmed via grep.
- The hybrid GLA backend's `_capture_metadata` / `_replay_metadata` were written assuming spec_info has Eagle's shape (tree-mask with `topk`, `draft_token_num`, `retrive_next_token`, etc.).
- Therefore **Stage 3b (Medusa + cuda-graph) will hit the identical AttributeError** the moment we re-enable cuda-graph for TARGET_VERIFY. The crash is not Medusa-specific — it is a hybrid-backend gap that affects any non-Eagle speculative algorithm (NGRAM, Medusa, future standalone draft) on this backend.

### Implications

1. **Stage 3a (eager) is still safe to write** — the crash is inside cuda-graph capture, which Stage 3a disables via `SOAR_SPEC_MEDUSA_EAGER=1`. We can validate correctness in eager mode without touching the hybrid backend.
2. **Stage 3b (cuda-graph) now has a known prerequisite fix in `hybrid_linear_attn_backend.py`** before any speed gain can materialize:
   - Lines 515 / 575: gate the eagle-tree-mask branch on `getattr(spec_info, "topk", 1) > 1` (linear K-token verify has effective topk=1, no tree).
   - Line 570: replace `spec_info.draft_token_num` with `getattr(spec_info, "draft_token_num", None) or self.speculative_num_draft_tokens` (or read it from the worker's static config).
   - This is a ~5-line tolerance patch, low risk, and benefits NGRAM + Medusa + any future linear-verify algorithm symmetrically.
3. **Submission risk for v23**: v23 has `SOAR_SPEC_MEDUSA=1` default-on. Stage 2 medusa_worker.py is pure pass-through (flips `spec_algorithm=NONE` before `get_model_worker_batch`), so we never enter TARGET_VERIFY at runtime — the hybrid-backend gap does NOT affect v23. v23 stays safe.

### Probe verdict

**Highly positive ROI.** A 5-minute probe (zero new code) located the exact line that would block Stage 3b cuda-graph speed gains, and confirmed Stage 3a (eager-only) infrastructure is sound up to that crash point (model load, KV alloc, hybrid pool init, KV dtype fp8_e5m2, all succeeded). The lesson from §11 is reinforced: cheap end-to-end probes before speculative code-writing save hours of false-positive debugging.

### Stage plan update

| Stage | Was | Now |
|-------|-----|-----|
| 3a | Write Medusa worker + run with `SOAR_SPEC_MEDUSA_EAGER=1` | Same (no change) |
| 3b | Disable EAGER toggle → cuda-graph back on → measure speed | **Prerequisite**: patch `hybrid_linear_attn_backend.py` `_capture_metadata` + `_replay_metadata` to tolerate Medusa-shape spec_info, then disable EAGER toggle and measure |

This finding will be reflected in `CHANGE_0156` when Stage 3 work starts.

## 13. Tolerance patch applied (2026-05-11) — §12 prerequisite resolved

Applied the 2-line tolerance patch identified in §12 to `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`:

- **Line 515** (`_capture_metadata`): `spec_info.topk > 1` → `getattr(spec_info, "topk", 1) > 1`
- **Line 575** (`_replay_metadata`): same change

No change to line 570 yet: `spec_info.draft_token_num` is already present on `NgramVerifyInput` so the existing access works for the immediate NGRAM re-probe. When Stage 3a starts wiring Medusa, we will add a `draft_token_num` attribute to `MedusaInput` (or use the static fallback) at the same time.

### Why this is safe to apply now (without per-feature gate)

1. **Eagle/Standalone unaffected**: their spec_info has `topk` explicitly set, so `getattr(..., "topk", 1)` returns the real value → identical behavior to before.
2. **NGRAM/Medusa**: missing attr → falls back to 1 → takes the non-tree-mask branch, which is exactly what linear K-token verify needs (custom_mask alone suffices; no `retrive_next_token` retrieval pointers).
3. **No control-flow changes**: only the gate condition becomes attr-tolerant. The two branches' bodies are untouched.
4. **v23 byte-equality preserved**: v23 default config runs Stage 2 pass-through (flips spec_algorithm=NONE before forward), never enters TARGET_VERIFY at runtime, so this code path is not exercised in v23 submission.

### Plan for re-probe

1. Push the patch.
2. Sync to fcloud (`fcloud_workflow.py sync` copies `python/sglang` into `submission_sim/sglang/python`).
3. Restart NGRAM probe with same env: `--env SOAR_SPEC_MEDUSA=0 --env SOAR_SPEC_NGRAM=1`.
4. **Expected**: cuda-graph capture proceeds past bs=24 verify bucket; server reaches READY.
5. Run accuracy eval. NGRAM with stock parameters (`speculative_num_draft_tokens=12`, `speculative_ngram_branch_length=18`, etc.) on our config.
6. **Pass criterion**: `ori_accuracy ≥ 80.0%` (within local noise of Stage 2 cuda-graph 80.11%) → proves the verify pipeline is end-to-end functional on our exact stack → green light to write Stage 3a Medusa code.
7. **If accuracy degrades meaningfully** (e.g. < 78%) → there is a second-order verify bug beyond the tree-mask gate; diagnose before committing Medusa code.

### Risk note

The patch only widens tolerance — it does NOT prove the non-tree branch is functionally correct on the hybrid backend for verify-shape attention. The NGRAM re-probe is the only way to validate. If the probe still fails after the patch, we'll add a Stage 3 prerequisite document to track the additional fix.

