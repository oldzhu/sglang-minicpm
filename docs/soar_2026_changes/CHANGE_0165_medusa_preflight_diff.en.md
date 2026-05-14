# CHANGE_0165 — Medusa Pre-flight Diff (MedusaWorker vs NgramWorker substep comparison)

Status: **PROPOSAL** — awaiting explicit user approval before any fcloud action.
Owner: agent
Related: CHANGE_0163 (Stage 3b trained heads), CHANGE_0164 (ndt=2 catastrophic attempt), PROPOSAL_medusa_k1_positional_offbyone_fix.

## 1. Background and motivation

Stage 3a (current shipped state, commit `a489d78d4`) keeps `MedusaWorker.draft_token_num = num_heads = 1`. The verify kernel walks a single-node tree (the bonus root), commits its argmax, and ends; effectively this is a dense decode with extra overhead. Accuracy is ~0.6pt below dense baseline; speedup is structurally ~0 (no draft positions to accept beyond bonus).

Stage 3b ndt=2 (CHANGE_0164) tried to fix this by setting `draft_token_num = num_heads + 1 = 2` plus hand-rolled positions / retrive_* metadata. Result: **15.13% accuracy, C=0, mcq runaway (avg_out=64460), eval 7911s**.

Three working theories survive (all unverified by code-level inspection alone):

1. **Positional invariant violation** — feeding `output_ids[-1]` at slot `seq_lens` mis-aligns KV against model's expected autoregressive state for position `seq_lens+1`.
2. **retrive_* metadata orientation** — `verify_tree_greedy` may expect orientation produced by `reconstruct_indices_from_tree_mask` kernel; hand-rolled equivalents may differ in a subtle way at ndt=2.
3. **Hidden-state capture position** — `CaptureHiddenMode.LAST` returns hidden at the LAST verify position; for ndt=2 the trained head may need hidden at the bonus position instead.

A fourth (less likely) concern noted in [hybrid_linear_attn_backend.py](python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py):

4. **Lightning attention `SpeculativeState` mamba pool** — verify path asserts the mamba pool is `SpeculativeState`; need to confirm MedusaWorker triggers that allocation. (Stage 3a did not crash, so this is probably fine, but worth verifying once.)

Without runtime evidence we cannot rank these. A 2 h accuracy eval per hypothesis is too expensive (and fragile — see CHANGE_0164 for "interesting but unproductive" failures). The pre-flight diff is the de-risker.

## 2. Rule-compliance statement

- Pre-flight is a **diagnostic script under `benchmark/soar/demo_sala/`** — runs through the existing sglang `Engine` API at single-process scope.
- No edits to `eval_model_001.py` (per eval-script-integrity rule).
- No edits to submission-side files until a diff is identified and the fix is documented + approved.
- All fcloud actions follow the **cost-saving rule** (pause between iterations) and require user approval each round.

## 3. Pre-flight script: design (proposed)

**Filename**: `benchmark/soar/demo_sala/preflight_medusa_vs_ngram.py`

**What it does** (one fcloud invocation per iteration):

1. Build a deterministic 1-batch input: a single fixed prompt of ~256 tokens (use the first mcq sample from `perf_public_set.jsonl`).
2. **Phase A — Reference run with NgramWorker**:
   - Launch sglang Engine with `--speculative-algorithm NGRAM --speculative-num-draft-tokens 2`.
   - Run prefill + 1 verify step.
   - Capture and dump, in JSON/pickle:
     - `batch.seq_lens` (before and after `prepare_for_verify`)
     - `spec_info.draft_token`, `positions`, `tree_mask`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling`, `draft_token_num`
     - `model_worker_batch.input_ids`
     - `out_cache_loc` (slots written)
     - Post-forward: `logits_output.next_token_logits` argmax at each verify position
     - `spec_info.accept_length`, `next_token_ids`
3. **Phase B — Test run with MedusaWorker**:
   - Restart Engine with `--speculative-algorithm MEDUSA --speculative-num-medusa-heads 1 --speculative-num-draft-tokens 2`.
   - Same prompt, same prefill, 1 verify step.
   - Dump the same fields.
4. **Phase C — Diff**:
   - Field-by-field diff between (A) and (B).
   - Print a structured report: `[FIELD] [A_value] [B_value] [DIFFERS?] [reason if known]`.

**Sanity checks the script must include**:
- Both runs use identical model weights (GPTQ + FP8 KV).
- Both runs use identical RNG seed and decode mode (`temperature=0`, greedy).
- For the Ngram run, force the ngram_cache to return a known token sequence (or use a primed cache) so `draft_token[0..1]` is deterministic; otherwise the comparison's draft-content axis is uncontrolled.
  - **Fallback if priming is hard**: inject a `_force_draft_tokens=[tok0, tok1]` override into NgramWorker for this experiment only (revert before any real eval).
- For the Medusa run with the same `tok0`, `tok1` forced as draft, the diff isolates **metadata-only** differences from draft-content differences.

**Estimated runtime per iteration**: ~5–10 min on fcloud (engine init dominates; the actual two single-step runs are seconds).

## 4. Diff iteration protocol (mandatory)

Every diff iteration follows this protocol. **All iterations append to this same doc pair** as numbered subsections (`§5.1`, `§5.2`, ...).

For each detected diff between MedusaWorker (B) and NgramWorker (A):

| Field | A (Ngram) | B (Medusa) | Differs? | Severity | Decision | Why |
|---|---|---|---|---|---|---|
| (one row per inspected field) | | | yes/no | critical / cosmetic / unknown | clean / keep / investigate | (one-line justification) |

**Severity labels**:
- **critical**: changes model forward semantics or KV writes. Must clean before fcloud eval.
- **cosmetic**: doesn't affect KV/logits/accept walk (e.g., dtype `int64` vs `int32` where kernel accepts both). May leave as-is and document.
- **unknown**: not obvious which side is right. Either (a) add a probe and re-run pre-flight, or (b) try cleaning to match Ngram and re-run pre-flight to see if it helps or hurts.

**Decision rule**:
- **clean** = modify MedusaWorker to produce the same value as NgramWorker.
- **keep** = leave Medusa's version, but justify why it's intentional and not breaking.
- **investigate** = need more probes; do not change code, add diagnostic and re-run.

Each iteration ends with:
- Updated MedusaWorker code (in `python/sglang/srt/speculative/medusa_worker.py` and the submission mirror).
- Commit message: `medusa: preflight iter N — clean <field>` or `medusa: preflight iter N — investigate <field>`.
- Push to `minicpm-src`.
- Append §5.N to this doc with the table above + code diff summary.
- Re-run pre-flight on fcloud to confirm the diff is gone (and no new diff appeared).
- Pause fcloud.

**Exit condition**: pre-flight reports zero **critical** diffs. **cosmetic** diffs may remain documented. At that point we run the full accuracy + speed eval; if it passes (acc ≥ 78%, S1 ≤ Stage 3a S1), Stage 3b ndt=2 is shipped; if not, the working theory in §1 was wrong and we open a new investigation under a new CHANGE_xxxx.

## 5. Iteration log

(Empty — to be populated after each pre-flight run.)

## 6. Risks

- **R1**: Engine startup non-determinism (e.g., flashinfer kernel JIT compile order) could change KV layout between A and B runs. Mitigation: persist Engine across the two phases if possible; otherwise, pin random seeds and run B immediately after A in the same Python process.
- **R2**: Forcing draft tokens in Ngram for determinism may not be straightforward. Fallback: dump *both* values, accept that draft content differs, and only diff the **metadata** fields (positions, tree_mask, retrive_*). Draft content will be normalized in §5 reports.
- **R3**: If the bug is **post-verify** (in `spec_info.verify()` accept walk consuming wrong logits index), the single-step pre-flight may show clean inputs but wrong output. Mitigation: dump `next_token_logits` AND `accept_length` AND `next_token_ids`, so the verify step itself is also diffed.
- **R4**: Stage 3b trained-head pathway adds a 6th comparison axis (head forward on captured hidden). Defer that to a later iteration once Stage 3a-equivalent ndt=2 is clean. Initial pre-flight uses Stage 3a fallback (`output_ids[-1]` as draft) for both A's draft-content override AND B's natural path.

## 7. Approval gate

Before any fcloud action the agent will:
- (a) Show this document for user review.
- (b) Show the proposed pre-flight script content.
- (c) Wait for explicit "go" on script content + fcloud start-instance.

After each iteration the agent will:
- (a) Show the §5.N table.
- (b) Show the proposed code change (if any).
- (c) Wait for explicit "go" on the code change AND on the next fcloud round.

## 8. Validation commands

Pre-flight (per iteration, on fcloud):
```bash
# user starts fcloud (after JWT check)
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py sync

# run pre-flight (~5–10 min)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && source prepare_env.sh && \
   python3 preflight_medusa_vs_ngram.py 2>&1 | tee /tmp/preflight_iterN.log'

# pull report
python3 scripts/fcloud/fcloud_exec.py download \
  --remote /tmp/preflight_iterN.log \
  --local benchmark/soar/demo_sala/preflight_iterN.log

python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

Final eval (only after zero critical diffs):
```bash
python3 scripts/fcloud/fcloud_workflow.py start-instance
python3 scripts/fcloud/fcloud_workflow.py full     # sync + restart + accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
python3 scripts/fcloud/fcloud_workflow.py pause-instance
```

## 9. Rollback

If the protocol exits without a working ndt=2:
- Revert any Stage 3b-related edits to `medusa_worker.py` (current baseline = `a489d78d4` Stage 3a).
- Leave Stage 3a shipping config unchanged.
- Document the failed-direction analysis in §5 and open the next CHANGE_xxxx exploring an alternative (e.g., Eagle path instead of Ngram path for Medusa).

## 10. Next-step suggestions (post-iteration)

- If clean ndt=2 ships with acc ≥ 78% and S1 < Stage 3a S1: extend to ndt > 2 via tree drafts (Stage 4 territory, K=2/K=4).
- If clean ndt=2 ships but speed gain is marginal: profile the verify forward to confirm the kernel overhead vs bonus-only baseline — if verify_forward / decode_forward > 2× we cannot win at K=1.
