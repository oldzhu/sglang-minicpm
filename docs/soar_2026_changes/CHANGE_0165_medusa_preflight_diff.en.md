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

### §5.1 Iteration 1 — Stage 3a baseline (commit f375082a2 + 19740212d)

**Setup**
- Driver: `benchmark/soar/demo_sala/preflight_drive.sh {ngram|medusa}` (commits d9394bec6 → 19740212d).
  - Strips `--enable-torch-compile` and `--torch-compile-max-bs N` and forces `--disable-cuda-graph` to bring server startup from ~15 min (full CUDA-graph capture across 16 buckets) to ~2 min — pre-flight only needs ONE verify step.
  - Each mode launches its own sglang server with `bs=1`, `ndt={2 for ngram, H+1=2 for medusa-stage-3a (but MedusaWorker overrides to 1 internally)}`.
- Probe prompt: `"Question: What is the capital city of France?\nAnswer:"`, `temperature=0.0`, `max_new_tokens=3`.
- Dumps captured at three phases on each side: `pre_verify`, `post_forward`, `post_verify`.
  - `/tmp/dump_ngram.pkl` = 41908 B (27 records: 9 verify steps × 3 phases)
  - `/tmp/dump_medusa.pkl` = 41207 B (27 records)
- Diff tool: `benchmark/soar/demo_sala/preflight_diff.py --ngram … --medusa …` (compares **first** verify step on each side).

**Findings (13 fields differ)**

| Phase | Field | ngram | medusa (stage 3a) | Severity | Decision | Why |
|---|---|---|---|---|---|---|
| pre_verify | `draft_token_num` | 2 | 1 | **critical / root cause** | clean in iter 2 | MedusaWorker hard-codes `ndt=1` regardless of `--speculative-num-draft-tokens`. **All other shape diffs cascade from this.** |
| pre_verify | `input_ids` | shape (2,) `[11225, 0]` | shape (1,) `[11225]` | critical (cascade) | clean by ndt=2 | extra slot = padded draft token `0` |
| pre_verify | `out_cache_loc` | shape (2,) `[8, 9]` | shape (1,) `[9]` | critical (cascade) | clean by ndt=2 | two KV slots vs one |
| pre_verify | `spec_draft_token` | (2,) `[11225, 0]` | (1,) `[11225]` | critical (cascade) | clean by ndt=2 | same as `input_ids` (ngram passes `output_ids[-1]` + 1 ngram-suggested token padded with `0`) |
| pre_verify | `spec_positions` | (2,) `[7, 8]` | (1,) `[8]` | critical (cascade) | clean by ndt=2 | positions span verifies 2 slots from current `seq_len` |
| pre_verify | `spec_custom_mask` | (18,) | (8,) | critical (cascade) | clean by ndt=2 | full-mask layout: `(seq_len + ndt) * ndt` ⇒ `(7+2)*2=18` vs `(8+1)*1=9`… actually `8*1=8`. Encodes diagonal-causal mask over the `ndt` query slots × `seq_len + ndt` key slots. |
| pre_verify | `spec_retrive_index` | (1, 2) `[0, 1]` | (1, 1) `[0]` | critical (cascade) | clean by ndt=2 | tree retrieval index, K=1 chain shape = `(bs, ndt)` |
| pre_verify | `spec_retrive_next_token` | (1, 2) `[1, -1]` | (1, 1) `[-1]` | critical (cascade) | clean by ndt=2 | chain pointer (0→1→leaf) |
| pre_verify | `spec_retrive_next_sibling` | (1, 2) `[-1, -1]` | (1, 1) `[-1]` | critical (cascade) | clean by ndt=2 | no siblings for chain |
| pre_verify | `seq_lens` / `seq_lens_cpu` | 7 | 8 | cosmetic / derived | accept | Side-effect of `ndt=1` running an extra decode iteration to consume what `ndt=2` would have consumed in one verify. Not an algorithmic bug, just a step-counting offset between the two paths. |
| post_forward | `logits_argmax` | (2,) `[72, 72]` | (1,) `[72]` | critical (cascade) | clean by ndt=2 | direct consequence of input shape diff; **values agree on the overlapping slot** (`72`) → forward pass is correct given inputs. |
| post_forward | `logits_shape` | `(2, 73448)` | `(1, 73448)` | critical (cascade) | clean by ndt=2 | same |
| post_verify | `accept_length`, `accepted_indices`, `next_token_ids`, `num_accepted_tokens` | all equal | all equal | n/a (clean) | keep | Verify step itself agrees on the accepted output (`next_token=72`, accept 0 bonus). Confirms verify-walk logic is **not** the bug. |

**Conclusion**

- Single root cause: `MedusaWorker.draft_token_num = 1` in Stage 3a. Every shape diff cascades from this. No verify-walk / tree-index / scoring logic bug — when fed identical-shape inputs the model produces matching logits and the verify step agrees on the output.
- This is the **expected** Iter 1 baseline (Stage 3a unchanged per CHANGE_0165 §0 plan): we needed empirical confirmation that the diffs reduce to one root cause before touching code.

**Iter 2 plan**

1. Modify `MedusaWorker` to: (a) honor `draft_token_num=2` (matching `--speculative-num-draft-tokens 2`); (b) build `spec_positions`, `spec_custom_mask`, `spec_retrive_*` via the same `reconstruct_indices_from_tree_mask` kernel call NgramWorker uses; (c) supply `[output_ids[-1], 0]` (padding) as draft tokens for the K=1 chain — this matches Ngram's draft-content shape and keeps the Stage 3a semantics (head 0 not actually consulted yet; trained-head path is Stage 3c).
2. Re-run `preflight_drive.sh medusa`, expect §5.2 to report **0 critical diffs** in `pre_verify` and `post_forward`, and equal `post_verify`.
3. Only then run full accuracy + speed eval.

**Artifacts**

- Dumps: `/tmp/dump_ngram.pkl`, `/tmp/dump_medusa.pkl` on fcloud (also mirrored to `/tmp/` locally).
- Diff log: `/tmp/iter1_diff.txt`.
- Server logs: `/tmp/server_ngram.log`, `/tmp/server_medusa.log` on fcloud.
- Commits this iter: `d9394bec6` (preflight infra), `11555decd` (driver), `283c8a106` (set -e fix), `f375082a2` (MODEL_PATH export), `19740212d` (disable graph+compile).

### §5.2 Iteration 2 — ndt=2 + kernel-built retrive_* + full mask (commit `89a30d5d0`)

**Setup**

- Code change (single commit, `python/sglang/srt/speculative/medusa_worker.py`):
  1. `self.draft_token_num = self.num_heads + 1` (=2 for K=1) instead of `=num_heads`.
  2. `_forward_verify_k1` rewritten to mirror NgramWorker layout:
     - Build per-req draft chain `[base, head_pred or 0]` (Stage 3a: head untrained → `head_pred=0`).
     - Build tri `(ndt, ndt)` lower-triangular tree mask, flatten as `(ndt, seq_len-1+ndt)` per req (USE_FULL_MASK style), concat across batch.
     - Allocate `positions`, `retrive_index`, `retrive_next_token`, `retrive_next_sibling` and let `sgl_kernel.speculative.reconstruct_indices_from_tree_mask` populate them — same canonical kernel NgramWorker uses.
  3. Hidden-state slicing for trained heads: `hs.view(bs, ndt, -1)[:, 0, :]` since `hs.shape[0] == bs * ndt = 2` now.
- Pre-flight run: same driver, same prompt. Ngram dump 41908 B / 27 records; **Medusa dump 23529 B / 15 records** (one fewer verify step because scheduler crashed on idle KV check after the probe — this is post-probe and does NOT affect the captured records: phases pre/post/post are intact).

**Findings (5 fields differ — 13 → 5)**

| Phase | Field | ngram | medusa (iter 2) | Severity | Diagnosis |
|---|---|---|---|---|---|
| pre_verify | `draft_token_num` | 2 | **2** | ✅ EQUAL | Fix #1 confirmed. |
| pre_verify | `input_ids` | (2,) `[11225, 0]` | (2,) `[11225, 0]` | ✅ EQUAL | ndt=2 layout adopted; pad=0 matches. |
| pre_verify | `spec_draft_token` | (2,) `[11225, 0]` | (2,) `[11225, 0]` | ✅ EQUAL | Stage 3a head returns 0 → identical to ngram fallback. |
| pre_verify | `spec_custom_mask` | (18,) | (18,) | ✅ EQUAL | USE_FULL_MASK + tri lowered to identical layout. |
| pre_verify | `spec_retrive_index` / `next_token` / `next_sibling` | (1,2) | (1,2) | ✅ EQUAL | `reconstruct_indices_from_tree_mask` kernel produces identical tree. |
| pre_verify | `seq_lens` / `seq_lens_cpu` | 7 | **8** | semantic off-by-one | **Remaining root cause.** MedusaWorker's verify is invoked when the request's `seq_lens` is already 1 ahead of ngram's. |
| pre_verify | `spec_positions` | `[7, 8]` | `[8, 9]` | cascade of seq_lens | kernel writes `[seq_len-1, seq_len, ..., seq_len-2+ndt]` — values shifted by +1 because seq_lens=8. |
| pre_verify | `out_cache_loc` | `[8, 9]` | `[9, 10]` | cascade of seq_lens | KV allocator hands out the next 2 free slots — also +1 because medusa has one more committed slot. |
| post_forward | `logits_argmax[0]` | 72 | 72 | ✅ equal at slot 0 | Forward over the (now identical-shape) input produces matching token at the base position. |
| post_forward | `logits_argmax[1]` | 72 | **59320** | cascade of seq_lens | Model is invoked at position 9 vs 8 with different KV history depth → different speculative slot logits. |
| post_forward | `logits_shape` | (2, 73448) | (2, 73448) | ✅ EQUAL | Layout fixed. |
| post_verify | accept_length / accepted_indices / next_token_ids / num_accepted_tokens | all equal | all equal | ✅ EQUAL | Despite the slot-1 logits diverging, accept_length=0 (no draft accepted because draft was `0`) → next_token=72 on both. Verify walk is clean. |

**Conclusion**

- Iter 2 delivers exactly what the proposal promised for the layout fix: **8 previously-broken fields now align** (`draft_token_num`, `input_ids`, `spec_draft_token`, `spec_custom_mask`, `spec_retrive_index`, `spec_retrive_next_token`, `spec_retrive_next_sibling`, `logits_shape`). `post_verify` remains 4×EQUAL.
- The **single remaining root cause** is `seq_lens` ahead by 1 in MedusaWorker. This is the long-standing positional off-by-one that CHANGE_0160 / CHANGE_0161 / `PROPOSAL_medusa_k1_positional_offbyone_fix` have been circling around. Now the pre-flight harness pins it to **a single field** (`seq_lens`) in a single phase (`pre_verify`) — eliminating ambiguity from layout cascade.

**Where does the extra +1 come from?**

Hypotheses (to be tested in iter 3):
1. **H1 (most likely)** — Stage 3a-style "bonus position" KV write: when MedusaWorker enters its verify path, the previous decode step appended the bonus token's KV at `seq_len`, advancing `seq_lens` by 1 before `prepare_for_verify` runs. NgramWorker doesn't write bonus KV during decode, so it stays at the pre-verify seq_len.
2. **H2** — `prepare_for_verify` in MedusaWorker increments `seq_lens` by `+1` instead of leaving it untouched (the increment-by-ndt is supposed to happen during the verify forward, not before).
3. **H3** — The probe is the *second* batch step (after the initial decode that emitted token `11225`), and Medusa's "extend" mode for that initial decode incorrectly committed two slots.

The pre-flight harness can disambiguate by dumping a 4th phase: `pre_prepare_for_verify` (state immediately before `spec_info.prepare_for_verify`). That isolates whether the +1 comes from decode-side bookkeeping (H1/H3 → bug visible at this earlier phase) or `prepare_for_verify` itself (H2 → diff appears only after).

**Iter 3 plan (proposal — needs user "go")**

1. Add `phase="pre_prepare_for_verify"` dump in both workers, capturing `seq_lens`, `seq_lens_cpu`, `out_cache_loc` (whatever is already allocated), `req_pool_indices`.
2. Re-run preflight, diff.
3. If H1/H3: locate where MedusaWorker's main-decode commits the bonus slot → either skip the commit OR shift verify's `seq_lens` back by 1 before `prepare_for_verify`. Per CHANGE_0160/0161, the safer path is the latter (subtract 1 from `batch.seq_lens` for the verify-input view) — but the **correct** fix is to not double-count the bonus slot in the first place.
4. If H2: simplify `prepare_for_verify` to remove the spurious increment.
5. Iterate to 0 critical diffs in pre_verify; only then run accuracy.

**Pass criterion for iter 3**: `seq_lens`, `seq_lens_cpu`, `spec_positions`, `out_cache_loc`, `logits_argmax` all EQUAL between ngram and medusa. `post_verify` already-equal must remain equal.

**Artifacts**

- Dumps: `/tmp/dump_ngram.pkl`, `/tmp/dump_medusa.pkl` on fcloud (this iter).
- Diff log: `/tmp/iter2_diff.txt` (also mirrored to local `/tmp/`).
- Commit this iter: `89a30d5d0` (medusa: preflight iter 2 — adopt ndt=2 + kernel-built retrive_* + full mask).

### 5.3 Iter 3 result — bonus-slot bookkeeping isolated

**Goal**: Disambiguate H1/H3 (upstream KV write before verify-prep) vs H2 (prepare_for_verify increments) by adding a fourth dump phase `pre_prepare_for_verify` to both workers, capturing the batch state just before `spec_info.prepare_for_verify(batch, page_size)` is called.

**Code changes** (commits `3d2435d26` + `346c3f666`):
- `ngram_worker.py`: dump (`seq_lens`, `seq_lens_cpu`, `out_cache_loc`, `req_pool_indices`) just before `batch.spec_info.prepare_for_verify(...)`.
- `medusa_worker.py`: same dump just before `spec_info.prepare_for_verify(...)`.
- `preflight_diff.py`: register the new phase in `PHASE_ORDER` and argparse `choices`.

**Result (5 differing fields at pre_verify; 3 at pre_prepare_for_verify)**:

| phase | field | ngram | medusa | status |
|---|---|---|---|---|
| pre_prepare_for_verify | `seq_lens` | `[7]` | `[8]` | **DIFFER** |
| pre_prepare_for_verify | `seq_lens_cpu` | `[7]` | `[8]` | **DIFFER** |
| pre_prepare_for_verify | `out_cache_loc` | `[1,2,3,4,5,6,7]` (shape 7) | `[8]` (shape 1) | **DIFFER (shape+values)** |
| pre_prepare_for_verify | `batch_size`, `req_pool_indices` | — | — | EQUAL |

**Interpretation — H1/H3 confirmed, H2 ruled out**:

The `seq_lens` +1 is already present **before** `prepare_for_verify` runs. Therefore `prepare_for_verify` is innocent; the bonus-slot commit happens **upstream**, in the path between EXTEND completion and the next iteration's DECODE/verify entry.

**Root cause located** in [`schedule_batch.py`](../../python/sglang/srt/managers/schedule_batch.py) `prepare_for_decode()` line 1948:

```python
def prepare_for_decode(self):
    ...
    if not self.spec_algorithm.is_none():
        return  # spec workers manage their own decode-prep
    # else: alloc 1 slot, seq_lens.add_(1), kv_committed_len += 1
```

And the asymmetry between workers' EXTEND paths:
- **NgramWorker** EXTEND: `_prepare_for_speculative_decoding` returns early; `spec_algorithm` stays `NGRAM` → next iteration's `prepare_for_decode` returns early → `seq_lens` stays at 7.
- **MedusaWorker** EXTEND: explicitly sets `batch.spec_algorithm = SpeculativeAlgorithm.NONE` → next iteration's `prepare_for_decode` falls through → allocates 1 slot, advances `seq_lens` to 8, increments `kv_committed_len`. By the time MedusaWorker re-enters DECODE and rebuilds spec_info, the bonus slot is already committed.

This is exactly the symptom CHANGE_0160 / CHANGE_0161 chased (the "+1" mystery), now precisely pinpointed.

**Artifacts**

- Dumps: `/tmp/dump_ngram.pkl` (52937B / 36 records), `/tmp/dump_medusa.pkl` (29590B / 20 records) on fcloud.
- Diff logs: `/tmp/iter3_diff.txt` (all phases), `/tmp/iter3_pre_prep.txt` (pre_prepare_for_verify only).
- Commits this iter: `3d2435d26` (iter3 dump instrumentation), `a87da57dd` + `346c3f666` (preflight_diff phase registration fix).

### 5.4 Iter 4 plan — drop the spec_algorithm reset in MedusaWorker EXTEND

**Proposed fix** (single line in `medusa_worker.py::forward_batch_generation`):

```python
# EXTEND path
if batch.forward_mode.is_extend():
-    batch.spec_algorithm = SpeculativeAlgorithm.NONE   # drop this
    model_worker_batch = batch.get_model_worker_batch()
    batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
    return GenerationBatchResult(...)
```

**Why this is safe**:
1. NgramWorker (the canonical reference) does not reset `spec_algorithm` during extend.
2. The `model_worker_batch` for EXTEND is built with `forward_mode=EXTEND`, so the target model runs normal extend regardless of `spec_algorithm`.
3. Keeping `spec_algorithm=NGRAM` makes the scheduler's `prepare_for_decode` take the early-return path (matching ngram), which leaves KV/seq_lens management entirely to `prepare_for_verify` — the intended design.

**Validation plan**:
1. Apply fix, push.
2. Re-run `preflight_drive.sh ngram` + `preflight_drive.sh medusa` on fcloud.
3. Re-run `preflight_diff.py` — expect all phases (including pre_prepare_for_verify, pre_verify, post_forward, post_verify) to show **0 differing fields**.
4. **Pass criterion**: total differing fields across all 4 phases = 0.

**Risks**:
- R-iter4-1: If some scheduler path branches on `spec_algorithm.is_none()` specifically for the extend output (e.g., output_processor's `next_token_ids.tolist()` at line 372), behavior may shift. Mitigation: read the EXTEND-result branches in `scheduler_output_processor_mixin.py` before applying; if the path under `is_none()` does something MedusaWorker actually needs, find an alternative (e.g., wrap target_worker call to manually fetch the per-req append).

Followup once preflight diff hits zero:
- Re-enable cuda-graph + torch-compile by swapping `preflight_drive.sh` for the standard `fcloud_workflow.py restart-server` flow.
- Run full accuracy eval + speed S1/S8/Smax (requires explicit user confirm per project rules).

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
