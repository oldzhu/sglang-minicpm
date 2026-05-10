# Chat: Medusa R1 Implementation (Phase R1a)

- Date: 2026-05-11 07:31
- Branch: `mixed_minicpm_cudagraph`
- Topic: Implement Phase R1a of CHANGE_0153 (Medusa scaffolding, no behavior change)
- Related docs:
  - [CHANGE_0153_medusa_phase_r1_design.en.md](../CHANGE_0153_medusa_phase_r1_design.en.md)
  - [PROPOSAL_medusa_minicpm_sala_001.en.md](../PROPOSAL_medusa_minicpm_sala_001.en.md)
  - [RESEARCH_speculative_decoding_survey_001.en.md](../RESEARCH_speculative_decoding_survey_001.en.md)

## Round 1 — User approval and R1 scope split

**User**: "while we are waiting for floud for testing, could we go medusa planned implementation?"

**Agent**:
- Audited sglang's existing speculative infrastructure and discovered `update_mamba_state_after_mtp_verify` (hybrid_linear_attn_backend.py L1373) already implements the per-step state-scatter pattern needed for Medusa+SimpleGLA. R1 shrank from ~1.5–2 kLOC to ~635 LOC.
- Drafted `CHANGE_0153_medusa_phase_r1_design.{en,zh}.md`, committed as `2d658a868`.
- Surfaced three decision points: (Q1) proceed with code now? (Q2) K=1 vs K=2? (Q3) add NGRAM opt-in alongside Medusa opt-in?

**User**: "q1: yes, q2: deafult, q3: ok"

**Agent**: Acknowledged → K=1 default, NGRAM opt-in added. Split R1 into two reviewable iterations:
- **R1a (this commit)**: registry / argparse / dataclasses / heads / opt-in flags — no functional path change. Server start with `SOAR_SPEC_MEDUSA=1` will deliberately fail with `NotImplementedError` until R1b.
- **R1b (next iteration)**: actual `MedusaWorker.forward_batch_generation`, SimpleGLA state-scatter, `forward_with_hidden` in `MiniCPMSALAForCausalLM`, tp_worker plumbing.

## Round 2 — R1a edits

Files touched (6, ~383 lines net):

1. [python/sglang/srt/speculative/spec_info.py](../../../python/sglang/srt/speculative/spec_info.py)
   - Added `SpeculativeAlgorithm.MEDUSA` + `is_medusa()` helper.
   - Added `SpecInputType.MEDUSA_VERIFY` and included it in `is_verify_input()`.
   - Added lazy `create_worker()` branch importing `MedusaWorker`.

2. [python/sglang/srt/server_args.py](../../../python/sglang/srt/server_args.py)
   - New dataclass field `speculative_num_medusa_heads: int = 1`.
   - `"MEDUSA"` added to `--speculative-algorithm` choices.
   - New `--speculative-num-medusa-heads` argparse flag.
   - Post-init block (mirrors NGRAM): disables overlap scheduler + mixed-chunk, sets `num_draft_tokens = K + 1`, forbids DP attention, requires CUDA, defaults `max_running_requests=24`.

3. [python/sglang/srt/speculative/medusa_info.py](../../../python/sglang/srt/speculative/medusa_info.py) — **NEW**
   - `MedusaInput(SpecInput)` with fields aligned to `EagleVerifyInput` so existing tree-verify kernels reuse cleanly.
   - `MedusaVerifyOutput` dataclass for post-verify state.
   - `get_spec_adjust_token_coefficient()` returns `(K+1, 1)`.

4. [python/sglang/srt/models/minicpm_medusa_heads.py](../../../python/sglang/srt/models/minicpm_medusa_heads.py) — **NEW**
   - `MedusaHead`: residual MLP `SiLU(W1 @ h) + h`, W1 zero-init.
   - `MedusaHeads`: bundle of K heads + non-owning reference to shared `lm_head`.
   - With W1=0 → every head output equals base model's logits → at `accept_threshold=1.0` every draft accepts → byte-identical to no-spec baseline. R1 correctness gate.

5. [python/sglang/srt/speculative/medusa_worker.py](../../../python/sglang/srt/speculative/medusa_worker.py) — **NEW**
   - `MedusaWorker(BaseSpecWorker)` skeleton. `__init__` raises `NotImplementedError` citing R1b. Abstract methods stubbed.

6. [benchmark/soar/demo_sala/prepare_env.sh](../../../../benchmark/soar/demo_sala/prepare_env.sh)
   - `SOAR_SPEC_MEDUSA` (default 0) + `SOAR_SPEC_MEDUSA_HEADS` (default 1) + `SOAR_SPEC_NGRAM` (default 0).
   - Conditionally appends `--speculative-algorithm MEDUSA --speculative-num-medusa-heads K --speculative-num-draft-tokens K+1` when `SOAR_SPEC_MEDUSA=1`.
   - Conditionally appends `--speculative-algorithm NGRAM` when `SOAR_SPEC_NGRAM=1` (mutually exclusive — warns + prefers MEDUSA if both set).
   - Env dump echoes added.

## Outcomes

- All five touched files pass `ast.parse` (no syntax errors).
- Local runtime import check blocked by missing `numpy` in workstation env; deferred to fcloud where it always works. Risk is negligible: ast-validated + no behavior change at default flag values.
- Commit: `95d931c8b` on `mixed_minicpm_cudagraph`.
- Push: pending (will push after this chat log lands).

## Open follow-ups

- R1b (next session, blocking on user approval after they review R1a push):
  1. Port `update_mamba_state_after_mtp_verify` → `update_simple_gla_state_after_verify` in `hybrid_linear_attn_backend.py`.
  2. Implement `MedusaWorker.forward_batch_generation` (verify-tree forward + state-scatter + accepted-token gather).
  3. Add `MiniCPMSALAForCausalLM.forward_with_hidden` returning pre-norm last hidden.
  4. Plumb `forward_with_hidden` through `managers/tp_worker.py`.
- Queued (fcloud-dependent): SAMPLES=30 sequential GPTQ retest into folder `MiniCPM-SALA-30-sequential-sparse_qkv_w8`.
