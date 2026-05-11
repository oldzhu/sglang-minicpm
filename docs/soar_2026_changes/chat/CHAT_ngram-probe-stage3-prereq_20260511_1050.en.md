# CHAT — NGRAM probe locates Stage 3b cuda-graph prerequisite

- **Date**: 2026-05-11 10:50 (start) → 10:55 (paused)
- **Branch / HEAD**: `mixed_minicpm_cudagraph` @ `580d86582`
- **fcloud instance**: `ai-e7e98a7c52`

## 1. User asks / decisions

- User had already approved (previous turn) running an NGRAM probe — zero new code, just `SOAR_SPEC_NGRAM=1 SOAR_SPEC_MEDUSA=0` — as a cheap de-risk to verify whether our exact runtime config (stock flashinfer + GPTQ + FP8 KV + dense + mixed-chunk + torch.compile + 16-bs cuda-graph buckets) can host any TARGET_VERIFY workload at all, before committing to writing Medusa Stage 3a code.
- Directive earlier in the session: "break the rule a little bit, I would like you working on medusa feature until it can be tested in fcloud". Probe is part of that loop.

## 2. Agent actions

1. Retried `restart-server` after first attempt failed with `unrecognized arguments: --extra-env` — correct flag is `--env KEY=VAL` (action='append', repeatable). Successful command:
   ```
   python3 scripts/fcloud/fcloud_workflow.py restart-server \
     --env SOAR_SPEC_MEDUSA=0 --env SOAR_SPEC_NGRAM=1
   ```
2. `wait-server` timed out twice at 300s — checked server logs.
3. **Outcome**: server boot reached cuda-graph capture and crashed on the bs=24 verify-shape bucket:
   ```
   File "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 515, in _capture_metadata
       if forward_mode.is_target_verify() and spec_info.topk > 1:
   AttributeError: 'NgramVerifyInput' object has no attribute 'topk'
   ```
4. Confirmed via grep that `MedusaInput` (`python/sglang/srt/speculative/medusa_info.py`) **also has neither `topk` nor `draft_token_num`** — so Stage 3b (Medusa + cuda-graph) would hit the identical crash.
5. Paused fcloud (`pause-instance` — 504 on first call, succeeded after 5s retry).

## 3. Outcomes

- **Probe ROI**: very high. ~5 minutes of fcloud time, zero new code, located the exact line that would block Stage 3b cuda-graph speed gains.
- **Stage 3a (eager) unaffected** — crash happens inside cuda-graph capture, which `SOAR_SPEC_MEDUSA_EAGER=1` disables. Safe to write.
- **Stage 3b prerequisite logged**: ~5-line tolerance patch in `hybrid_linear_attn_backend.py` (lines 515 / 570 / 575) — `getattr(spec_info, "topk", 1) > 1` and `getattr(spec_info, "draft_token_num", None) or <static_K>`. Benefits NGRAM + Medusa + any future linear-verify algorithm symmetrically. Low risk.
- **v23 submission safe** — Stage 2 medusa_worker.py is pure pass-through, never enters TARGET_VERIFY at runtime, so the hybrid-backend gap doesn't affect v23.
- **Pre-cuda-graph infrastructure validated**: model load, KV alloc, hybrid pool init, FP8 KV dtype, `--enable-fused-qk-norm-rope`, `--quantization gptq_marlin` — all succeeded. Stage 3a only needs to plug in draft+verify logic.

## 4. Lessons logged

- Cheap end-to-end probes before speculative code-writing pay off — same lesson as proposal §11 (false NotImplementedError blocker), reinforced.
- The probe also caught an internal tool bug: agent used `--extra-env` (which doesn't exist) instead of `--env`. Fixed mid-session.

## 5. Cross-references

- Proposal updated: `docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.{en,zh}.md` §12 (new section).
- Test row added: `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` → `NGRAM-probe` row above Medusa-Stage2-cgraph.
- Memory note: `/memories/repo/hybrid_backend_spec_info_shape.md` (new).

## 6. Next steps

- **Stage 3a code-up** (no infra blockers): rewrite `medusa_worker.py` from Stage 2 pass-through to draft+verify, run with `SOAR_SPEC_MEDUSA_EAGER=1`. Target: byte-equality with v22 at zero-init heads (greedy decode) → ori_accuracy ≥ 80%.
- **Stage 3b** (after 3a passes): apply the 5-line tolerance patch to `hybrid_linear_attn_backend.py`, flip EAGER off, measure speed.
