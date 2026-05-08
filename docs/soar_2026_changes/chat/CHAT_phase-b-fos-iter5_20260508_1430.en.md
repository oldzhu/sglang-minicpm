# CHAT — Phase B FOS iter-5 + fcloud setup git-clone fix

Date: 2026-05-08 14:30 (start)
Topic: NVFP4-FOS Option A (re-quant SAMPLES=32 sequential), and replace
fcloud setup tarball upload with real git clone.

## User asks (round 1, verbatim)

> go and besides that, pls also git clone from https://github.com/oldzhu/sglang-minicpm.git
> at /root/sglang-minicpm so that we could git pull and synce from the
> remote repo which our local change pushed default. Pls also add a note
> into the fcloud instance environment re-setup so that furure re-setup
> would create the clone at /root/sglang-minicpm too.

Decoded into 3 parallel goals:
1. **go** = proceed with iter-4 follow-up Option A (re-quant `SAMPLES=32
   sequential` to mimic iter-1).
2. Replace the broken `/root/sglang-minicpm` (tarball extract, no `.git`)
   with a real git clone so `fcloud_workflow.py sync` uses `git pull`.
3. Update setup docs/script so future re-setups always do a real git clone.

## Agent actions

| Step | Action | Outcome |
|------|--------|---------|
| 1 | Patch `scripts/fcloud/fcloud_workflow.py::step_setup` Step 2: replace tarball upload + extract with `git clone --depth 1 --branch mixed_minicpm_cudagraph https://github.com/oldzhu/sglang-minicpm.git /root/sglang-minicpm` (timeout 600s) | Local edit ✓ |
| 2 | Patch `.github/copilot-instructions.md`: update setup Step 2 + add manual fallback recipe for instances missing `.git/` | Local edit ✓ |
| 3 | `start-instance` (console API) | HTTP 200, JupyterLab READY after ~3 min |
| 4 | Manual remediation: `rm -rf /root/sglang-minicpm && git clone --depth 1 …` | First call returned exit 0 but left half-initialized `.git/`; retry as background job; final `git rev-parse HEAD` = `6ae5ef33f59a7507e9a5380f6b49e08159ae2815` (matches latest pushed commit before this round) ✓ |
| 5 | Improve `step_sync` to handle pre==post sha case (no new commits → `(no new commits)` instead of force-copy) | Local edit ✓ |
| 6 | Verify `fcloud_workflow.py sync` | Reports `[git pull] Already up to date.` and `(no new commits)` ✓ |
| 7 | Commit + push 2 patches | `e41ecb518` pushed to `minicpm-src/mixed_minicpm_cudagraph` ✓ |
| 8 | Re-run `sync` after the new push | Pulls `e41ecb518`, reports `[changed files]` correctly, no force-copy ✓ |
| 9 | Iter-5 quant: background `preprocess_model.py --mode nvfp4` with `SAMPLES=32 SAMPLING=sequential MAX_CALIB_SEQ_LEN=4096 FOS=1` | Completed in ~1 min; pct_m4=43.14%; **but tokenizer files missing** in dst ✓⚠ |
| 10 | First server start (with `--quant-mode noquant`) | FAILED: `ValueError: Unrecognized configuration class …MiniCPMSALAConfig` (tokenizer missing) |
| 11 | Manual fix: copy `tokenizer*` + `special_tokens_map.json` from `/root/models/openbmb/MiniCPM-SALA` to dst; re-run `_patch_chat_template_for_mcq` | ✓ |
| 12 | Second server start (with `--quant-mode noquant`) | FAILED: `ModelOptFp8Config only supports static FP8 quantization` (wrong `--quantization` flag because `noquant` branch omits it) |
| 13 | Third server start with `--quant-mode gptq` (which selects `--quantization modelopt_fp4` via `SOAR_QUANT_PROFILE=nvfp4_fos`) | Server READY ✓ |
| 14 | Accuracy run-1 (single pass) | **71.24%** ori_accuracy ✓ |
| 15 | Pause instance | OK on retry (first attempt 504, second 200) ✓ |
| 16 | Document iter-5 in `CHANGE_0151_phase_b_four_over_six_004.{en,zh}.md` + this chat log + TEST_RESULTS_TRACKING `NVFP4-FOS-5` | this commit |

## Outcomes

- **Iter-5 result**: 71.24% ori_accuracy. Passes the >70% abort gate. +5.24pt
  vs iter-4 (66.00%). Residual ~1.9pt gap to iter-1 (~73.13%).
- **Conclusion**: within the qa,mcq,cwe-restricted calibration pool,
  `SAMPLES=32 sequential` is materially better than `SAMPLES=90 stratified`
  for NVFP4 weight calibration. Calibration *content/selection* dominates;
  calib_seq_len (iter-4) was a red herring.
- **Infrastructure win**: `fcloud_workflow.py sync` now uses `git pull`
  end-to-end after the real-clone fix; force-copy fallback retained but
  no longer the default. Future re-setups will create the clone correctly.
- **Open issue (carried to next iter)**: `preprocess_model.py` does not
  save tokenizer files in NVFP4 streaming-export path; manual cp required.
  Fix recommended as Option C in iter-6 plan.

## Files created/modified in this round

| Path | Type | Summary |
|------|------|---------|
| `scripts/fcloud/fcloud_workflow.py` | code edit (committed `e41ecb518`) | step_setup now uses git clone; step_sync handles no-new-commits cleanly |
| `.github/copilot-instructions.md` | doc edit (committed `e41ecb518`) | setup Step 2 doc + manual fallback recipe |
| `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_004.en.md` | new doc (this commit) | iter-5 results, comparison, next-step options |
| `docs/soar_2026_changes/CHANGE_0151_phase_b_four_over_six_004.zh.md` | new doc (this commit) | Chinese mirror |
| `docs/soar_2026_changes/chat/CHAT_phase-b-fos-iter5_20260508_1430.{en,zh}.md` | new docs (this commit) | this chat log |
| `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` | row append (this commit) | `NVFP4-FOS-5` row, 71.24% |

## Open follow-ups

1. (Code) Add `tokenizer.save_pretrained(dst)` inside
   `run_nvfp4_quantization` so NVFP4 quants are self-contained.
2. (Test) Iter-5 run-2 + S1/S8/Smax speed bench to confirm reproducibility
   and measure speed footprint of the FOS-32 ckpt.
3. (Test) Iter-6 = re-quant `SAMPLES=32 sequential FOS=0`; if it reaches
   ≥73% then FOS itself is the residual regression source and should be
   permanently parked.

## Cross-references

- iter-4 doc: [CHANGE_0151_phase_b_four_over_six_003.en.md](../CHANGE_0151_phase_b_four_over_six_003.en.md)
- iter-5 doc (this iter): [CHANGE_0151_phase_b_four_over_six_004.en.md](../CHANGE_0151_phase_b_four_over_six_004.en.md)
- Setup script commit: `e41ecb518` (`minicpm-src/mixed_minicpm_cudagraph`)
