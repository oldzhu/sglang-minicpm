# CHAT — Round 13e Option-B sparse-line profile (2026-04-29 ~12:12 local)

Companion chat log for `CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.{en,zh}.md`.

## User asks (this round)
1. "approve" — green-light CHANGE_0135 (drop `--dense-as-sparse` from prepare_env.sh + run Option-B single-request torch profile at 32k / 64k / 128k under sparse routing).
2. "go fcloud started" — fcloud instance is up; proceed with the profile run.

## Agent actions
1. `git push minicpm-src mixed_minicpm_cudagraph` — pushed CHANGE_0135 (commit `f4097eef6`: doc pair + `prepare_env.sh` cleanup) so fcloud can pull it.
2. `python3 scripts/fcloud/fcloud_workflow.py sync` — synced.
3. Cleared old `/root/profile_round13e/`; killed any stale server.
4. Wrote `/root/launch_profile_server.sh` (sources `prepare_env.sh`, sets `SOAR_QUANT_MODE=noquant` + `SGLANG_TORCH_PROFILER_DIR=/root/profile_round13e/`, exec sglang launcher).
5. First server start failed: `command 'clang' failed: No such file or directory` while `uv pip install --force-reinstall pypcre`. Cause: Python 3.10.19 sysconfig sets `CC=clang -pthread`, fcloud image only has `gcc`. Workaround applied at runtime: `ln -sf /usr/bin/gcc /usr/local/bin/clang && ln -sf /usr/bin/g++ /usr/local/bin/clang++`. Server then booted in ~36 s, healthy.
6. Wrote local `profile_driver.py`, base64-uploaded to `/root/profile_driver.py`. Driver reads `/root/data/perf_public_set.jsonl`, picks the sample with `prompt_tokens` nearest the target, wraps `/generate` POST with `/start_profile` + `/stop_profile`, `max_new_tokens=64`.
7. Ran three single-request profiles: 32k (sample 129, 9.7 s), 64k (sample 49, 24.7 s), 128k (sample 149, 49.3 s). Renamed each emitted trace to `round13e_<size>.trace.json.gz` (sizes 56 / 112 / 211 MB).
8. Uploaded `analyze_profile.py`; ran in background; collected the per-trace top-25 + category breakdowns into `/root/analyze.out` and copied to `docs/soar_2026_changes/profile_data/round13e_analyze.txt`.
9. Ran `python3 scripts/fcloud/fcloud_workflow.py shutdown` per cost-saving rule.
10. Wrote `CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.{en,zh}.md` (this continuation pair) capturing the full breakdown, decision-tree resolution, and recommendation.

## Outcomes / key numbers

| Length | Wall  | GPU kernel time | BF16 256×128 GEMM | sparse FA | sparse meta+topk | GPU active |
|--------|-------|-----------------|--------------------|-----------|-------------------|------------|
| 32k    |  9.7s | 5 658 ms        | 66.7%              | 3.6%      | ~2%               | 58%        |
| 64k    | 24.7s |10 344 ms        | 73.2%              | 4.0%      | ~2%               | 42%        |
| 128k   | 49.3s |19 775 ms        | 76.9%              | 4.3%      | ~2%               | 40%        |

- Bottleneck = BF16 cuTLASS GEMM (linear/MLP weights at full precision). NOT sparse attention.
- Sparse attention itself is ≤4.3% of GPU time at all three lengths. Scales linearly. Healthy.
- compress_k1/k2 fill (post-CHANGE_0133 bounded fix) is now 1.0–1.1% across all lengths — fix is sufficient.
- CHANGE_0135 decision tree resolves to **branch (d): BF16 weights are the cost** → close BF16+sparse line for SOAR submission, keep dense + GPTQ + FP8 KV baseline (Test 12) as the submission path.
- Optional follow-up directions (medium-term research, not required for next submission):
  - GPTQ + sparse + FP8 KV with re-tuned calibration (avoid historical sparse_qkv_w8 50% accuracy collapse).
  - SM120 mxfp8 / W8A8 weights for all layers under sparse routing.
- Tail kernel cleanup (8% Other) deferred — capped upside vs 77% GEMM bottleneck.

## Open follow-ups
- Fold the clang→gcc symlink fix into `prepare_env.sh` or pre-bake a pypcre wheel in `submission_sim.tar` so noquant boot doesn't need a manual workaround on fresh fcloud images.
- Trace files (≈379 MB total) remain on fcloud `/root/profile_round13e/`. Not downloaded. Reproducible via `profile_driver.py` if a TensorBoard deep-dive is needed later.

## Cross-references
- Proposal: [CHANGE_0135_sparse_path_cleanup_and_profile_plan.en.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan.en.md), [.zh.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan.zh.md)
- Result (this round): [CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md), [.zh.md](../CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.zh.md)
- Raw kernel data: [profile_data/round13e_analyze.txt](../profile_data/round13e_analyze.txt)
- Earlier related work: CHANGE_0133 (compress_k bounded fill), CHANGE_0134 (tokenizer mismatch ruled out as Round 13e timeout cause).
- Catalog: [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- Submission baseline reference: `TEST_RESULTS_TRACKING.md` Test 12 (S₁=121.71 s, S₈=44.09 s, S∞=35.86 s, ori_acc=79.29%, C=1.0).
