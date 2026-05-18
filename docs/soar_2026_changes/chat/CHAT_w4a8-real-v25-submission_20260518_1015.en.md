# CHAT: W4A8 REAL v25 Submission Preparation — 2026-05-18

## Session Info
- **Date**: 2026-05-18
- **Topic**: w4a8-real-v25-submission
- **Commits**: ff8894120, 377e54197, 16d1919d7

## 1. User Requests

1. "let us prepare a new tar.gz for a new official submission and then move to fused optimization" — User requested v25 submission tarball creation.

## 2. Agent Actions

### Wheel Rebuild
- Problem: Earlier wheel rebuild only contained .so files, missing Python `.dist-info` and all Python source files (`__init__.py`, etc.)
- Root cause: The `unzip` of old wheel failed, so repack only had manually-copied .so files
- Solution: Rebuilt wheel from source using `make build` on fcloud (incremental, ~5min with warm ccache)
- Result: Full 550MB wheel with all Python + .so + .dist-info files

### Kernel Op Verification
- Verified `torch.ops.sgl_kernel.gptq_int4_to_fp8_blockwise` after source build
- Confirmed `import sgl_kernel` is REQUIRED before torch.ops call (TORCH_LIBRARY_FRAGMENT static init)

### Env Default
- Changed `SOAR_W4A8_REAL_FP8_GEMM` default from `0` to `1` in prepare_env.sh
- Updated comment: "Default ON (v25). Set SOAR_W4A8_REAL_FP8_GEMM=0 to disable."

### Tarball Creation
- Source: `/root/submission_sim/` on fcloud
- Command: `tar --exclude='__pycache__' --exclude='*.pyc' -czf /root/minicpm_sala_submit_v25.tar.gz sgl_kernel-*.whl prepare_env.sh prepare_model.sh preprocess_model.py perf_public_set.jsonl sglang`
- Size: 529MB compressed
- Contents verified: 1823 files, includes wheel, sglang source, scripts

### Tarball Download
- Copied to `/workspace/` (JupyterLab root)
- Downloaded via JupyterLab API: `GET /api/contents/minicpm_sala_submit_v25.tar.gz`
- Saved to: `benchmark/soar/demo_sala/minicpm_sala_submit_v25.tar.gz` (554MB)

### Documentation
- Created `CHANGE_W4A8_REAL_FP8_GEMM.en.md` and `.zh.md`
- Updated `TEST_RESULTS_TRACKING.md` with v25 results

### Cost Saving
- Paused fcloud instance via `pause-instance` (console API)

## 3. Key Decisions

1. **Source build for wheel**: Chose `make build` over manual zipfile repack to ensure complete wheel
2. **Env default ON**: v25 ships with W4A8 REAL enabled by default since accuracy/speed are superior
3. **Pause instance**: Paused fcloud after tarball download to save costs while working on docs

## 4. Issues Encountered

1. **Multi-line exec syntax errors**: fcloud_exec.py heredoc approach fails with bash syntax errors for multi-line scripts. Solution: write scripts to file first, then execute.
2. **Wheel missing .dist-info**: Earlier manual zipfile rebuild only included .so files. Fixed by full source rebuild.
3. **Console 504 on first pause attempt**: Transient gateway timeout; retry succeeded.

## 5. Results Summary

| Metric | Test 12 Baseline | v25 W4A8 REAL |
|--------|-----------------|---------------|
| Accuracy | 79.29% | **81.07%** |
| S1 | 121.71s | **110.79s** |
| S8 | 44.09s | **40.51s** |
| Smax | 35.86s | **32.67s** |
| C | 1.0 | **1.0** |

## 6. Cross-References

- `CHANGE_W4A8_REAL_FP8_GEMM.en.md` — Full implementation documentation
- `CHANGE_W4A8_REAL_FP8_GEMM.zh.md` — Chinese version
- `TEST_RESULTS_TRACKING.md` — Updated with v25 entries
- `benchmark/soar/demo_sala/prepare_env.sh` — Default changed to SOAR_W4A8_REAL_FP8_GEMM=1
- `benchmark/soar/demo_sala/minicpm_sala_submit_v25.tar.gz` — Submission tarball (529MB)

## 7. Next Steps (Pending)

1. Upload `minicpm_sala_submit_v25.tar.gz` to SOAR official submission site
2. Fused GEMM dequant kernel (CHANGE_W4A8_FUSED): expected 3-4 day effort, 5-10% further S1 improvement
3. Track official leaderboard after v25 submission
