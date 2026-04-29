# CHANGE_0136 — Tunable `sparse_dense_len` runtime override (per-request dense/sparse routing threshold)

## Status: **PARKED** (2026-04-29)

Implementation landed on `minicpm-src/mixed_minicpm_cudagraph` at commits `6a070110b` (core), `45e316ee0`, `7a7a568eb` (env-side fixes) and is ready to use. Validation is **blocked** by a pre-existing sparse-path off-by-one in `MiniCPMSparseBackend.forward_extend`'s dense fallback writer — see [CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md](CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md). The `SOAR_SPARSE_DENSE_LEN=524288` sanity step crashes on the very first prefill request (size 103 vs 104). `=65536` and `=16384` would hit the same code path. Resume validation once CHANGE_0137 (and ideally CHANGE_0133) lands.

In parallel, the team is investigating whether the Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`) line can be evolved into a viable submission baseline at higher priority. See [RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md).

---

## Status: PROPOSAL (awaiting approval)

## 1. Background and motivation

The CHANGE_0135_001 single-request profile (Round 13e Option-B) proved that on the BF16 + native-sparse + FP8 KV path:

- BF16 cuTLASS GEMM dominates 67–77% of GPU time at 32k / 64k / 128k.
- Sparse FA itself is only **3.6–4.3%** of GPU time and scales linearly.
- compress_k1/k2 fill (post-CHANGE_0133) is now 1.0–1.1% — under control.

That run was BF16 (no GPTQ) precisely because previous attempts at `GPTQ + sparse + FP8 KV` collapsed accuracy on the public set (Tests 5/6/8). But the profile is telling us something interesting: **the sparse mechanism itself is fast and well-behaved.** What killed the BF16+sparse line on SOAR was the BF16 GEMM cost, not the sparse algorithm.

That suggests an **untested combination** that could plausibly beat Test 12:
> `--quantization gptq_marlin` (Marlin INT4 GEMM everywhere — same as Test 12)
> + `minicpm_flashinfer` backend (sparse routing **enabled** for long requests only)
> + FP8 KV
> + a **high `dense_len` threshold** so only sequences longer than that threshold actually take the sparse path; short sequences run the dense path with full Marlin INT4 speed.

Test 12 forces dense for **all** sequences via `--force-dense-minicpm`. CHANGE_0136 keeps Test 12's behaviour for short prompts but lets the long ones (which spend most of their time in attention I/O) take advantage of sparse top-k.

The historical accuracy collapse of GPTQ+sparse may have been because **all** layers always took the sparse path. With a high threshold, only the few long-context samples route sparse — bounding the failure-mode blast radius.

## 2. Rule-compliance statement

- Server-side runtime knob only. No model file changes (`config.json`, weights, tokenizer untouched).
- No eval-script changes.
- No sgl-kernel rebuild needed.
- Submission packaging compatible (the CLI flag/env var lives in `prepare_env.sh`'s `SGLANG_SERVER_ARGS`, exactly the official-supported customization surface).
- Default behaviour identical to today (no flag set → falls through to `hf_config.sparse_dense_len`).

## 3. Detailed implementation plan (before-change view)

### 3a. Where the threshold is read today

Three sites read `sparse_dense_len`:

- [`python/sglang/srt/layers/attention/minicpm_backend.py:238-239`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L238-L239) (the per-batch dispatch decision, **the actual routing site**):
  ```python
  self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
  self.config_dense_len = hf_config.sparse_dense_len
  ```
- [`python/sglang/srt/layers/attention/minicpm_sparse_utils.py:958`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L958):
  ```python
  dense_len = hf_config.sparse_dense_len
  ```
- [`python/sglang/srt/configs/model_config.py:281-283`](../../python/sglang/srt/configs/model_config.py#L281-L283) — accessor with default 512.
- [`python/sglang/srt/configs/minicpm.py:47,83,95`](../../python/sglang/srt/configs/minicpm.py) — config plumbing (default 512, overridable from `config.json`'s `sparse_config.dense_len`).

### 3b. Per-request decision logic

`MiniCPMAttentionBackend.dense_len` is the single per-request branch threshold: if `seq_len < dense_len` the request runs the dense FlashInfer path inside the sparse layers; otherwise it runs the sparse top-k + paged-KV path. So overriding `dense_len` at server-arg time is sufficient to bias the routing without touching the model.

## 4. Proposed change (after-change view)

**Single env var, read at backend construction time. No CLI parser change required (keeps the patch surface minimal).**

In `python/sglang/srt/layers/attention/minicpm_backend.py` around L238:

```python
import os

self.dense_as_sparse = model_runner.server_args.dense_as_sparse
_env_override = os.environ.get("SOAR_SPARSE_DENSE_LEN")
if _env_override is not None and not self.dense_as_sparse:
    try:
        _override = int(_env_override)
        if _override < 0:
            raise ValueError("must be >= 0")
        self.dense_len = _override
        self.config_dense_len = _override
        logger.info(
            f"SOAR_SPARSE_DENSE_LEN override: dense_len={_override} "
            f"(model config default was {hf_config.sparse_dense_len})"
        )
    except (ValueError, TypeError) as e:
        logger.warning(
            f"SOAR_SPARSE_DENSE_LEN={_env_override!r} invalid ({e}); "
            f"falling back to config value {hf_config.sparse_dense_len}"
        )
        self.dense_len = hf_config.sparse_dense_len
        self.config_dense_len = hf_config.sparse_dense_len
else:
    self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
    self.config_dense_len = hf_config.sparse_dense_len
```

And mirror the same override at [`minicpm_sparse_utils.py:958`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L958) (where `sparse_dense_len` is read for some metadata path), or — preferred — pass the resolved value through from the backend instead of re-reading the config. Will pick the cleaner of the two during patch authoring.

In `benchmark/soar/demo_sala/prepare_env.sh` (no behaviour change unless the env var is set):

```bash
# Optional: override the per-request sparse routing threshold.
# Sequences with seq_len < SOAR_SPARSE_DENSE_LEN run the dense FlashInfer path;
# longer ones take the sparse top-k path. Default = unset → hf_config.sparse_dense_len (512).
# Recommended sweep values: 524288 (sanity = always dense), 65536, 32768, 16384.
if [[ -n "$SOAR_SPARSE_DENSE_LEN" ]]; then
    export SOAR_SPARSE_DENSE_LEN
fi
```

`SGLANG_SERVER_ARGS` itself does **not** change in any submission run — the env var controls everything.

## 5. Validation plan (3-step matrix on Test 12 baseline config)

Server config kept identical to **Test 12** for every step:
GPTQ + FP8_e5m2 KV + `--attention-backend minicpm_flashinfer` (NOT `--force-dense-minicpm`) + chunk=32K, prefill-max-req=1, running=24, sched-cons=1.0, mixed-chunk, torch.compile bs=8.

**Important:** drop `--force-dense-minicpm` for this experiment — that flag short-circuits the dispatch and makes the threshold irrelevant. Routing is entirely controlled by `SOAR_SPARSE_DENSE_LEN`.

| Step | `SOAR_SPARSE_DENSE_LEN` | Expected behaviour                                | Pass criterion |
|------|-------------------------|----------------------------------------------------|----------------|
| **a (sanity)** | `524288` | All requests route dense → must reproduce Test 12 numbers exactly | ori_acc within ±1pt of Test 12; S₁ within ±2% of 121.71s |
| **b (conservative)** | `65536` | Only the longest-tail samples route sparse | norm_acc ≥ 99% (C=1.0); S₁/S₈/S∞ each within ±2% of Test 12 OR better |
| **c (aggressive)** | `16384` | Most long-context samples route sparse | norm_acc ≥ 99% (C=1.0) AND ≥ one of S₁/S₈/S∞ improves > 2% vs Test 12 |

If step (a) deviates from Test 12 → patch is buggy, fix before continuing.
If step (b) accuracy drops < 99% normalized → revert immediately, this path is unsafe (matches the historical GPTQ-sparse collapse). Do not run step (c).
If step (b) passes accuracy but no speed gain on any tier → conclude that long samples in the public set are too short / too few to benefit; document and stop.
If step (c) passes both → adopt as new submission baseline candidate; re-run with `--max-concurrent 32` to verify under official-style load.

## 6. Risk analysis

- **Code surface:** ≤ 30 lines of Python, single function. No kernel rebuild.
- **Default behaviour:** unchanged when env var is unset. Test 12 path completely untouched.
- **Submission compatibility:** `prepare_env.sh` is the official customization surface; env-gated knobs are the cleanest way to add per-instance variations.
- **Accuracy risk:** moderate — the path being exercised (`GPTQ + sparse routing on long requests`) is the same one that collapsed in Tests 5/6/8. **Mitigation: step (b) is gated on normalized accuracy ≥ 99%; if it fails we abort before step (c).**
- **Speed risk:** none vs Test 12 — step (a) reproduces Test 12 exactly; step (b)/(c) only diverge on long requests.
- **CHANGE_0133 dependency:** the sparse path now requires the bounded compress_k fill from CHANGE_0133. That fix is already in HEAD. If for any reason CHANGE_0133 has been reverted before this experiment runs, abort and restore it first.

## 7. Result summary table (placeholder — to be filled after run)

| Step | dense_len | ori_acc | norm_acc | C   | S₁ (s) | S₈ (s) | S∞ (s) | Notes |
|------|-----------|---------|----------|-----|--------|--------|--------|-------|
| a    | 524288    |    —    |    —     |  —  |   —    |   —    |   —    | sanity |
| b    | 65536     |    —    |    —     |  —  |   —    |   —    |   —    | conservative |
| c    | 16384     |    —    |    —     |  —  |   —    |   —    |   —    | aggressive |

Test 12 reference: ori_acc=79.29%, norm=99.11%, C=1.0, S₁=121.71s, S₈=44.09s, S∞=35.86s.

## 8. Validation commands

```bash
# Step (a) sanity
SOAR_SPARSE_DENSE_LEN=524288 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# Step (b)
SOAR_SPARSE_DENSE_LEN=65536  python3 scripts/fcloud/fcloud_workflow.py restart-server
# ... same accuracy + speed commands

# Step (c) — only if (b) passes
SOAR_SPARSE_DENSE_LEN=16384  python3 scripts/fcloud/fcloud_workflow.py restart-server
# ... same accuracy + speed commands
```

(`fcloud_workflow.py restart-server` already sources `prepare_env.sh`, which honours the env var per §4.)

## 9. Rollback

Pure Python edit, single file, env-gated:

```bash
git checkout python/sglang/srt/layers/attention/minicpm_backend.py
git checkout python/sglang/srt/layers/attention/minicpm_sparse_utils.py  # if also edited
git checkout benchmark/soar/demo_sala/prepare_env.sh
```

Or simply unset `SOAR_SPARSE_DENSE_LEN` — default behaviour is unchanged.

## 10. Next-step suggestions

- If CHANGE_0136 step (b) or (c) wins on speed *and* accuracy, the natural follow-up is to re-quantize with sparse-aware GPTQ calibration to push the safe threshold lower (e.g. dense_len=4096) and capture more of the workload on the sparse path.
- If CHANGE_0136 step (b) fails on accuracy, that confirms the historical "GPTQ + sparse routing collapses accuracy at any threshold" finding and we close this line for SOAR submission. The conclusion would be: dense + GPTQ + FP8 KV (Test 12) is the global optimum on the current quantization quality.
- Independent of CHANGE_0136, if `--attention-backend flashinfer` (Round 13f-1 smoketest) is shown to be a working alias of the dense path with no regression, we may simplify `prepare_env.sh` to use the official default.
