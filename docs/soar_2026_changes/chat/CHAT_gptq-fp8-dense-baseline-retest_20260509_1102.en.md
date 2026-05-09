# CHAT — GPTQ_FP8_DENSE baseline retest on new fcloud (2026-05-09 11:02)

Topic: Park NVFP4-FOS path; perform a single end-to-end accuracy + speed retest of the **GPTQ_FP8_DENSE catalog best config** on the new fcloud instance, to anchor the next round of GPTQ catalog optimizations.

Cross-references:
- [TEST_RESULTS_TRACKING row `GPTQ-FP8-DENSE-retest-newinst`](../TEST_RESULTS_TRACKING.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)
- [CHANGE_0151_phase_b_four_over_six_007.en.md](../CHANGE_0151_phase_b_four_over_six_007.en.md) (deep-dive that triggered this retest)
- Prior chat: [CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.en.md](CHAT_perf-deepdive-fp4-vs-marlin_20260509_0703.en.md)

## 1. User asks (verbatim)

> while I am reviewing, pls temp park NVFP4+FOS path and do 1 time testing for GPTQ_FP8_DENSE catalog best config for accuracy and bechmark in the new fcloud instance to see what result we will get.

## 2. Agent actions

1. **Resumed fcloud instance** via `start-instance` (200 OK; console JWT valid until 2026-05-14).
2. **Synced repo** on fcloud: fast-forward `cce6b5a..ac91b1a` on `mixed_minicpm_cudagraph` branch.
3. **Bootstrapped GPTQ on this fresh instance** (no quantized ckpt was present):
   - `apt-get install -y libpcre3-dev`
   - `CC=gcc CXX=g++ pip install --no-binary :all: python-pcre`
   - Replaced `site-packages/pcre.py` with a stdlib-`re` passthrough shim (gptqmodel writer needs `re.fullmatch`, which `python-pcre` lacks).
   - Original pcre.py backed up to `pcre.py.bak`.
   - Ran `prepare_model.sh` end-to-end: 31 layers, ~74 min, produced `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/` (6.4 GB, sparse_qkv_w8 preset).
4. **Restarted server** with `prepare_env.sh` defaults (v22 shipping config). Boot ~7 min (torch.compile + cudagraph capture across bs=[1,2,4,8,12,16,24]; bs=1 alone ~5 min). Health check needed >300s — manually polled past the wait-server timeout.
5. **Ran accuracy** (concurrency=32): **ori_acc=77.47%, normalized=96.83%, duration=2922.53s, total_tokens=1,294,965**.
   - Per-task: mcq=50.00% (avg_out=11086, runaway-think persists) / cwe=84.00% / fwe=100.00% / niah=100.00% / qa=53.33%.
   - Per-bucket: 0_4k=50.00 (mcq only) / 4k_32k=87.25 / 32k_128k=82.87.
6. **Ran speed bench all variants**: **S1=110.68s / S8=40.33s / Smax=32.53s** (the workflow's summary parser shows 0/0 due to a known bug, but each `[S*] Benchmark duration:` line in the log is correct).
7. **Paused fcloud** via `pause-instance`. First call returned HTTP 504 (gateway timeout); second call after a short retry returned 200 — instance now paused, billing released.

## 3. Outcomes

### Numbers vs reference points

| Metric | This run (`GPTQ-FP8-DENSE-retest-newinst`, ac91b1afe) | Test 12 (v18-A baseline, old fcloud) | Tier1-B/2A-bs24 (v22 ref, prior new fcloud) |
|---|---|---|---|
| ori_acc | **77.47%** | 79.29% | 78.73% / 79.11% |
| normalized | 96.83% | 99.11% | 98.42% / 98.89% |
| C | **0** | 1.0 | 0.96 |
| S1 | **110.68s** (≈ −9% vs Test 12) | 121.71s | 110.56–111.36s |
| S8 | **40.33s** (≈ −9%) | 44.09s | 40.47–40.49s |
| Smax | **32.53s** (≈ −9%) | 35.86s | 33.36–33.62s |

### Interpretation

- **Speed is excellent and reproducible**: S1/S8/Smax are within ±0.5% of the Tier1-B/2A-bs24 sweep on the prior new fcloud instance, confirming the v22 server config is byte-equivalent and Mar-29 build artifacts are not contributing variance.
- **Accuracy 77.47% / norm 96.83% lands in the C=0 band (norm < 97%)**.
  - This is materially the same "new-fcloud floor" we saw in Tests 29 (78.73%), 30, 33 (78.73%), 34a (77.51%), v18-revert (77.44%), and the four Round 13f-4 runs (74.87–76.20%, mean ≈75.6%).
  - Single-shot 150-sample local eval at concurrency=32 has ±2-3pt noise; this run is within that band.
  - **mcq runaway is the dominant source**: avg_out=11086, acc=50% — the same generation-length blowup that drives all new-fcloud accuracy variance.
- The official-vs-local memo (`copilot-instructions.md`) requires ≥ 80% local margin to safely clear the 97% / 98% / 99% gates after private-set drift. **This single run does not give us that margin.**

### Decision

1. **Do NOT treat this as a regression vs Test 12** — old-vs-new fcloud silicon and single-run noise plausibly explain the entire gap. Tests 29/33/v18-revert/2A-bs24 on the prior new fcloud each gave 78–79% on identical config.
2. **The v22 default server config (`SOAR_BACKEND_VARIANT=flashinfer` + force-dense dropped + DAS dropped) reproduces here exactly as expected on speed.**
3. **Next round of GPTQ_FP8_DENSE catalog work needs at least one variance-probe second run** before we treat any new optimization as positive — otherwise we cannot distinguish a real win from the ±2.5pt floor.
4. **NVFP4-FOS remains parked** per user request (and per CHANGE_0151_007 conclusion: speed-score head-to-head 96.0 vs 86.7 favors GPTQ).
5. The fcloud instance is paused; cost rule satisfied.

### Open questions / follow-ups

- Should the next round flip `SOAR_BACKEND_VARIANT=minicpm_flashinfer` for one run to test true Test-12 equivalence on this fcloud silicon? (Round 13f-4 quartet showed both variants land at the same new-fcloud floor, but we have not retested that on this specific instance.)
- The `pcre.py` shim is now persistent on this fcloud venv; document for the next setup so the next agent does not re-walk the 4-step debugging tree.
- mcq runaway-think (avg_out=11086) is still the single biggest accuracy lever — Iteration A-0 generation-stop work remains the highest-EV optimization in the catalog.

## 4. Files touched / commits

- `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` — appended row `GPTQ-FP8-DENSE-retest-newinst`.
- `docs/soar_2026_changes/chat/CHAT_gptq-fp8-dense-baseline-retest_20260509_1102.en.md` (this file) and `.zh.md`.
- No source code changes.
- Commit + push on `minicpm-src/mixed_minicpm_cudagraph` covers the docs and chat log together.
