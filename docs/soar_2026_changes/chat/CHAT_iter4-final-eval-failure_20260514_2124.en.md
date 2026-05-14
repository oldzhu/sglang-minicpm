# Chat — CHANGE_0165 iter 4 final eval (catastrophic failure)

**Session**: 2026-05-14 ~21:24
**Topic**: Final accuracy + speed eval for CHANGE_0165 iter 4 (preflight diff=0 fix). MEDUSA Stage 3a K=1 passthrough with cuda-graph + torch-compile.

## Round 1 — user "go"

**User**: "go" — authorized the final eval pipeline per CHANGE_0165 §5.6.

**Agent actions**:
1. `fcloud_workflow.py start-instance` → HTTP 200 "任务已启动".
2. Polled `wait-server` (12×10s) → READY.
3. `fcloud_workflow.py full` → sync (0e4634c..757ea5a fast-forward) + restart-server with MEDUSA + cuda-graph + torch-compile + force-dense + dense-as-sparse args + accuracy eval.
4. Server health 200 after ~8 min (cuda-graph capture took ~6 min, 16 buckets bs=[1..24]).
5. Eval ran ~2h55m, completed.
6. Read summary at `/root/data/outputs/20260514_101850/{summary.json,summary.txt}` → **catastrophic**:
   - ori_accuracy=**13.16%** (vs 79.29% baseline) → C=0 (disqualified)
   - normalized=**16.44%** (vs 99.11%)
   - duration=10488.91s (~2h55m, +147% vs baseline 4244s)
   - mcq=0% (0/30), avg_out_tokens=55,816 → runaway
   - niah=6.67%, cwe=8%, fwe=27.78%, qa=23.33%
7. Inspected 3 mcq predictions → repetition garbage: `</think>\n</think>\n...`, `_ _ _ _ ...`, `$\n$\n...`.
8. Server log during decode reported `cuda graph: False` on every batch (despite startup capturing 16 buckets cleanly) and `accept_len=1.03–1.19, accept_rate=0.51–0.60` (kernel *thought* spec was working).
9. Skipped speed tests (clearly slower than baseline; C=0 anyway).
10. `pause-instance` first attempt returned openresty 502 HTML; **retried**, succeeded with HTTP 200 "任务已暂停". fcloud paused.

## Outcomes & artifacts

- `TEST_RESULTS_TRACKING.md` — added row `Stage3a-preflight0-CATASTROPHIC` (commits 0e4634c1d + 757ea5a6d).
- `CHANGE_0165_medusa_preflight_diff.en.md` / `.zh.md` — appended §5.7 "Iteration 4 final eval — CATASTROPHIC FAILURE" with full numbers, 3 sample mcq strings, 4-hypothesis root-cause list (H1–H4), and 4 next-step options.
- fcloud artifacts (persist across pause):
  - `/root/data/outputs/20260514_101850/{predictions.jsonl,summary.json,summary.txt}`
  - `/tmp/dump_ngram.pkl` (52937B), `/tmp/dump_medusa.pkl` (52938B), `/tmp/iter4_diff.txt` (TOTAL=0) — preflight evidence

## Key finding

**Preflight 0-diff at 4 boundaries × 25 fields is NECESSARY but NOT SUFFICIENT for spec-decoding correctness.** The preflight only captured the FIRST verify iteration. Multi-iteration divergence (GLA state drift, eager-mode decode path differences, in-kernel logits indexing) is invisible to single-iteration boundary snapshots.

## Open questions for user

1. **Revert** commit `0e4634c1d` (the iter-4 fix) and re-eval to localize the regression source?
2. **Instrument further** — multi-iteration preflight (iter 1, 2, 5, 10, 25) and/or in-kernel verify tracer?
3. **Abandon Stage 3a Medusa** and ship v23 = **Stage 2-cgraph** (`46553947b`, S1=118.28s, 80.11%, C=1.0) as final?

Recommended: try (1) cheaply; if still broken, lock (3) for submission.

## Cross-references

- CHANGE_0165 §5.5 (preflight 0-diff achieved) / §5.6 (proposed final eval plan) / §5.7 (this failure).
- Best-known-good Medusa baseline: Stage 2-cgraph (TEST_RESULTS_TRACKING row, commit `46553947b`).
- Best-known-good overall: Test 12 / v18-revert (`8d1e4d12b`, S1=110.51s, 79.29%, C=1.0).
