# PROPOSAL — Round 13f-2: Recover Test 12 accuracy on the explicit `flashinfer` backend path

## Status: PROPOSAL (awaiting approval)

## Background

Round 13f-1 (`SOAR_BACKEND_VARIANT=flashinfer`, commit `6a070110b`) measured:

| Metric | Test 12 baseline | Round 13f-1 | Δ |
|---|---|---|---|
| ori_accuracy | 79.29% | **76.91%** | −2.38pt |
| Normalized | 99.11% (C=1.0) | ~96.1% | C=0 (eliminated) |
| S₁ | 121.71s | **110.76s** | −9.0% |
| S₈ | 44.09s | **40.50s** | −8.1% |
| S∞ | 35.86s | **33.66s** | −6.1% |

Speed gain is real and meaningful. Accuracy lands below the C=0 cutoff, so the
config is unusable as-is.

The RESEARCH doc
[`RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md`](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md)
established that:

- Test 12 (`--attention-backend minicpm_flashinfer --force-dense-minicpm`)
  internally rewrites to `attention_backend=flashinfer` at
  `server_args.py:1525`, so **both Test 12 and Round 13f-1 run the same stock
  FlashInfer std-attn kernel**.
- Lightning-mixer `recurrent_threshold` is the same (`prepare_env.sh:129`
  exports `=128` unconditionally).
- The remaining structural difference is `model_config.force_dense_minicpm`:
  - True (Test 12): `has_sparse_attention=False`, `sparse_layer_ids=[]`.
  - False (Round 13f-1): `has_sparse_attention=True`, populated layer ids.

`has_sparse_attention` is consumed in 6 hot places that affect KV cache pool
type, scheduler chunking, and per-request slot allocation
(`schedule_batch.py:1473,1525,1983,2036`,
`model_runner_kv_cache_mixin.py:369,408`,
`minicpm_backend.py:222`). Stock FlashInfer never reads compress_k1/k2 cache,
but that flag changes how the scheduler / KV pool is set up, which in turn
changes batch composition and numerics.

## Hypothesis

The 2.4-pt accuracy regression is caused by `has_sparse_attention=True`
side-effects on the scheduler / KV pool, **not** by the std-attn kernel and
**not** by the lightning mixer. Restoring `force_dense_minicpm=True` while
keeping the literal backend string `flashinfer` should:

- Reproduce Test 12 acc within local noise (±2pt).
- Either preserve Round 13f-1's speed (then we have a new viable baseline
  with the `flashinfer` string explicit) or regress to Test 12 speed (then
  we know the speed gain came from the sparse scheduler path, and we have
  to characterize whether that path can be made accuracy-stable).

## Plan

One-iteration test: **Round 13f-2** = Round 13f-1 with `--force-dense-minicpm`
re-added.

### Approach 1: minimal `prepare_env.sh` patch (preferred)

Add a sub-mode to `SOAR_BACKEND_VARIANT=flashinfer` that does NOT clear
`FORCE_DENSE_ARG`. Use `SOAR_BACKEND_KEEP_FORCE_DENSE=1` as the gate so the
existing default behavior is unchanged.

```bash
# diff against benchmark/soar/demo_sala/prepare_env.sh, gptq branch
	if [[ "$SOAR_BACKEND_VARIANT" == "flashinfer" ]]; then
		BACKEND_ARG=" --attention-backend flashinfer"
-		FORCE_DENSE_ARG=""
+		# Round 13f-2: by default keep dropping --force-dense-minicpm
+		# (Round 13f-1 behaviour). Set SOAR_BACKEND_KEEP_FORCE_DENSE=1
+		# to keep --force-dense-minicpm so model_config exposes
+		# has_sparse_attention=False / sparse_layer_ids=[]; isolates
+		# whether the 2.4pt acc drop in 13f-1 is owned by those flags.
+		if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
+			# leave FORCE_DENSE_ARG as previously set above (i.e.,
+			# " --force-dense-minicpm" from the SOAR_SPARSE_MODE!=1 branch)
+			:
+		else
+			FORCE_DENSE_ARG=""
+		fi
		DENSE_AS_SPARSE_ARG=""
-		echo "[prepare_env] SOAR_BACKEND_VARIANT=flashinfer -> using stock flashinfer backend, dropping --force-dense-minicpm and --dense-as-sparse"
+		echo "[prepare_env] SOAR_BACKEND_VARIANT=flashinfer KEEP_FORCE_DENSE=${SOAR_BACKEND_KEEP_FORCE_DENSE:-0} -> using stock flashinfer backend, FORCE_DENSE_ARG='${FORCE_DENSE_ARG}', dropping --dense-as-sparse"
	else
```

No source-code change needed. Commit message: `prep_env: 13f-2 add SOAR_BACKEND_KEEP_FORCE_DENSE switch`.

### Risk assessment

- **Boot risk**: very low. The combination `--attention-backend flashinfer
  --force-dense-minicpm` is **internally** what Test 12 already runs (via the
  rewrite at `server_args.py:1525`). The only difference is that we now
  arrive at that state via the explicit `flashinfer` string instead of
  through the rewrite. Any code that reads `attention_backend` will see
  `"flashinfer"` (same as Test 12 post-rewrite). No new code paths reached.
- **Accuracy risk**: low. Worst case acc ≈ Test 12 (79.29%) → C=1.0. There
  is no regression vector relative to Test 12 because we are not introducing
  any new component.
- **Speed risk**: unknown. If the speed gain in 13f-1 came from removing
  `force_dense_minicpm`'s scheduler side-effects, 13f-2 will regress to
  Test 12 speed. That's still a successful experiment (it tells us the
  `flashinfer` string itself is irrelevant; the speed gain is owned by the
  sparse scheduler path, which is then accuracy-fatal — meaning Round 13f
  line is dead).

### Rule compliance

- No model-quantization changes; same GPTQ + FP8_e5m2 KV.
- No eval script edits.
- No architectural changes; only a server-flag combination not previously tested.
- Within submission constraints (≤2GB, ≤5h, no prefix cache).

## Test commands (after user approval)

```bash
# 1. Sync prepared env change to fcloud
python3 scripts/fcloud/fcloud_workflow.py sync

# 2. Restart server with the new env:
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --env SOAR_BACKEND_VARIANT=flashinfer \
  --env SOAR_BACKEND_KEEP_FORCE_DENSE=1
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 3. Verify the cmdline:
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server"
# MUST contain: "--attention-backend flashinfer --force-dense-minicpm"
# MUST NOT contain: "--dense-as-sparse" "minicpm_flashinfer"

# 4. Accuracy first (gating).
python3 scripts/fcloud/fcloud_workflow.py accuracy
# Pass criterion: ori_accuracy ≥ 78.5% (Test 12 ±0.8pt local noise floor)

# 5. If acc passes, run S1/S8/Smax (one at a time per known --variant all bug):
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# 6. Shutdown.
python3 scripts/fcloud/fcloud_workflow.py shutdown
```

## Decision matrix after results

| Outcome | acc | S₁ | Decision |
|---|---|---|---|
| **A** | ≥78.5% | ≤115s | **WIN — promote to v20 candidate.** Round 13f flashinfer line is the new submission baseline. Update prepare_env.sh default + record OPT catalog. |
| **B** | ≥78.5% | ≥118s (Test 12 level) | **NEUTRAL.** Speed gain owed entirely to dropping force-dense; explicit-flashinfer string brings nothing extra. Park Round 13f, return to dense+GPTQ optimization. |
| **C** | <78.5% | any | **FAIL.** `has_sparse_attention=True/False` flip is NOT the dominant cause of the 2.4pt regression. Run Exp D (compile re-enabled) next, or consider: was 13f-1's 76.91% just one bad sample of local noise? Recommend a re-run of plain 13f-1 first to confirm 76.91% is reproducible. |

## Rollback

The change is opt-in (`SOAR_BACKEND_KEEP_FORCE_DENSE=1`). To revert: unset
the env var. To remove from history: revert the prepare_env.sh diff.

## Cross-references

- RESEARCH: [RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.en.md)
- 13f-1 chat: [chat/CHAT_round13f_change0136_validation_20260429_1410.en.md](chat/CHAT_round13f_change0136_validation_20260429_1410.en.md)
- 13f-1 row: TEST_RESULTS_TRACKING.md `R13f1-flashinfer`.
- Parked 13f sibling work: [CHANGE_0136_minicpm_sparse_dense_len_flag.en.md](CHANGE_0136_minicpm_sparse_dense_len_flag.en.md), [CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md](CHANGE_0137_sparse_prefill_page_table_off_by_one.en.md).
- Code anchors:
  - [python/sglang/srt/server_args.py](../../python/sglang/srt/server_args.py#L1521-L1525)
  - [python/sglang/srt/configs/model_config.py](../../python/sglang/srt/configs/model_config.py#L236-L248)
  - [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py#L1473)
