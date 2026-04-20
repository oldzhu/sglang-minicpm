# SOAR 2026 Collaboration Instructions (MiniCPM-SALA)

This repository is used for SOAR 2026 optimization work on MiniCPM-SALA.

## Competition GPU Hardware Reference (must consult)

- Full hardware specs, CUDA programming notes, and optimization opportunities for the competition GPU:
  **`docs/soar_2026_changes/SM120_RTX_PRO_HARDWARE.md`**
- Key facts: SM120 (Blackwell), 96 SMs, 148/296/593 TFLOPS BF16/FP8/FP4, 84GB GDDR7, 1398 GB/s, 112MB L2
- MMA is **warp-level** (not warpgroup); TMA and QMMA (mxfp8) are supported
- Before proposing any kernel or GEMM optimization, **consult this file** to verify SM120 compatibility and estimate realistic throughput gain

## Git push rule (CRITICAL — must enforce)

- **ALWAYS push to `minicpm-src` remote**, NEVER to `origin`.
- fcloud pulls from `oldzhu/sglang-minicpm` (`minicpm-src`), NOT from `oldzhu/sglang` (`origin`).
- Correct: `git push minicpm-src mixed_minicpm_cudagraph`
- Wrong: `git push origin mixed_minicpm_cudagraph` ← fcloud won't see changes

## Mandatory workflow for every optimization

1. **Proposal first, no direct code changes**
   - Before any source change, provide a detailed optimization proposal including:
     - objective and expected gain
     - rule-compliance check (SOAR constraints)
     - risk to accuracy/stability
     - exact files/functions to change
     - test and benchmark commands
   - Wait for explicit user approval before editing code.

2. **One improving feature at a time**
   - Each iteration should deliver one complete optimization feature (can include related updates across multiple files).
   - Keep scope cohesive: all edits in the iteration must serve the same optimization objective.
   - Avoid mixing unrelated goals in one iteration.

3. **Bilingual documentation per feature (required)**
   - For each approved change, create two documents:
     - English: `docs/soar_2026_changes/CHANGE_XXXX_<short_title>.en.md`
     - Chinese: `docs/soar_2026_changes/CHANGE_XXXX_<short_title>.zh.md`
   - One document pair corresponds to exactly one optimization feature iteration.
    - If appending new content to an existing feature doc would be long, do not over-append in place. Create a new continuation document pair with the same base filename and an incremented numeric suffix before locale, for example:
       - `docs/soar_2026_changes/CHANGE_0030_<short_title>_001.en.md`
       - `docs/soar_2026_changes/CHANGE_0030_<short_title>_001.zh.md`
    - Keep continuation EN/ZH docs synchronized with the same suffix number (`_001`, `_002`, ...).

4. **Documentation must include**
   - Background and motivation
   - Rule-compliance statement (what is allowed and why)
   - Detailed implementation plan (before change)
   - Actual code changes (after change)
   - Validation commands (correctness + speed)
   - Result summary table (baseline vs new)
   - Rollback instructions
   - Next-step suggestions

5. **Execution model with user’s fcloud instance**
   - Agent proposes and documents changes in this workspace.
   - User applies/runs commands in fcloud instance and reports metrics/errors.
   - Agent iterates based on returned results.

## Official Scoring & Ranking Rules (from https://soar.openbmb.cn/competition, verified 2026-04-12)

### Final Score (HIGHER = BETTER)
```
Final Score = Performance Score × Correctness Coefficient C
```

### Performance Score (relative to best player)
```
Performance Score = S₁ × 40% + S₈ × 30% + S∞ × 30%
S_N = (Duration_best / Duration_player) × 100
```
- `Duration_best` = shortest benchmark_duration among ALL players for that concurrency tier
- Fastest player scores 100 per tier; others score proportionally less

### Correctness Coefficient C (4 tiers)
| Normalized Accuracy | C |
|---------------------|-----|
| ≤ 97% | 0 (eliminated) |
| (97%, 98%] | 0.92 |
| (98%, 99%] | 0.96 |
| (99%, 100%] | 1.0 |

### Concurrency Tiers
| Tier | Flag | Weight |
|------|------|--------|
| S₁ | `--max-concurrent 1` | 40% |
| S₈ | `--max-concurrent 8` | 30% |
| S∞ | no `--max-concurrent` | 30% |

### Submission Constraints
- All files ≤ 2GB total
- Quantized models must be quantized on-site (cannot submit pre-quantized weights)
- Quantization + evaluation time ≤ 5 hours
- Speculative heads allowed (count toward 2GB)
- Code: Apache 2.0, reproducible, explainable

## Competition guardrails (must enforce)

- Keep normalized accuracy > 97% so C ≠ 0 (ideally > 99% for C=1.0).
- Do not rely on forbidden tricks (e.g., privately re-enabling prefix cache during official eval).
- Respect fixed concurrency evaluation settings (`--flush-cache`, fixed `--max-concurrent`).
- Keep submission package constraints in mind (≤ 2GB, on-site quantization, ≤ 5h total).

## Baseline config (must enforce)

- The **current best config** is: **GPTQ (sparse_qkv_w8) + FP8 KV cache + dense mode** (`--force-dense-minicpm --kv-cache-dtype fp8_e5m2`).
- All optimization work and accuracy/speed testing **MUST** use this config unless:
  1. We have exhausted all improvement avenues on this config, AND
  2. We explicitly decide to try an alternative config with documented rationale.
- **Never** test optimizations on GPTQ + sparse mode — it gives ~50% accuracy on the old fcloud instance and is not the submission config.
- The baseline reference results (Test 12): S1=121.71s, S8=44.09s, Smax=35.86s, ori_accuracy=79.29%, normalized=99.11%, C=1.0.

## Server launch rule (must enforce)

- **All sglang server args must be defined in `prepare_env.sh` via `SGLANG_SERVER_ARGS`**, not hardcoded in launch commands.
- To tune or add server args, **modify `benchmark/soar/demo_sala/prepare_env.sh`** and then start sglang using:
  ```bash
  source ./prepare_env.sh
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    "${SGLANG_SERVER_ARGS[@]}"
  ```
- This is required because official evaluation launches sglang using the exported `SGLANG_SERVER_ARGS` from `prepare_env.sh`.
- **Never** pass server args directly in the launch command bypassing `SGLANG_SERVER_ARGS`.
- The `fcloud_workflow.py restart-server` command already follows this pattern.

## Rule freshness requirement (must enforce)

- Whenever optimization/compliance decisions depend on competition rules, re-check the latest official pages first:
   - https://soar.openbmb.cn/competition
   - https://soar.openbmb.cn/toolkit
- Before starting optimization/customization stages, explicitly review and refer to the `技术路径指引` section on the toolkit page to align with officially suggested technical directions.
- If any conflict appears between prior assumptions and latest official text, follow the official pages and explicitly call out the update.

## Leaderboard tracking requirement (must enforce)

- Our team name is **team-beta** (currently #19, score 56.63). Target: **top 5** (currently ≥79.55).
- After every official submission that produces a new score, **immediately**:
   1. Fetch leaderboard from https://soar.openbmb.cn/leaderboard
   2. Record team-beta's updated rank and score
   3. Record top 5 teams' scores
   4. Calculate remaining gap to #5 and improvement ratio needed
   5. Assess whether any top 5 teams improved (moving target)
   6. Update `/memories/soar_2026_leaderboard.md` with the new snapshot
- Use this gap analysis to prioritize next optimization direction:
   - If gap > 30%: need fundamental speed improvement (kernel optimization, speculative decoding, architecture changes)
   - If gap 10-30%: targeted optimizations (operator fusion, scheduling tuning, memory layout)
   - If gap < 10%: fine-tuning (server arg tweaks, batch sizing, minor kernel improvements)

## Submission preparation requirement (must enforce)

- When the task involves preparing competition submission artifacts (e.g., `prepare_env.sh`, `prepare_model.sh`, `preprocess_model.py`, packaging layout), explicitly refer to and follow the latest `提交说明` section on:
   - https://soar.openbmb.cn/toolkit
- For submission-related customization, align scripts with official execution model and interfaces (including `prepare_env.sh` and `prepare_model.sh --input/--output` contract), and state any assumptions if local/fcloud environment differs from official runtime.

### Official submission packaging steps (fcloud)
1. On fcloud: `cd /root/submission_sim`
2. Create tarball:
   ```bash
   tar --exclude='__pycache__' --exclude='*.pyc' -czf /root/minicpm_sala_submit_v<VERSION>.tar.gz *.whl *.sh *.py perf_public_set.jsonl sglang
   ```
3. Download the `.tar.gz` from fcloud to local `benchmark/soar/demo_sala/`
4. Upload to official site manually

## Test results tracking (mandatory)

- **All automated test results** (accuracy and speed benchmarks) must be recorded in:
  `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`
- After every accuracy or speed test completes, update the corresponding table in that file with the test number, date, commit, config, and all result metrics.
- This file is the single source of truth for comparing configurations across test runs.

## fcloud automated testing

The workspace includes automation scripts for remote testing on the fcloud instance:

- **Scripts location**: `scripts/fcloud/fcloud_exec.py` (JupyterLab terminal API client), `scripts/fcloud/fcloud_workflow.py` (test workflow orchestrator)
- **Config**: `~/.fcloud_config` stores `FCLOUD_URL` and `FCLOUD_TOKEN`
- **Available commands**:
  - `python3 scripts/fcloud/fcloud_workflow.py setup` — bootstrap a clean fcloud instance (clone, upload, extract, sync)
  - `python3 scripts/fcloud/fcloud_workflow.py sync` — git pull + copy changed files to fcloud
  - `python3 scripts/fcloud/fcloud_workflow.py restart-server` — kill old server and start new one
  - `python3 scripts/fcloud/fcloud_workflow.py wait-server` — wait until server health check passes
  - `python3 scripts/fcloud/fcloud_workflow.py accuracy` — run accuracy eval
  - `python3 scripts/fcloud/fcloud_workflow.py speed --variant s1|s8|smax|all` — run speed benchmarks
  - `python3 scripts/fcloud/fcloud_workflow.py full` — sync + restart + accuracy (full pipeline)
  - `python3 scripts/fcloud/fcloud_workflow.py server-logs --lines N` — view server logs
  - `python3 scripts/fcloud/fcloud_workflow.py shutdown` — shut down the fcloud instance to save cost
- **fcloud paths**:
  - Repo: `/root/sglang-minicpm`
  - Models: `/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` (GPTQ), `/root/models/openbmb/MiniCPM-SALA-Copy` (non-quantized)
  - Eval script: `/root/data/eval_model_001.py` (uses `--data_path /root/data/perf_public_set.jsonl`)
  - Speed data: `/root/data/speed_{s1,s8,smax}.jsonl`
  - Submission sim: `/root/submission_sim`
- **Pre-launch requirement**: Always run `source /root/submission_sim/prepare_env.sh` before starting sglang server to set `PYTORCH_CUDA_ALLOC_CONF` (avoids CUDA OOM)

**IMPORTANT**: Always ask the user for explicit approval before starting any fcloud automated test (sync, restart, accuracy, speed, or full). The fcloud instance is a shared resource — never run tests without user confirmation.

**COST-SAVING RULE (mandatory)**:
- After each round of automated fcloud testing completes and you have collected all outputs needed for analysis, **immediately shut down the fcloud instance** by running `python3 scripts/fcloud/fcloud_workflow.py shutdown` in the terminal. Do not leave it running while analyzing results or planning next steps.
- When you need to start a new round of testing, **ask the user to start the fcloud instance** before running any fcloud commands. Do not assume it is already running.
- Workflow: user starts fcloud → agent runs tests → agent collects output → agent runs shutdown command → agent analyzes results offline → agent proposes next steps → repeat.

## fcloud instance setup / re-setup (mandatory)

When a new or restored fcloud instance needs bootstrapping, use the automated setup command:

```bash
python3 scripts/fcloud/fcloud_workflow.py setup          # skip existing paths
python3 scripts/fcloud/fcloud_workflow.py setup --force   # re-setup everything
```

**Pre-requisites for setup** (must exist locally before running):
- `benchmark/soar/demo_sala/submission_sim.tar` — submission runtime template (~731 MB)
- `benchmark/soar/demo_sala/data.tar.gz` — speed benchmark + eval data (~8 MB)
- `~/.fcloud_config` — must have correct `FCLOUD_URL` and `FCLOUD_TOKEN` for the target instance

**What setup does** (7 steps):
1. Check if `/root/sglang-minicpm`, `/root/submission_sim`, `/root/data` exist — skip if yes
2. `git clone https://github.com/oldzhu/sglang-minicpm.git /root/sglang-minicpm`
3. Upload `submission_sim.tar` to instance, extract to `/root/submission_sim`
4. Copy all files under `/root/sglang-minicpm/python/` to `/root/submission_sim/sglang/python/`
5. Sync `gptqmodel_minicpm_sala.py`, `preprocess_model.py`, `prepare_env.sh`, `prepare_model.sh`, `perf_public_set.jsonl` from `/root/sglang-minicpm/benchmark/soar/demo_sala/` to `/root/submission_sim/`
6. Upload `data.tar.gz` to instance, extract to `/root/data`
7. Sync `eval_model_001.py`, `eval_model.py` from `/root/sglang-minicpm/benchmark/soar/demo_sala/` to `/root/data/`

**When all paths already exist**, setup runs an incremental sync (git pull + copy python/ + sync demo_sala files) without re-uploading tarballs.

**Switching fcloud instances**: Update `~/.fcloud_config` with the new instance's URL and token before running setup or any other fcloud command.

**IMPORTANT**: Always ask the user for explicit approval before running setup on any fcloud instance.

## sgl-kernel build & test on fcloud (must enforce)

When CUDA kernel code in `sgl-kernel/` is modified (e.g., Marlin GEMM changes), rebuild and test on fcloud using:

1. **Sync changes to fcloud repo**:
   ```bash
   cd /root/sglang-minicpm && git pull
   ```

2. **First-time build setup** (once per fcloud instance):
   ```bash
   cd /root/sglang-minicpm/sgl-kernel
   export CXX=g++ CC=gcc                    # Required — fcloud has no default CXX
   apt install -y ccache                     # Enable compilation cache
   export CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=10G
   python -m pip install -U uv scikit-build-core ninja
   make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
   ```

3. **Incremental rebuild** (after editing .cu/.cc files):
   ```bash
   cd /root/sglang-minicpm/sgl-kernel
   export CXX=g++ CC=gcc
   export CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=10G
   # DO NOT run `rm -rf build` — this destroys incremental build cache!
   make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
   ```
   Ninja detects changed files and only recompiles those + relinks.

4. **Copy the built wheel to submission_sim**:
   ```bash
   cp /root/sglang-minicpm/sgl-kernel/dist/sgl_kernel-*.whl /root/submission_sim/
   ```

5. **Install and test**:
   ```bash
   cd /root/submission_sim
   source prepare_env.sh   # This installs the local sgl-kernel wheel
   ```

**IMPORTANT**: There is NO `sgl-kernel` source directory under `/root/submission_sim`. The submission directory installs sgl-kernel from a pre-built `.whl` file only. Do NOT attempt `pip install -e sgl-kernel/` inside `/root/submission_sim`.

**Build time notes**:
- **Full build** (first time): ~4 hours with `MAX_JOBS=2` on fcloud (413 object files, 550MB wheel)
- **Incremental build** (1-file change, `build/` preserved): ~1-3 minutes
- **ccache warm rebuild** (even after `rm -rf build`): ~5-8 minutes
- The `CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"` limits per-file parallelism to avoid OOM
- **NEVER run `rm -rf build`** unless you need a completely clean rebuild (e.g., CMake config changes). Use `make rebuild` only for that case.

## Optimization catalog (must reference)

- The complete top-to-bottom optimization catalog for the baseline GPTQ + FP8 KV + dense config is maintained at:
  `docs/soar_2026_changes/OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`
- This catalog lists every known speed optimization vector (5 layers: scheduling → model → attention → kernels → GEMM), with priority ranking, expected gains, effort, and risk.
- Before starting any new optimization, check this catalog to avoid duplicate work and follow the priority order.
- After testing any optimization, update the catalog with actual results.

## Prioritization strategy

1. Low-risk, high-impact runtime optimizations first.
2. Then quantization/preprocessing path that preserves correctness.
3. Finally higher-risk algorithmic changes (e.g., speculative decoding variants).

## Communication style for this project

- Always provide:
  - what to change,
  - why it should help,
  - how to verify,
  - and what success/failure looks like.
- Ask for approval before each code modification.
- Keep English + Chinese docs synchronized.

## Instruction priority order

When multiple instructions exist, follow this priority (high -> low):

1. System/developer policy constraints from the runtime.
2. This file (`.github/copilot-instructions.md`) for repository-specific rules.
3. User's current request in the active conversation.
4. Other repository docs (`README`, `AGENTS.md`, scripts, comments) as supporting context.

If conflicts happen, follow the higher-priority source and explain the conflict briefly.
