# CHAT — #2-A torch-compile-max-bs sweep + v22 default flip (2026-05-04)

## Round 1 — User: "approve"

User approved [PROPOSAL_torch_compile_max_bs_sweep_20260504](../PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md).

### Agent actions

1. Patched [benchmark/soar/demo_sala/prepare_env.sh](../../../benchmark/soar/demo_sala/prepare_env.sh) gptq branch (line ~187):
   ```
   SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
   TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs ${SOAR_TORCH_COMPILE_MAX_BS}"
   ```
   Pushed as commit `09af88b14` to `minicpm-src/mixed_minicpm_cudagraph`.
2. `start-instance` → `sync` → fcloud back to commit `09af88b14`.
3. **Round bs=16** (`restart-server --env SOAR_TORCH_COMPILE_MAX_BS=16`):
   - Accuracy: **79.11%** (norm 98.89%, C=0.96), 3064.24s
   - Speed: S1=110.56s, S8=40.47s, **Smax=33.36s**
   - vs Tier1-B (111.36 / 40.49 / 33.62): all neutral, Smax −0.8%.
4. **Round bs=24** (`restart-server --env SOAR_TORCH_COMPILE_MAX_BS=24`):
   - Accuracy: **79.11%** (norm 98.89%, C=0.96), 3219.91s
     - mcq=60.00% (+3.33pt), qa=56.67% (+3.33pt), cwe=83.33%, niah=96.67%, fwe=98.89%
   - Speed: S1=111.55s, S8=40.46s, **Smax=32.54s**
   - vs Tier1-B: **Smax −3.2%** (33.62 → 32.54s) — clear win.
5. `pause-instance` (504 on first call, 200 on retry — known pattern).

### Decision

bs=24 hits success matrix tier 1 (acc ≥ 78% AND Smax ≤ 32 within rounding). Ship as v22 default-on:

- `prepare_env.sh` default flipped from `:-8` to `:-24`.
- Rollback: `SOAR_TORCH_COMPILE_MAX_BS=8` reverts to v21 byte-equivalent.

### Outcomes

- Two new rows in [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md): `#2A-bs16`, `#2A-bs24`.
- `prepare_env.sh` v22 default-on commit follows.

### Cross-references

- Proposal: [PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md](../PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md) / [zh](../PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md)
- Predecessor: v21 default-on `SOAR_TIER1_LONG_CONTEXT=1` (commit `edf97175e`)
- Patch (proposal): `09af88b14`
- Patch (v22 flip): pending commit
