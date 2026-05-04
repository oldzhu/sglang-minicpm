# Chat — Tier 1 long-context retest (2026-05-04 ~08:52)

## Topic

Speed-pivot kickoff: rank optimization-catalog top-3 speed candidates for new
long-context dataset, then run #1 (Tier 1 catalog config retest) on fcloud,
gated behind a one-line env flag for safe rollback.

## Round 1 — Console-config refresh tooling (prereq for fcloud round)

### User ask

> "(a) and by the way, before go (a) have you documented the steps to re-Set
> up console config when cookie expired after 1 week, if u haven't, pls create
> a new document for it. it is possible to automize the steps to resetup the
> console config — I mean f12->network->capture post/put request and get
> authorization and cookie and user name and job id automatically"

### Agent action

1. Created bilingual runbook for the weekly JWT refresh:
   - [docs/soar_2026_changes/HOWTO_refresh_fcloud_console_config.en.md](../HOWTO_refresh_fcloud_console_config.en.md)
   - [docs/soar_2026_changes/HOWTO_refresh_fcloud_console_config.zh.md](../HOWTO_refresh_fcloud_console_config.zh.md)
   - Sections: quick check, manual refresh, HAR semi-automated refresh, why
     full automation is infeasible (3 architectural blockers), TL;DR,
     troubleshooting.
2. Implemented [scripts/fcloud/refresh_console_config_from_har.py](../../scripts/fcloud/refresh_console_config_from_har.py)
   (~190 lines) — parses a DevTools-exported HAR, extracts authorization /
   cookie / JOB_ID, decodes JWT for username, backs up old
   `~/.fcloud_console_config`, writes new file with mode `0600`.
3. End-to-end tested with synthetic HAR (well-formed JWT, JSESSIONID cookie,
   start endpoint URL); refresh summary printed and `.bak` left in place.
4. Updated `.github/copilot-instructions.md` and
   `scripts/fcloud/fcloud_console_config.example` to point to the new fast
   path.
5. Commit `a6b16370e` pushed to `minicpm-src/mixed_minicpm_cudagraph`.

### Outcome

- Future weekly refresh now takes ~30s of human time (F12 → click Start/Pause
  → "Save all as HAR with content" → `python3 refresh_console_config_from_har.py
  console.har`) plus an automated parse step.
- Honest documented limit: full F12 capture cannot be automated headlessly
  because (a) the JWT lives only in the page's `Authorization` request header
  (not `document.cookie`), (b) cross-origin XHRs from any non-console page
  cannot read it, (c) the SSO + 2FA login itself can't be automated
  responsibly.

## Round 2 — fcloud Tier 1 retest

### User ask

> "go"

(after confirming the proposed 9-command fcloud sequence and the env-var
plumbing approach `(I)` — no tooling change needed; `restart-server --env KEY=VAL`
already exists in `fcloud_workflow.py`.)

### Agent action / sequence

| # | Step | Outcome |
|---|------|---------|
| 1 | `console-token-info` | JWT valid, ~2.91 days remaining |
| 2 | `start-instance` | HTTP 200 — task started |
| 3 | poll fcloud_exec until ready | ready on first attempt |
| 4 | `sync` | pulled `a6b16370e`; `prepare_env.sh` copied to `/root/submission_sim/` |
| 5 | `restart-server` (Run A, env unset) | server up after 214s |
| 6 | `speed --variant all` then `speed --variant s1` (recover scrolled S1) | Run A: S1=110.53s, S8=40.56s, Smax=33.65s |
| 7 | `restart-server --env SOAR_TIER1_LONG_CONTEXT=1` | server up after 214s; cmdline verified via `/proc/$pid/cmdline` shows `--chunked-prefill-size 65536 --max-prefill-tokens 65536 --prefill-max-requests 4 --schedule-conservativeness 0.8` |
| 8 | `accuracy` | Run B accuracy = **78.73%** (mcq 56.67, cwe 83.67, fwe 100, niah 100, qa 53.33) |
| 9 | `speed --variant {s1,s8,smax}` | Run B: S1=111.36s, S8=40.49s, Smax=33.62s |
| 10 | `pause-instance` | first attempt 504 (upstream gateway), retry HTTP 200 — task paused |
| 11 | TEST_RESULTS_TRACKING.md updated | added Tier1-A-baseline + Tier1-B-candidate rows |
| 12 | this chat log | written (EN+ZH) |

### Quantitative results

| Metric | Run A (baseline) | Run B (TIER1=1) | Δ | Sig? |
|---|---|---|---|---|
| S1 (s) | 110.53 | 111.36 | +0.75% | no |
| S8 (s) | 40.56 | 40.49 | −0.17% | no |
| Smax (s) | 33.65 | 33.62 | −0.09% | no |
| Accuracy (orig) | — | 78.73% | — | n/a |
| Accuracy (norm) | — | 98.42% | — | C=0.96 tier |
| mcq | — | 56.67% | (=Test 29) | within noise |
| qa | — | 53.33% | (−3pt vs Test 12) | within noise |
| niah / fwe | — | 100% / 100% | (=) | — |

### Outcome / decision

- **Tier 1 args are NEUTRAL on local short-context speed dataset.** Expected
  outcome: our local `speed_*.jsonl` inputs are ~1K tokens, far below the 65K
  chunked-prefill threshold; the new args never engage on these prompts.
- **Local validates only the negative side**: no regression on short-context
  perf, no accuracy collapse (78.73% is identical to Test 29 baseline,
  −0.56pt vs Test 12 — both inside the ±1pt local floor).
- **Official upside cannot be measured locally**: the long-context official
  speed set (68% of inputs in 32K–512K range) is precisely where
  chunked-prefill=65K + `prefill-max-req=4` + `sched-cons=0.8` should help.
- **Decision**: ship `SOAR_TIER1_LONG_CONTEXT=1` as v21 candidate. The flag is
  fully off by default in `prepare_env.sh`, so v20 byte-equivalent reproduces
  with `unset SOAR_TIER1_LONG_CONTEXT`. v21 sets the export at the top of
  `prepare_env.sh` (or via a one-line edit) before submission packaging.

### Open follow-ups

- Submit v21 once #1 ship-decision is confirmed; then immediately:
  - run **#2-A**: `--torch-compile-max-bs` sweep on top of v21 (try bs=16,
    possibly bs=24; user noted prior OOM but worth retesting on this fcloud).
  - watch leaderboard refresh ~24h after submission.
- Local speed dataset is misleading for Tier-1-style optimizations. Future
  speed proposals targeting prefill-throughput / chunk-size / scheduler
  should explicitly state "local sweep can't validate; ship-and-measure
  required" — same caveat applies here.
- **No new ZH-only or unrelated edits** in this round.

### Cross-references

- Proposal: [PROPOSAL_tier1_long_context_retest_20260430.en.md](../PROPOSAL_tier1_long_context_retest_20260430.en.md)
- Code commit (env switch): `d45b3ff1d` `feat(prepare_env): SOAR_TIER1_LONG_CONTEXT env switch`
- Code commit (refresh tooling): `a6b16370e` `feat(fcloud): HAR-based console-config refresh + bilingual howto`
- Test rows: TEST_RESULTS_TRACKING.md → `Tier1-A-baseline`, `Tier1-B-candidate`
- Superseded: [PROPOSAL_nvfp4_kv_dense_smoke_20260430.en.md](../PROPOSAL_nvfp4_kv_dense_smoke_20260430.en.md)
  (architectural blocker per CHANGE_0132 §8 — replaced by 2-A torch-compile-max-bs sweep).
