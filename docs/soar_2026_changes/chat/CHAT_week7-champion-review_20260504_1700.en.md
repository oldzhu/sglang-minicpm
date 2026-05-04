# CHAT — v22 packaging + Week 7 champion review (2026-05-04)

## Round 1 — User: "package v22 now and regarding the next, ..."

User asked for three items:
1. Package v22 now.
2. Review and document Week 7 champion's blog (https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q): NVFP4 FourOverSix + Medusa GLA tree-verify.
3. Re-fetch leaderboard — current rank.

### Agent actions

1. **Packaging v22** (in parallel with research fetches):
   - `start-instance` (HTTP 200), waited 75s for jupyter, `sync` pulled 234f3fed8 + copied prepare_env.sh.
   - On fcloud: `tar` + sha256: `d550e6ea7a1e4cedf933a76b6dcb227191abfc1e84fb42c3161e902b7cc54030`, 778847649 bytes.
   - Symlinked `/workspace/minicpm_sala_submit_v22.tar.gz → /root/...`.
   - Downloaded via `/files/...?token=...`, sha256 verified locally identical.
   - `pause-instance` (504 first call, 200 retry — known pattern).

2. **Champion blog fetched** — extracted core content:
   - **FourOverSix** (arXiv:2512.02010, MIT+NVIDIA): per-block adaptive M=6 vs M=4 in NVFP4 quantization. ~40-43% blocks pick M=4. Output format unchanged (4-bit + FP8 scale); kernel untouched. Embedded inside GPTQ iteration.
   - **Medusa** (ICML 2024, Cai et al.) adapted to MiniCPM-SALA's GLA hybrid attention: tree verify needs sibling branches each fork from same parent GLA state (not linear inheritance). Per-branch GLA recurrence override + training data weighted toward eval distribution. K=1 verify overhead 0.39 ms.

3. **Leaderboard re-fetched (2026-05-04 13:56)**:
   | # | Team | Score |
   |---|------|-------|
   | 1 | 香草小张 | 88.35 (Week 7 champ — this blog's recipe) |
   | 2 | FlashSALA | 86.68 |
   | 3 | dwq | 67.8 |
   | 4 | Slightwind | 62.05 |
   | 5 | 智算一队 | 50.66 |
   | 22 | **team-beta** | **30.04** |
   - Top-2 broke decisively from rest (#3 = 67.8). Same recipe both teams.
   - Gap to #5: +20.62 (68% improvement).
   - Updated `/memories/soar_2026_leaderboard.md` with new snapshot.

4. **Documents created** (bilingual, all under `docs/soar_2026_changes/`):
   - [RESEARCH_week7_champion_review_20260504.en.md](../RESEARCH_week7_champion_review_20260504.en.md) / [zh](../RESEARCH_week7_champion_review_20260504.zh.md) — captures both pillars in detail with code snippets, hardware rationale, and open questions.
   - [PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md](../PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md) / [zh](../PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md) — 4-phase implementation plan (A: NVFP4 baseline; B: FourOverSix; C: Medusa head training; D: sglang GLA-aware tree verify), each phase independently shippable with env-gate, success/failure matrix, rollback. Recommends approve A first.

### Outcomes

- v22 tarball ready at `benchmark/soar/demo_sala/minicpm_sala_submit_v22.tar.gz` (sha256 d550e6ea...4030, 743MB).
- fcloud paused.
- Research + proposal pair (4 docs) drafted, ready for user review.
- Leaderboard memory updated.

### Open questions for user

- Phase ordering: approve A only, or A+B together, or commit to A-D up front?
- Phase C requires off-fcloud GPU (H100/A100). Available?

### Cross-references

- Predecessor: v22 (`234f3fed8`)
- Champion blog: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- FourOverSix: arXiv:2512.02010
- Memory updated: `/memories/soar_2026_leaderboard.md`
