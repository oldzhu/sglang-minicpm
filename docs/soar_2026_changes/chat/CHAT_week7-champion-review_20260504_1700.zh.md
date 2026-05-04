# 对话记录 — v22 打包 + 第 7 周冠军复盘 (2026-05-04)

## 第 1 轮 — 用户："package v22 now and regarding the next, ..."

用户三项要求：
1. 立即打包 v22；
2. 学习并归档第 7 周冠军博客（https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q）：NVFP4 FourOverSix + Medusa GLA tree-verify；
3. 重新抓取排行榜 —— 当前排名。

### Agent 操作

1. **打包 v22**（与调研并行）：
   - `start-instance`（HTTP 200），等待 jupyter 75s，`sync` 拉取 234f3fed8 + 复制 prepare_env.sh；
   - fcloud 端：`tar` + sha256：`d550e6ea7a1e4cedf933a76b6dcb227191abfc1e84fb42c3161e902b7cc54030`，778847649 字节；
   - 软链 `/workspace/minicpm_sala_submit_v22.tar.gz → /root/...`；
   - 通过 `/files/...?token=...` 下载，本地 sha256 校验一致；
   - `pause-instance`（首次 504，重试 200 —— 已知规律）。

2. **冠军博客抓取** —— 抽取核心内容：
   - **FourOverSix**（arXiv:2512.02010，MIT+NVIDIA）：在 NVFP4 量化中按 block 自适应 M=6 vs M=4。约 40-43% block 选择 M=4。输出格式不变（4-bit + FP8 scale）；kernel 不动。嵌入 GPTQ 迭代之内。
   - **Medusa**（ICML 2024，Cai 等）适配 MiniCPM-SALA 混合注意力：tree verify 要求 sibling 分支各自从同一 parent GLA state 出发（而非线性继承）。Per-branch GLA 递推 override + 训练数据按评测分布加权采样。K=1 verify 开销 0.39 ms。

3. **重新抓取排行榜（2026-05-04 13:56）**：
   | # | 队伍 | 得分 |
   |---|------|------|
   | 1 | 香草小张 | 88.35（第 7 周冠军 —— 即本博客方案）|
   | 2 | FlashSALA | 86.68 |
   | 3 | dwq | 67.8 |
   | 4 | Slightwind | 62.05 |
   | 5 | 智算一队 | 50.66 |
   | 22 | **team-beta** | **30.04** |
   - 榜首 1-2 已与第 3 名（67.8）拉开决定性差距 —— 应是同一方案。
   - 距 #5 差 +20.62（68%）。
   - 已更新 `/memories/soar_2026_leaderboard.md` 快照。

4. **文档新建**（中英对照，均位于 `docs/soar_2026_changes/`）：
   - [RESEARCH_week7_champion_review_20260504.en.md](../RESEARCH_week7_champion_review_20260504.en.md) / [zh](../RESEARCH_week7_champion_review_20260504.zh.md) —— 详细记录两个支柱：含代码片段、硬件论证、未决问题。
   - [PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md](../PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.en.md) / [zh](../PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md) —— 4 阶段实施方案（A: NVFP4 基线；B: FourOverSix；C: Medusa head 训练；D: sglang GLA-aware tree verify），每段独立可发版、env-gate、含成功/失败判据与回滚。推荐先批 A。

### 产出

- v22 tarball：`benchmark/soar/demo_sala/minicpm_sala_submit_v22.tar.gz`（sha256 d550e6ea...4030，743MB）；
- fcloud 已暂停；
- 研究 + 提案文档 4 份完成，待用户评审；
- 排行榜内存已更新。

### 待用户回复的问题

- 阶段顺序：仅批 A，还是 A+B，还是 A-D 一并承诺？
- Phase C 需非 fcloud GPU（H100/A100），是否可用？

### 交叉引用

- 前置版本：v22（`234f3fed8`）
- 冠军博客：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- FourOverSix：arXiv:2512.02010
- 内存更新：`/memories/soar_2026_leaderboard.md`
