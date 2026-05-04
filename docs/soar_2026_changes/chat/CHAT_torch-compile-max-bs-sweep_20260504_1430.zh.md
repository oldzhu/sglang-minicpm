# 对话记录 — #2-A torch-compile-max-bs 扫描 + v22 默认翻转 (2026-05-04)

## 第 1 轮 — 用户："approve"

用户批准了 [PROPOSAL_torch_compile_max_bs_sweep_20260504](../PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md)。

### Agent 操作

1. 修改 [benchmark/soar/demo_sala/prepare_env.sh](../../../benchmark/soar/demo_sala/prepare_env.sh) 的 gptq 分支（约 187 行）：
   ```
   SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
   TORCH_COMPILE_ARGS=" --enable-torch-compile --torch-compile-max-bs ${SOAR_TORCH_COMPILE_MAX_BS}"
   ```
   作为提交 `09af88b14` 推送到 `minicpm-src/mixed_minicpm_cudagraph`。
2. `start-instance` → `sync` → fcloud 同步至提交 `09af88b14`。
3. **bs=16 轮**（`restart-server --env SOAR_TORCH_COMPILE_MAX_BS=16`）：
   - 准确率：**79.11%**（归一化 98.89%，C=0.96），3064.24s
   - 速度：S1=110.56s, S8=40.47s, **Smax=33.36s**
   - 相对 Tier1-B（111.36 / 40.49 / 33.62）：均处于噪声范围，Smax −0.8%。
4. **bs=24 轮**（`restart-server --env SOAR_TORCH_COMPILE_MAX_BS=24`）：
   - 准确率：**79.11%**（归一化 98.89%，C=0.96），3219.91s
     - mcq=60.00%（+3.33pt），qa=56.67%（+3.33pt），cwe=83.33%，niah=96.67%，fwe=98.89%
   - 速度：S1=111.55s, S8=40.46s, **Smax=32.54s**
   - 相对 Tier1-B：**Smax −3.2%**（33.62 → 32.54s）—— 明确胜出。
5. `pause-instance`（首次 504，重试 200 —— 已知规律）。

### 决策

bs=24 命中成功判据第一档（acc ≥ 78% 且 Smax 接近 32s）。作为 v22 默认开启发版：

- `prepare_env.sh` 默认值由 `:-8` 翻转为 `:-24`。
- 回滚：`SOAR_TORCH_COMPILE_MAX_BS=8` 即恢复为 v21 字节级等价。

### 产出

- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) 新增两行：`#2A-bs16`、`#2A-bs24`。
- `prepare_env.sh` v22 默认开启提交紧随其后。

### 交叉引用

- 提案：[PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md](../PROPOSAL_torch_compile_max_bs_sweep_20260504.zh.md) / [en](../PROPOSAL_torch_compile_max_bs_sweep_20260504.en.md)
- 前置：v21 默认开启 `SOAR_TIER1_LONG_CONTEXT=1`（提交 `edf97175e`）
- patch（提案）：`09af88b14`
- patch（v22 翻转）：待提交
