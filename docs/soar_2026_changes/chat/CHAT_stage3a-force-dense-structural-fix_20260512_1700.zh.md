# 聊天记录：Stage 3a Force-Dense 结构性修复验证
**主题**：`stage3a-force-dense-structural-fix`  
**日期**：2026-05-12  
**参与者**：用户（oldzhu）、Agent（GitHub Copilot）  
**fcloud 实例**：ai-e7e98a7c52

---

## 1. 会话开始 — 收集准确性测试结果

**背景**：上一会话以 `SOAR_BACKEND_KEEP_FORCE_DENSE=1`（强制启用 `--force-dense-minicpm`）
启动了准确性评估，在 145/150 个请求时超时。评估有 2 个 HTTP 超时失败
（请求 127-128，长 NIAH 任务）。fcloud 保持运行状态。

**Agent 操作**：发现输出目录 `/root/data/outputs/20260512_083652/` 为空（评估进程在
3600s 时崩溃，未能写入 `summary.json`）。检测到评估进程已退出。以 2 小时超时重新运行：

```bash
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/data && python3 eval_model_001.py --api_base http://127.0.0.1:30000 \
   --model_path /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8 \
   --data_path /root/data/perf_public_set.jsonl --concurrency 32 2>&1' \
  --timeout 7200
```

**准确性结果**：
```
Average Score: 77.87%
Total Duration: 3338.76s
Total Tokens: In=8,644,406  Out=1,235,925

Per-task:
  cwe:  86.00%  (avg_out=24,453)
  fwe:  100.00% (avg_out=7,287)
  mcq:  46.67%  (avg_out=8,997)  ← 思维链失控
  niah: 100.00% (avg_out=352)
  qa:   56.67%  (avg_out=108)
```

---

## 2. 速度测试（force-dense 激活状态）

服务器仍以 `SOAR_BACKEND_KEEP_FORCE_DENSE=1` 运行。

**结果**：
| 变体 | 耗时 | 对比 Stage3a 无 force-dense |
|------|------|---------------------------|
| S1 | 202.70s | 202.96s（−0.1%） |
| S8 | 61.60s | 61.65s（−0.1%） |
| Smax | 43.29s | 43.40s（−0.3%） |

**结论**：添加 `--force-dense-minicpm` 对速度零可测量影响。

---

## 3. fcloud 已暂停

```bash
python3 scripts/fcloud/fcloud_workflow.py pause-instance
# → HTTP 200 "任务已暂停"
```
（第一次尝试：HTTP 504，重试成功。）

---

## 4. 分析与决策

### 4.1 为何 Force-Dense 对速度无影响

`--force-dense-minicpm` 仅在初始化时控制池类型选择：
- False：`MiniCPMHybridReqToTokenPool`（分配 K1/K2 表）
- True：`HybridReqToTokenPool`（无 K1/K2）

在 flashinfer 后端下，两种池类型都不影响前向计算或 KV 缓存访问模式。
GPU 计算内核完全相同。

### 4.2 结构性修复决策

**结果**：将 `SOAR_BACKEND_KEEP_FORCE_DENSE=1` 设为 `prepare_env.sh` 的默认值。

理由：
1. 零速度/准确性成本（已验证）
2. 消除整个 K1/K2 崩溃类——CHANGE_0158/0161 补丁变为不可达代码
3. 比补丁级修复更简洁；热路径中无运行时分支
4. 与 Stage 2 直通行为一致（该版本已使用 force-dense）

### 4.3 准确性分析

| 配置 | 准确性 | 归一化 | C |
|------|--------|--------|---|
| Stage 2（无 MEDUSA） | 80.11% | ~100% | 1.0 |
| Stage 3a（CHANGE_0161，无 force-dense） | 76.04% | ~95% | 0 |
| **Stage 3a（CHANGE_0162，force-dense）** | **77.87%** | **~97.34%** | **0.92** |

C=0.92 惩罚源于 mcq=46.67%（归一化≈97.34% < 98% → C=0.92 而非 1.0）。
mcq 思维链失控（avg_out=8,997 tokens）是影响 force-dense 和无 force-dense
两种配置的预先存在的 Stage 3a 问题。

**无论是否使用 force-dense，Stage 3a 都不是提交候选**：
- S1=202.70s 对比 Stage 2 基线 118.28s（+72% 更慢）——零初始化 heads 拒绝所有草稿
- 由于 mcq 失控，C=0.92——8% 性能惩罚

---

## 5. 已应用变更

### `benchmark/soar/demo_sala/prepare_env.sh`
- 添加 `export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"` 默认值
- 放置在 `export SOAR_BACKEND_VARIANT="${SOAR_BACKEND_VARIANT:-flashinfer}"` 之后
- 无逻辑变更；现有条件判断已正确处理此情况

### `docs/soar_2026_changes/TEST_RESULTS_TRACKING.md`
- 在主表中添加 **Stage3a-force-dense** 行
- 记录：commit 50e9466d0，准确性 77.87%（norm 97.34%，C=0.92），S1/S8/Smax

### 新建文档：
- `CHANGE_0162_force_dense_default_structural_fix.en.md`
- `CHANGE_0162_force_dense_default_structural_fix.zh.md`

---

## 6. 后续步骤（待处理）

### 优先：Stage 3b 训练好的 Medusa Heads

Stage 3a 建立了基础设施。Stage 3b 需要：
1. **训练数据收集**：在评估提示上运行基础模型，捕获隐藏状态 + 标签
2. **Head 训练**：1 个 ResBlock MedusaHead（hidden=4096 → vocab=122753），冻结基础模型，5 个 epoch
3. **接入**：用 head 前向替换 `MedusaWorker._forward_generate_k1` 中的零初始化草稿
4. **预期增益**：accept_rate ~60-70% → spec_accept_length ~1.6-1.7 → ~20-30% 加速
5. **大小**：K=1 head ≈ 1 GB BF16 — 在 2 GB 提交限制内

### mcq 准确性调查

MEDUSA 使 mcq 下降 ~10-15pt（47% 对比 Stage 2 的 ~57%）。可能原因：
- MEDUSA 验证开销改变了有效 token 预算
- GLA 状态与思维格式 MCQ 提示的交互
- 推测解码循环中 max_tokens 计数差异

在 Stage 3b 之前值得调查，以确保真实接受不会进一步降低准确性。

---

## 7. 交叉引用

| 项目 | 链接 |
|------|------|
| CHANGE_0162 EN | [CHANGE_0162_force_dense_default_structural_fix.en.md](CHANGE_0162_force_dense_default_structural_fix.en.md) |
| CHANGE_0162 ZH | [CHANGE_0162_force_dense_default_structural_fix.zh.md](CHANGE_0162_force_dense_default_structural_fix.zh.md) |
| 测试结果 | TEST_RESULTS_TRACKING.md 行 "Stage3a-force-dense" |
| 上一会话聊天记录 | CHAT_stage3a-k1-crash-k1k2-fix_20260512_XXXX.en.md |
| CHANGE_0161 | CHANGE_0161_medusa_stale_k1k2_zero_out.en.md |
