# 冲刺 Top 5 / Top 3 战略路线图 — SOAR 2026 (MiniCPM-SALA)

**日期**: 2026-04-23  
**当前排名**: team-beta 第 21 名，积分 **39.62**  
**冲击第 5 的差距**: ≥ 79.55 / 39.62 ≈ **需要 2.01 倍性能分**  
**当前工作基线**: GPTQ + FP8 KV (e5m2) + dense + torch.compile(max-bs=8) + mixed-chunk + v18 调度 (chunk=32K, prefill-max-req=1, sched-cons=1.0, running=24)

---

## 1. Tests 29–33 (新 fcloud SM120) 结论汇总

| 结论 | 证据 | 含义 |
|---|---|---|
| **新 fcloud 精度地板约 77–79%** | Tests 29/30/32/33 : 单变量调参均不能稳定提升平均分，mcq 在 40–96% 之间跳动 | 靠配置调参提精度是徒劳，需要结构性变更 |
| **各任务方差 ±20–40 分** | Test 29 mcq=96，Test 30 一次标志位翻转后 mcq=53 | 150 样本 concurrency=32 评测本身噪声很大 |
| **torch.compile 速度收益 ≈0%, 启动多 3 分钟** | Test 33 vs 29: 3016s vs 3005s；启动 36s vs 219s | 在 ≤5h 量化+评测总预算下启动成本很真实 |
| **KV 精度 (e5m2 vs e4m3) 只抖动各任务不变总分** | Test 30: cwe+25, fwe+14, mcq−43, 总分 −0.77 | 精度不是方差主因 |
| **调度 (v18 vs v19) 无法修复精度** | Test 32 v18 调度得 75.73%（C=0） | 调度激进度不是失败原因 |
| **评分机制是乘法式** | `Final = (S1×0.4 + S8×0.3 + Smax×0.3) × C` | 速度 +10% = 总分 +10%；C 0.96→1.0 只 +4.2% |

### 当前最佳成绩记录

| 指标 | 最佳场次 | 配置 | 数值 |
|---|---|---|---|
| 精度 | Test 20 | GPTQ + FP8 KV + dense + torch.compile(bs=8) + mixed-chunk + max-running-req=24 | acc_ori=**80.64%**, C=1.0 |
| S1 | Test 25A-spd | prefill-max-req=4, sched-cons=0.8, chunk=32K | **110.58s** |
| S8 | Test 24-spd | dense-calibrated (精度坏了) | **40.45s** |
| Smax | Test 20-spd | mixed-chunk + max-running-req=24 | **34.15s** |
| 当前 fcloud SM120 最优 | Test 29 | CHANGE_0130 回退后基线 | acc=78.73%, C=0.96 |

---

## 2. 下一步是拼精度还是拼速度？

**结论：速度是主要抓手，精度是条件性目标。**

### 为什么选速度

1. **精度收益递减**  
   C=0.96 → C=1.0 仅为 1.042× 乘数，性能分从被 0.96 倍 ×（乘以 0.96）提到 1.0 倍，只涨 **+4.2%**。而速度提 10% 直接 +10%，**影响是 2.4 倍**。

2. **排名差距是乘法形式**  
   从 39.62 → 79.55（第 5 名）= 需 **2.01× 总分**。即便 C 从 0.96 升到 1.0（1.042×），也仅到 41.3。要达到 79.55，**必须把性能分翻倍**。唯一路径：更快的 kernel 或投机解码。

3. **精度已经触及模型能力上限**  
   Tests 29–33 证明任何单配置都无法稳定把精度抬到 79% 以上。150 样本 concurrency=32 评测本身有 ±2–3 分噪声地板（批次交叉+思考链长度不确定性所致）。要稳定 C=1.0（norm≥99%）需要 acc_ori ≥ 79.2%，我们已经贴着红线。

### 何时应转向精度为主要目标

**仅当出现以下情况之一**：
- **v18 官方复评结果**（用户当前正在等待）acc_ori < 77%，会触发 C=0（直接淘汰）。这种情况下必须先稳定精度再优化速度。
- 或发现具体结构性 bug（如 mcq 思考链失控），可以通过针对性代码修复（非配置）解决。

---

## 3. 三轮迭代冲 Top 5 路线图

### 迭代 A — Marlin GPTQ SM120 解码 tile 专用化（高影响）

**目标**：解码 TPS +5–15%（S1 和 S8 下降 5–15%）  
**原因**：当前解码 TPS=439（Test 29）。SM120 BF16/FP16 峰值=148 TFLOPS。W4A16 小 M（batch 1–8）场景下 Marlin GEMM 严重浪费张量核。扩展 tile 表（CHANGE_0125）编译进了新形状但调度器很少选它。需要**主动偏好**小 M SM120 tile 的 dispatch 逻辑。

**行动计划**：
1. 在 SM120 上通过 `ncu` / PyTorch profiler 剖析 Marlin 调用，找出 tile 选择失配点
2. 在 `gptq_marlin.cu` 的 tile 表中为 M∈{1, 2, 4, 8} × N∈{4096, 7168, 13824, …}（MiniCPM-SALA MLP/QKV 形状）显式增加 (M, N, K, num_threads) 条目
3. 修改调度器（`heuristic.cc` 或等价文件）为 SM120 解码 tile 在 M ≤ 16 时增加评分偏置
4. 重编译 wheel（首次 ~4h / 增量 ~3min），跑速度基准确认

**风险**：kernel 构建时间；精度中性（tile 选择 = 相同数学）。回滚 = revert commit。  
**工作量**：1–2 次迭代。参考：vLLM M100 / TensorRT-LLM SM120 W4A16 kernel dispatch。  
**文档**：`CHANGE_0140_sm120_decode_marlin_tiles.{en,zh}.md`

---

### 迭代 B — 运行时调参 + 关闭 torch.compile 省启动时间（中影响）

**目标**：每次服务重启省 3 分钟（官方 5h 预算内若中途重启很关键），无运行时回归  
**原因**：Test 33 证明 torch.compile 在此工作负载下速度收益 ≈0% 但多耗 3 分钟预热。官方 5h 上限（量化+评测）下每分钟都很重要。CUDA graphs（sglang 自动）已覆盖解码路径。

**行动计划**：
1. 从 prepare_env.sh 移除 `--enable-torch-compile --torch-compile-max-bs 8`
2. 验证 `--cuda-graph-max-bs`（或等效）覆盖解码批次
3. 重新测速 — 确认无回归
4. 在迭代 A 之后应用，避免变量交叉

**风险**：极低。  
**工作量**：1 次测试。  
**打包**：与迭代 A 合并为 v21 submission。

---

### 迭代 C — 投机解码（高上限，高风险）

**目标**：在重复性任务上解码吞吐 +30–100%  
**原因**：此前 EAGLE3 尝试（CHANGE_0090）失败因为 draft 未训练（accept_rate=0.26）。一个可用的投机路径能提供接近 #5 所需的乘法级加速。

**两种方案**：

- **C1 — n-gram 投机解码（低工作量）**  
  sglang 内置。无需训练。匹配 prompt / 先前生成中的 token 模式。在 NIAH/FWE 这类重复性结构任务上效果最好。  
  预期：NIAH/FWE 专项 +10–30%。  
  工作量：1 次迭代（加标志+评测）。

- **C2 — 训练 EAGLE draft（高工作量，高回报）**  
  在提供的 calibration 数据上训练 1 层 EAGLE draft head（SM120 上约 1–2h GPU 时间）。目标 accept_rate ≥ 0.6。  
  预期：解码 TPS +50–100%。  
  工作量：多迭代，需训练基础设施和正确性保护。

**风险**：accept_rate 崩盘会比基线更慢；精度必须保持 >97% 归一化。  
**顺序**：C1 先做作为廉价探针；C2 只在 A+B 后仍有空间时启动。

---

## 4. 今天建议的执行顺序

1. **Test 34a — 复现 Test 20 基线** 在新 fcloud SM120（现有 GPTQ 模型）上。确认仍能稳定拿到 ~80% 精度 + C=0.96/1.0。同时跑速度基准确立当前 fcloud 基线数字（S1/S8/Smax）。
2. **Test 34b — 现场新量化** 通过 `prepare_model.sh` → 针对当前 CUDA/Triton/kernel 栈在 SM120 上新校准 GPTQ 模型。对比精度+速度 vs 34a。
3. **决策节点**：  
   - 若 34a 或 34b ≥ 79% acc_ori → 锁定为 v20 提交候选，推进 **迭代 A**（kernel 工作）。  
   - 若两者均 < 78% → 精度不稳定；调查 mcq 失控（通过自定义 eos_token_id 调参，或 decode 侧 logit_bias 强制终止）。
4. **并行等待 v18 官方复评结果** → 校准本地-官方差距。若 v18 落在 C=0.96 或 C=1.0，说明本地测试可靠，可放心提交 v20。

---

## 5. 决策触发表

| 触发条件 | 行动 |
|---|---|
| v18 官方 C=1.0 (norm≥99%) | 立即提交 v20 = Test 34a 配置；启动迭代 A |
| v18 官方 C=0.96 | 同上，并安排迭代 B（启动省时） |
| v18 官方 C=0.92 | 调查自原始 v18 以来服务端代码漂移；稳定前暂停激进速度工作 |
| v18 官方 C=0 | **精度成为主要目标**，调查 mcq 结构性修复 |
| 迭代 A 提速 ≥10% | 立即提交 v21，预期排名升至 #10–#15 |
| 迭代 A+B+C1 合计 ≥20% 且 C≥0.96 | 冲 Top 10 可行；冲 Top 5 还需迭代 C2 |

---

## 6. 回滚安全

每次迭代都 commit 到 `mixed_minicpm_cudagraph` 分支。要回滚任何变更：
```bash
git log --oneline -20
git revert <commit>
# 或：
git reset --hard <last_known_good>
git push minicpm-src mixed_minicpm_cudagraph --force-with-lease
```
对于 kernel 变更，`/root/submission_sim/sgl_kernel-*.whl` 下的预构建 wheel 是恢复资产。
