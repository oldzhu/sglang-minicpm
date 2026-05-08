# CHANGE_0151 — Phase B FourOverSix NVFP4(续 001：迭代 2 与结论)

承接 [CHANGE_0151_phase_b_four_over_six.zh.md](CHANGE_0151_phase_b_four_over_six.zh.md)。
本迭代结束时分支 HEAD：`a6b34a41a`,位于 `minicpm-src/mixed_minicpm_cudagraph`。

## 背景

迭代 1（CHANGE_0151）得到 **ori 75.98%**(< 77% 阈值,C=0)。本会话第 2 轮重跑同一 iter-1 ckpt 得到 **70.27%**,`mcq` 出现 runaway-think(avg_out=10124),证实结果是调度引发的运行间方差,与
[TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) 中 GPTQ Tests 30/32/33 的模式一致。

迭代 2 假设两个稳定器:
1. **保守调度**(Test 12 系列):`chunk=32K`、`prefill-max-req=1`、
   `schedule-conservativeness=1.0`、`torch-compile-max-bs=8`。消除与长 thinking
   块交互不良的激进批处理。
2. **更长 + 任务平衡的校准**:`SOAR_NVFP4_MAX_CALIB_SEQ_LEN=16384`(原默认 4096)
   且使用 90 个分层抽样的 `qa,mcq,cwe` 样本,而非默认 `SOAR_GPTQ_CALIBRATION_*`
   产生的 32 个顺序样本。

## 实现

`benchmark/soar/demo_sala/prepare_env.sh`(commit `a6b34a41a`)— 在
`SOAR_QUANT_PROFILE=nvfp4_fos` 分支内:

```bash
export SOAR_NVFP4_FOUR_OVER_SIX="${SOAR_NVFP4_FOUR_OVER_SIX:-1}"
export SOAR_NVFP4_MAX_CALIB_SEQ_LEN="${SOAR_NVFP4_MAX_CALIB_SEQ_LEN:-16384}"
export SOAR_TIER1_LONG_CONTEXT=0   # 强制走保守调度分支
export SOAR_TORCH_COMPILE_MAX_BS="${SOAR_TORCH_COMPILE_MAX_BS:-8}"
```

后续的 `SOAR_GPTQ_CALIBRATION_*` 默认值(samples=90, stratified,
task_include=qa,mcq,cwe)被继承,因为 `load_calibration_texts` 辅助函数被 GPTQ
和 NVFP4 路径共享。

## 验证命令

```bash
# 量化(同步执行 — 不用 nohup,约 3 分钟)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cd /root/submission_sim && export SOAR_QUANT_PROFILE=nvfp4_fos SOAR_NVFP4_FOUR_OVER_SIX=1 \
   SOAR_QUANT_FORCE=1 && source ./prepare_env.sh >/tmp/prep.log 2>&1; \
   python3 -u preprocess_model.py --input /root/models/openbmb/MiniCPM-SALA \
     --output /root/models/MiniCPM-SALA-NVFP4-FOS --mode nvfp4' --timeout 2400

# 复制 tokenizer(modelopt 手动导出不包含)
python3 scripts/fcloud/fcloud_exec.py exec \
  'cp /root/models/openbmb/MiniCPM-SALA/tokenizer.* \
      /root/models/openbmb/MiniCPM-SALA/special_tokens_map.json \
      /root/models/MiniCPM-SALA-NVFP4-FOS/'

# 启动 server、跑 2 次 accuracy、跑 speed
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos --env SOAR_NVFP4_FOUR_OVER_SIX=1
python3 scripts/fcloud/fcloud_workflow.py accuracy --model-path /root/models/MiniCPM-SALA-NVFP4-FOS  # x2
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 结果 — 迭代 2 相对迭代 1 是退化

### 准确率(2 次)

| 运行 | ori_acc | norm_acc | duration | mcq | qa | niah | cwe | fwe | TPS |
|----:|--------:|---------:|---------:|----:|---:|-----:|----:|----:|----:|
| Iter 1 run 1 | 75.98% | (gate) | — | 63.33 | 56.67 | 93.33 | 74.33 | 92.22 | — |
| Iter 1 run 2 | 70.27% | — | 2536.95s | 46.67 | 50.00 | 83.33 | 74.67 | 96.67 | 319.44 |
| **Iter 2 run 1** | **60.73%** | 75.92% | 3614.90s | 50.00 | 50.00 | **73.33** | 60.33 | **70.00** | 227.36 |
| **Iter 2 run 2** | **63.31%** | 79.14% | 3631.80s | 53.33 | **36.67** | 70.00 | 74.33 | 82.22 | 262.18 |

- **迭代 2 平均 ori = 62.02%**(对比迭代 1 平均 ≈ 73.13%)— **回退 11pt**。
- 方差仍高(60.73 vs 63.31,±1.3pt),但**失败模式发生了转移**:迭代 1 是 mcq
  失控,迭代 2 是 fwe(run 2)或 niah/cwe(run 1)崩溃。runaway-think 生成
  并未消除,只是在任务间轮转。
- 本会话三次量化运行的 FOS pct_m4 均为 **43.14%**(默认 sequential-32、
  stratified-90+4096、stratified-90+16384)— 证实 FOS 的尺度选择**仅由权重决定**,
  校准数据不影响 M=4 vs M=6 的选择。
- 因此准确率退化来自校准期间**激活 amax 的变化**(更偏向 `qa/mcq/cwe` + 更长
  context → 不同的逐张量尺度 → `niah/fwe` 性能下降)。

### 速度

| | S1 | S8 | Smax |
|---|---:|---:|---:|
| GPTQ T12 baseline | 121.71s | 44.09s | 35.86s |
| Iter 1 (NVFP4-FOS, Tier1 long-ctx) | 175.08s | 47.37s | 31.01s |
| **Iter 2 (NVFP4-FOS, conservative)** | **173.69s** | **45.95s** | **34.39s** |

- S1/S8 与迭代 1 基本持平;保守调度未在本地短 prompt 集上恢复 S1 吞吐。
- Smax 变慢 11%(31.01s → 34.39s),因为 `torch-compile-max-bs=8`
  使 batch 在 [9, 24] 范围内退回 eager 模式。

## 结论 — Phase B FOS 作为即插即用方案不可行

两次迭代均未通过 C ≠ 0 阈值:
- **Iter 1**:75.98% / 70.27%(方差跨越 77% 线,均值 73%)。
- **Iter 2**:60.73% / 63.31%(明确低于 77%,均值 62%)。

迭代 2(保守调度 + 更长/平衡校准)能稳定准确率的假设**被否定**。两个改动要么对方差源
(调度)无效,要么使情况变差(校准)。

根因分析指向 **MiniCPM-SALA 在 `niah`/`fwe`/`qa` 长 context 路径上的激活离群值,
NVFP4 的 16 元素块粒度 FP4 无法表示**。FOS 仅调整每块的尺度(权重侧),无法扩展通过
`input_quantizer` 的激活动态范围。模型需要:

1. **混合精度**:将 `q_proj/k_proj/v_proj`(或专门为稀疏/lightning attention 提供
   feature 的层)保持在 int8 或 bf16,仅 MLP 用 NVFP4。迭代 1 的"下一步选项"将
   其列为 B 选项(工作量大)。
2. **NVFP4 配 INT8 `input_quantizer`** 替代 FP8 input scale — 但需要我们尚未验证
   的 modelopt 配置改动。
3. **完全跳过 Phase B**,回归 GPTQ + FP8 KV 速度优化,这是当前最佳已知配置
   (T12:ori 79.29%, norm 99.11%, C=1.0)。

## 回滚

```bash
git revert a6b34a41a   # 迭代 2 的 prepare_env 设置
# 或显式切回 gptq profile:
export SOAR_QUANT_PROFILE=gptq
```

迭代 2 的量化 ckpt `/root/models/MiniCPM-SALA-NVFP4-FOS` 可以删除(`rm -rf`)— 迭代 1
的 CHANGE_0151 ckpt 在本迭代被覆盖,但两者的重新生成方式都是 prepare_env profile 切换。

## 下一步建议

鉴于 FOS 两次迭代均未能通过 77% 阈值,推荐优先级为:

1. **暂停 Phase B FOS**,直到我们有清晰方案修复 niah/fwe 路径上的激活量化(混合
   精度或不同的 `input_quantizer` 配置)。
2. **回归 GPTQ+FP8+dense** 基线(T12 系列)。`OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`
   中尚未穷尽的优化向量:长 context 调度扫描、稀疏 attention 重启探测、kernel 融合。
3. 若必须继续 NVFP4:试 plain NVFP4(无 FOS,`SOAR_QUANT_PROFILE=nvfp4`) +
   迭代 1 默认校准(32 sequential,4096 seq)— 测试"迭代 1 之所以工作是因为
   LUCKY 校准样本选择"假设,无需设计新的混合精度配方。

更新的测试结果行见 [TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md)
中 Phase B 下的 NVFP4-FOS-2 与 NVFP4-FOS-2b。
