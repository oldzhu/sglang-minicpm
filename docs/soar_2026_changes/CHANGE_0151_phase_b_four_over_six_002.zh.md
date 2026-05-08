# CHANGE 0151 — Phase B FourOverSix 续篇 002

承接 [CHANGE_0151_phase_b_four_over_six_001.zh.md](CHANGE_0151_phase_b_four_over_six_001.zh.md)。
续 001 给出"放弃 FOS"的结论；用户推翻该结论，要求在最终下线前再做一次 A/B 测试。

## 背景

迭代 2 相对迭代 1 同时改变了两个变量：

| 调节项 | 迭代 1 | 迭代 2 |
|--------|--------|--------|
| 校准集 | 顺序 32 条（无任务过滤） | 分层抽样 90 条，仅 `qa,mcq,cwe` |
| `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` | 4096 | 16384 |
| 调度策略 | Tier1 长上下文（chunk=65536，prefill-max-req=4，sched-cons=0.8） | 保守 Test 12（chunk=32K，prefill-max-req=1，sched-cons=1.0） |
| `SOAR_TORCH_COMPILE_MAX_BS` | 24 | 8 |

迭代 1 平均 ori ≈ 73%；迭代 2 平均 ori ≈ 62%（−11pt）。要归因这一回归，必须把
调度与校准两个变量解耦。迭代 3 锁定迭代 2 ckpt，仅回退调度策略。

## 实施

### 服务端参数补丁

`benchmark/soar/demo_sala/prepare_env.sh` 第 61 行（在
`SOAR_QUANT_PROFILE == nvfp4_fos` 分支内）由：

```bash
export SOAR_TIER1_LONG_CONTEXT=0
```

改为：

```bash
export SOAR_TIER1_LONG_CONTEXT="${SOAR_TIER1_LONG_CONTEXT:-0}"
```

保留迭代 2 默认值（`0` → 保守调度），同时允许调用方覆盖（例如
`fcloud_workflow.py restart-server --env SOAR_TIER1_LONG_CONTEXT=1`）。
`SOAR_TORCH_COMPILE_MAX_BS` 已使用 `${VAR:-8}` 形式，无需补丁。

提交：`minicpm-src/mixed_minicpm_cudagraph` 上的 `fb6ee34d8`。

### fcloud 同步注意事项

fcloud 上 `/root/sglang-minicpm` 不是 git clone（来源是初始化时解压的
`submission_sim.tar` 快照），因此 `fcloud_workflow.py sync` 走的是从陈旧
快照树强制覆盖的回退路径，不会拉到本地补丁文件。绕过方法：直接通过
`base64 | base64 -d`（fcloud_exec）上传补丁后的 `prepare_env.sh`：

```bash
B64=$(base64 -w0 benchmark/soar/demo_sala/prepare_env.sh)
python3 scripts/fcloud/fcloud_exec.py exec \
  "echo '$B64' | base64 -d > /root/submission_sim/prepare_env.sh"
```

### 迭代 3 服务端启动

```
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --env SOAR_QUANT_PROFILE=nvfp4_fos \
  --env SOAR_NVFP4_FOUR_OVER_SIX=1 \
  --env SOAR_TIER1_LONG_CONTEXT=1 \
  --env SOAR_TORCH_COMPILE_MAX_BS=24
```

精度测试前实测 `/get_server_info`：

```
chunked_prefill = 65536
prefill_max_req = 4
sched_cons      = 0.8
max_run         = 24
quantization    = modelopt_fp4
kv_cache_dtype  = fp8_e5m2
```

## 预设中止门限

按用户指示：
> 如果第 1 次精度低于 70%，就停实例，把 `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096`
> 改回迭代 1 配置后重新量化测试。

迭代 3 第一轮 ori_accuracy < 70% → 跳过第二轮和速度测试，暂停实例，进入
迭代 4（calib_seq_len=4096 重新量化）。

## 结果 — 迭代 3 第 1 轮

`outputs/20260508_025542/predictions.jsonl`

| 指标 | 数值 |
|------|------|
| ori_accuracy（Average Score） | **68.20%** |
| cwe | 77.67% |
| fwe | 76.67% |
| mcq | 56.67% |
| niah | 80.00% |
| qa | 50.00% |
| 总时长 | 3179.25 s |
| 输出 TPS | 252.24 |

**触发中止门限**，跳过第 2 轮和速度测试。

## 对比

| 指标 | 迭代 1（均值） | 迭代 2（均值） | 迭代 3 run-1 |
|------|---------------|---------------|---------------|
| ori_accuracy | ~73.1% | ~62.0% | 68.20% |
| Δ vs 迭代 1 | — | −11pt | −5pt |
| Δ vs 迭代 2 | +11pt | — | +6pt |

仅恢复 Tier1 调度即可挽回迭代 1→迭代 2 回归中 ~6pt（约一半），剩余 ~5pt
应归因于校准集变化（seqlen 4096→16384 和/或内容 sequential→stratified-qa,mcq,cwe）。

## 下一步 — 迭代 4 计划

把 FOS ckpt 用 `SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096` 重新量化，保持迭代 2 的
"分层抽样 90 条 qa,mcq,cwe"（相同内容，更短 seqlen），借以独立衡量 seqlen
的影响 —— 这与迭代 1 的"顺序 32 条混合"内容并不相同。

```bash
SOAR_QUANT_PROFILE=nvfp4_fos \
SOAR_NVFP4_FOUR_OVER_SIX=1 \
SOAR_NVFP4_MAX_CALIB_SEQ_LEN=4096 \
SOAR_GPTQ_CALIBRATION_SAMPLES=90 \
SOAR_GPTQ_CALIBRATION_SAMPLING=stratified \
SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe \
SOAR_GPTQ_CALIBRATION_SEED=20260320 \
python3 preprocess_model.py --input <BF16> --output /root/models/MiniCPM-SALA-NVFP4-FOS
```

再用迭代 3 的服务端配置（TIER1=1，TORCH_COMPILE_MAX_BS=24）测试。

若迭代 4 仍 <70% → 校准 *内容*（qa,mcq,cwe 分层）才是主导变量，与迭代 1
的"无过滤顺序混合"差异显著。后续可选：
- 迭代 5A：去掉 `TASK_INCLUDE` 过滤（混合 5 类任务）
- 迭代 5B：`SAMPLES=32` 顺序，与迭代 1 完全对齐
- 选项 C：彻底放弃 FOS。

## 验证命令

```bash
# 启动后核对实际服务端配置
curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool

# 再次运行精度
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS
```

## 回滚

撤销提交 `fb6ee34d8`（单行补丁）。当调用方未传 `SOAR_TIER1_LONG_CONTEXT` 时，
`nvfp4_fos` 分支默认行为不变，与迭代 2 完全兼容。

## 关联引用

- 续 001：[CHANGE_0151_phase_b_four_over_six_001.zh.md](CHANGE_0151_phase_b_four_over_six_001.zh.md)
- 测试登记：TEST_RESULTS_TRACKING.md → NVFP4-FOS-3
- 对话记录：chat/CHAT_phase-b-fos-iter3_20260508_0250.zh.md
