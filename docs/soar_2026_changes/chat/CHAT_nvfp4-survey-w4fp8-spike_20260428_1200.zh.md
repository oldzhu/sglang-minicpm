# 会话记录 —— 第 10 轮：NVFP4 KV P1 调研 + W4-FP8 spike 脚手架（2026-04-28 12:00）

W4A8 深度会话延续。

## 用户提问
"在我审阅 explain 文档时，请进行 NVFP4 KV 提案（先做 P1 调研）和 1 天 W4-FP8 spike。"

## Agent 操作

### 1. NVFP4 KV P1 调研（离线，无 fcloud）
搜索 `python/sglang/srt/` 下所有 `nvfp4|fp4|e2m1` 与 `kv_cache_dtype` 引用。

**发现**：
- 树内已有约 80% 的 plumbing：server arg `--kv-cache-dtype fp4_e2m1`、`model_runner.configure_kv_cache_dtype`、`MHATokenToKVPoolFP4`、`KVFP4QuantizeUtil`（MXFP4 量化/反量化通过 `@torch.compile`）。
- **重要标签纠正**：sglang 的 `fp4_e2m1` 实为 **MXFP4**（16 元素 block、e8m0 uint8 scale），不是 NVFP4（e4m3 fp32 scale）。对我们而言这反而更好——MXFP4 在 SM120 上更硬件友好，省内存更多。
- 真实缺口 100% 在我们的自定义 **`minicpm_backend.py`**：4 个门控目前以 `kv_cache_dtype_str.startswith("fp8")` 为判据，需要扩展到 `fp4_e2m1`；以及 `set_kv_buffer` k_scale 在 FP4 时应传 `None`。
- 内存：MXFP4 KV **比 FP8 小约 44%**（1152 vs 2048 字节/token/层，按我们的形状）。
- 更新后的 P2 计划：约 3 天（先用 `--force-dense-minicpm` 冒烟以隔离 plumbing 与 sparse-attention 门控）。

**文档**：[SURVEY_NVFP4_KV_P1_20260428_1130.{en,zh}.md](../SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)

**P2 启动前 3 个用户待答**：
1. fcloud 上 PyTorch ≥ 2.8 吗（需 `torch.float4_e2m1fn_x2`）？
2. 批准 P2 从 `--force-dense-minicpm` 冒烟开始？
3. 精度回归容忍度（MXFP4 比 NVFP4 激进）？

### 2. W4-FP8 1 天 spike（脚手架已写，待 fcloud）
完成：
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu) —— 最小手写 W4 → FP8 反量化链 + 内联 PTX `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`。grid 维度 × K_LOOP_ITERS 让每次启动产生约 M·N·K FMA。用 cudaEvents 跑 100 iter，打印 TFLOPS + 与 281 TF FP8 上限的判定（GREEN/YELLOW/RED）。
- [spike_w4fp8/run_w4fp8_spike.sh](../spike_w4fp8/run_w4fp8_spike.sh) —— 用 `nvcc -arch=sm_120 -O3 -std=c++17` 编译，跑 4 种形状（default、small、decode、medium-square）。
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.{en,zh}.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md) —— 结果模板，含判定阈值与决策树。

**spike 脚手架注意事项**（结果模板中已记录）：
- 合成数据 —— 仅测吞吐，不验证正确性。
- 无 TMA/swizzle/流水 —— 测得 TFLOPS 是调优 kernel 可达性能的**下界**。
- 每 warp 单 tile 内层循环匹配总 FMA —— 合理代价代理，但不模拟 L2 压力。
- 反量化链是**最贵**版本（bf16 mul + bf16→FP8 cvt）。调优实现可再缩 10–20%。

### 等待用户
- **批准上传 spike .cu + .sh 到 fcloud 并运行**（约 1 fcloud 小时，无生产影响）。
- **决定 NVFP4 KV P2 启动条件**（上述 3 个问题）。

## 结果
- 所有 P1 交付物已离线就绪（无需 fcloud）。
- W4-FP8 spike 代码已可上传（尚未启动 fcloud——按成本节约规则，任何 fcloud 操作前会请示用户）。
- 两项均推进了完整一步而未消耗 fcloud 时间。

## 交叉引用
- [SURVEY_NVFP4_KV_P1_20260428_1130.zh.md](../SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_template.zh.md)
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu)
- [spike_w4fp8/run_w4fp8_spike.sh](../spike_w4fp8/run_w4fp8_spike.sh)
- [PROPOSAL_NVFP4_KV_CACHE_20260427_1730.zh.md](../PROPOSAL_NVFP4_KV_CACHE_20260427_1730.zh.md)
- [PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md](../PROPOSAL_W4_FP8_CUTLASS_SPIKE_20260427_1730.zh.md)
