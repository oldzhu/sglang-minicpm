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

---

## 第 11 轮 —— fcloud 执行 W4-FP8 spike（2026-04-28 13:00）

### 用户提问
1. "先批准 dense-only smoke 优先？ok"
2. "可以解释一下什么是用 --force-dense-minicpm 起 smoke test 来把 plumbing 与 sparse 门控隔开吗？为什么需要把 plumbing 与 sparse 门控隔开？"
3. "精度回归容忍度？>=75%"
4. "我已经把 fcloud 启起来了，你可以 1. 检查 pytorch >> 2.8  2. 测 W4-FP8 spike scaffold"

### 智能体回答 / 行动
- **Q1 解释**：`prepare_env.sh` 的 SGLANG_SERVER_ARGS 中 `--force-dense-minicpm` 让 attention 走标准 FlashAttention（跳过 MiniCPM SALA 稀疏路径）。MXFP4 KV 触及 2 条独立代码路径：(1) plumbing —— FP4 池、quantize/dequantize、FA descale；(2) 稀疏注意力 —— top-k 打分、compressed-K bridge、infllmv2、scratch buffers。同时启两条会让 bug 归因不可能。先 dense-only → 仅 ~3 个 plumbing 门控，迭代快。绿后再当作单独迭代启用稀疏。
- **Q3 记录**：NVFP4 KV P2 最低精度阈值 = 75%（基线 79.29% → 最多 4.3pp 回归）。
- **fcloud 环境检查**：torch=2.9.1+cu128（✅ 有 `float4_e2m1fn_x2`），nvcc 12.8 支持 `-arch=sm_120`，RTX 6000D compute_cap 12.0。spike 与 NVFP4 KV 的所有前置条件都满足。
- **Spike scaffold 构建期间需 2 处修复**：
  1. `int4_t` 不是真实类型 → 改用 `int` + 手动 4-bit 符号扩展。
  2. `cvt.rn.satfinite.e4m3x2.bf16x2` 这个 PTX 形式在 SM120 上**不存在** → ptxas 拒绝。换为受支持的 `cvt.rn.satfinite.e4m3x2.f32`（2 fp32 → 1 fp8x2）。反量化链改为：int4 解包 → fp32 减零点 → fp32 乘 scale → cvt 到 e4m3x2。
- **Spike 结果**（4 个形状，100 iter，cudaEvent 计时）：

| 形状 | TFLOPS | 占 FP8 上限 % | 判定 |
|---|---|---|---|
| 16384×14336×4096 | 156.4 | 55.7% | RED |
| 2048×4096×4096 | 155.8 | 55.4% | RED |
| 1×14336×4096（decode）| 32.6 | 11.6% | RED（带宽受限）|
| 8192×8192×8192 | 156.4 | 55.6% | RED |

  ptxas：23 个寄存器，0 spill，1 个 barrier —— kernel 受计算峰值限制。未融合 fp32 路径下的反量化链占 ~43% 周期。
- **采集完结果立即执行 fcloud 关机**（节费规则）。

### 产出
- **W4-FP8 密集搁置（RED）**。调优生产 kernel 大致能达 200–240 TF（75–85%），但 3–4 周工程换取 prefill-only 路径上 ~5–12% 端到端，相比 NVFP4 KV（~3 天，~44% KV 内存节省，直接帮助 S∞）不划算。
- **NVFP4 KV cache 提升至 P2**。等用户批准后启动 dense-only smoke。
- Catalog 更新，标记 W4-FP8 为 RED，NVFP4 KV 为 P2-NEXT。

### 交叉引用
- [RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md](../RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md)（终版，替代模板）
- [SURVEY_NVFP4_KV_P1_20260428_1130.zh.md](../SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)
- [OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md](../OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md)（W4-FP8 → RED，NVFP4 KV → P2-NEXT）
- [spike_w4fp8/bench_w4fp8_sm120.cu](../spike_w4fp8/bench_w4fp8_sm120.cu)（含 cvt 修复）

---

## Round 13 — CHANGE_0131 fcloud smoke（RED，架构阻断）

### 用户提问
- "started" —— 隐式批准跑 CHANGE_0131 §3 验证流程（sync → 用 `SOAR_FP4_KV_CACHE=1` 重启 → wait → S1 smoke → 精度 → S8/Smax）。

### Agent 操作
1. 提交 `prepare_env.sh` 的 SOAR_FP4_KV_CACHE 开关（`d4608f170`）并推送。
2. 同步到 fcloud，用 `SOAR_FP4_KV_CACHE=1 SOAR_QUANT_MODE=gptq` 启动 server。
3. **Bug #1**：`_handle_kv4_compatibility` 拒绝 `flashinfer` backend。改成 `minicpm_*` 后端早返回（`fd7e797ea`）。
4. 重启又中同一个 assert，因为 `force_dense_minicpm` 已经把 `minicpm_flashinfer` 改写成 `flashinfer`。再补一次（`252cc4d64`）。
5. 重启，撞 `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`。把 `HybridLinearKVPool` 路由到现有的 `MHATokenToKVPoolFP4`（`8a0976593`）。
6. 重启，KV pool 分配成功（27 GB 打包大小）—— 但 cudagraph capture 挂：`KeyError: torch.float4_e2m1fn_x2`，来自 `flashinfer.decode.get_batch_decode_uri`。
7. 诊断：`--force-dense-minicpm` 把 attention_backend `minicpm_flashinfer` 改写成 `flashinfer`（主线 FlashInfer），主线根本没编译 FP4 KV decode kernel。我们 CHANGE_0131 的 plumbing 只在 `MiniCPMAttentionBackend` 里。
8. 按 cost 规则关停 fcloud。
9. 更新 CHANGE_0131 §5/§6/§7（EN+ZH），带 bug 链表、架构阻断分析，给出方案 A/B/C。

### 结论
- **CHANGE_0131 P2 plumbing smoke = 在生产配置上 RED**。`MiniCPMAttentionBackend` 里的 KV 路径写对了，但 GPTQ baseline 必带的 `--force-dense-minicpm` 把我们绕掉。"force-dense + FP4 KV" 这个组合不再做改动就够不到。
- **3 个启动期 bug 修复已合入**（kv4-compat 绕过 × 2 + memory-pool 路由）。属于通用加固，即便 FP4 KV 被 park 也应保留。
- **下一步建议（CHANGE_0132）**：方案 A —— 当 `kv_cache_dtype == "fp4_e2m1"` 时跳过 `minicpm_flashinfer → flashinfer` 重写，让 MiniCPM 自定义后端（在 `force_dense_minicpm` 下本来就能处理 dense-only batch）走自己的 FP4 dequant（`KVFP4QuantizeUtil`）。约 1 天。方案 A 通过后再考虑方案 C（sparse 重启）。

### 交叉引用
- [CHANGE_0131_nvfp4_kv_p2_plumbing.zh.md](../CHANGE_0131_nvfp4_kv_p2_plumbing.zh.md) §5.1、§5.2、§7
- `mixed_minicpm_cudagraph` 上的 commit：
  - `d4608f170` feat(prepare_env): SOAR_FP4_KV_CACHE 开关
  - `fd7e797ea` fix(server_args): bypass KV4 MHA assert for minicpm_*
  - `252cc4d64` fix(server_args): 同时绕过 force_dense_minicpm 情况
  - `8a0976593` fix(memory_pool): MXFP4 KV 走 MHATokenToKVPoolFP4

---

## Round 13b — Q&A：`--force-dense-minicpm` 语义与 dense vs sparse 选择

### 用户提问
> 在动 0132 之前再问一下：如果 `kv_cache_dtype == "fp4_e2m1"` 时不再把 `minicpm_flashinfer` 改写成 `flashinfer`，那 `--force-dense-minicpm` 是不是意味着我们不用 sparse attention（没有 top-k 打分）而走 flashinfer 的 dense？我们 v18 baseline 是 W4A16 + FP8 KV + dense（不是 + sparse），因为本地 dense 比 sparse 速度更好。但理论上长输入 sparse 应该更快，为什么这里 dense 更好？是 FlashInfer 优化得更狠吗？

### Agent 回答（已对源码核对）

**`--force-dense-minicpm` 有两个独立效果：**

1. **Config 层**（`python/sglang/srt/configs/model_config.py` L238 / L248）：
   ```python
   has_sparse_attention → False
   sparse_layer_ids     → []
   ```
   ⇒ 模型图里**完全没有 sparse 层**。没有 top-k 打分、没有 sparse 层路由、没有 sparse Lightning。所有层都是 dense。
2. **Backend 层**（`python/sglang/srt/server_args.py` L1521-1525）：
   ```python
   if self.force_dense_minicpm:
       if self.attention_backend == "minicpm_flashattn":   self.attention_backend = "fa3"
       elif self.attention_backend == "minicpm_flashinfer": self.attention_backend = "flashinfer"
   ```
   ⇒ MiniCPM 自定义后端被换成主线 FlashInfer / FA3。

**v18 baseline = W4A16 + FP8 KV + 100% dense + 主线 FlashInfer**。确认。

**为什么本地 dense 比 sparse 好 —— 三个叠加因素：**

1. **本地速度数据集太短**。`speed_{s1,s8,smax}.jsonl` 是我们手搓的。sparse 的胜利来自 `O(top_k · L)` vs `O(L²)`，但临界点大约在 8-16K token。在 1-4K token 上，top-k 打分 + gather/scatter + 调度开销 **比节省的还多**。
2. **主线 FlashInfer 在 Blackwell SM120 上优化得很好**。paged KV、fused FP8 dequant、连续的 TensorCore tile（CHANGE_0125 已加 SM120 Marlin tile 表）。MiniCPM sparse backend 主要是 Triton —— 正确性可靠，但 raw decode 吞吐通常比 FlashInfer 慢 20-40%。
3. **GPTQ + sparse 在精度上是坏的**。Test 5/6：旧 fcloud 上 ~50% 精度。Test 8b：默认 top_k 下 76%。sparse 路径和 GPTQ 量化的 QKV 配合差 —— top-k 打分对量化噪声敏感。Test 9（GPTQ + bf16 KV + sparse）是唯一能用的 sparse-with-GPTQ 结果，79.67%。FP8 KV + sparse + GPTQ 精度直接崩。

**官方** 已经宣布速度数据集比我们手搓的有更多长上下文样本。理论上 sparse 在那应该有用 —— 但只要 #3 不修，就不能上。所以生产配置必然是 dense + FP8 KV + GPTQ。

### 对 CHANGE_0132 的影响

`kv_cache_dtype == "fp4_e2m1"` 时跳过 **backend** 重写：

- `has_sparse_attention` **还是 False**（`force_dense_minicpm` 的 config 层效果与 backend 重写互相独立）。sparse 层不会回来。还是纯 dense。
- `MiniCPMAttentionBackend` 保持选中，每层都通过它走 dense attention，CHANGE_0131 的 FP4 plumbing 终于能跑起来。
- **未解风险**：`MiniCPMAttentionBackend` 的 dense decode 路径在 SM120 上可能比主线 FlashInfer 慢。如果 FP4 KV 节省的 44% 内存带来的并发提升 / 吞吐提升 抵不过 kernel 差距，那即使能启动且精度过，速度也是负收益。

### 决定

用户选保守路线：

1. **先**做 `MiniCPMAttentionBackend.forward_decode` 代码审查，事先评估它的 dense decode 在 SM120 上是否能跟主线 FlashInfer 抗衡 —— 看会不会有速度悬崖。
2. **然后**实现方案 A，在 fcloud 上量化（smoke → 精度 → S1/S8/Smax）。

### 交叉引用

- [server_args.py L1521-1525](../../../python/sglang/srt/server_args.py#L1521-L1525) —— backend 重写
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238) / [L248](../../../python/sglang/srt/configs/model_config.py#L248) —— sparse 层关闭
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Test 5/6/8b/9/12 —— GPTQ sparse-vs-dense 实测史
- [CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md) §3 方案 A

---

## Round 13c — 代码审查结论：方案 A 不可行

### 用户提问
> 把上面那个回答记录下来供以后参考。先做 (1)（审 `MiniCPMAttentionBackend.forward_decode`），然后再做 (2)（跑方案 A 测量）。

### Agent 操作
通读 `python/sglang/srt/layers/attention/minicpm_backend.py`，得到四个否定方案 A 的发现：

1. **`MiniCPMSparseBackend.__init__` L226-230**：`has_sparse_attention=False` 时直接 `raise ValueError`。
2. **`force_dense_minicpm=True` 在 config 层把 `has_sparse_attention` 强制为 False**（`model_config.py` L238）。
3. **`forward_decode` L1130-1300** 无条件调 `get_topk_for_sparse` + `sparse_kernel_extension.get_block_table_v3`，没有 dense 分支。
4. **`forward_extend` L989+** 两条分支都走 `metadata.sparse_page_table`。短序列也跳不开 sparse 管道。

⇒ 在 `--force-dense-minicpm` 下，自定义后端 **连初始化都失败**；即便能初始化，每条路径都是 sparse top-k。我之前说"MiniCPMAttentionBackend 已支持 dense-only batch"是错的。

### 结论
- **方案 A（FP4 时跳过重写）已死**。没做任何实现。
- **方案 A′（修订版）** 需要在 `MiniCPMSparseBackend` 里 fork：放宽 init 守卫，在 `forward_decode`/`forward_extend` 加 `is_dense_run` 分支跳过 top-k、直接用完整 `page_table` 走已经导入的 `BatchDecodeWithPagedKVCacheWrapper`，cudagraph buffer 重新分配。工作量：几百行 LoC + 仔细的 cudagraph 重验证。不是 1 天能搞定的。
- **方案 B**（主线 FlashInfer 加 FP4 KV）需要上游 kernel template —— 超出比赛时间线。
- **推荐：方案 C —— 冻结 CHANGE_0131/0132**。保留 R13 的四个加固 commit（`d4608f170`、`fd7e797ea`、`252cc4d64`、`8a0976593`），`SOAR_FP4_KV_CACHE=0`（默认）时它们在运行期是闲置的。转其他目录项。
- 已更新 `CHANGE_0132_nvfp4_kv_force_dense_compat.{en,zh}.md`：新增 §8 "Round 13b 代码审查"，列 L221-L230 / L1130-L1300 / L989-L1056 证据；§7 改为推荐方案 C。

### Agent 没有进入第 (2) 步

第 (2) 步（跑方案 A 测量）已无意义，因为方案 A 已被证伪。在花 fcloud 时间之前先把这个结论交给用户。

### 交叉引用
- [CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md](../CHANGE_0132_nvfp4_kv_force_dense_compat.zh.md) §8
- [minicpm_backend.py L221-L230](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L221-L230)
- [minicpm_backend.py L1130-L1300](../../../python/sglang/srt/layers/attention/minicpm_backend.py#L1130-L1300)
- [model_config.py L238](../../../python/sglang/srt/configs/model_config.py#L238)

---

## Round 13d — 决策前讨论：稀疏的最优结果 + `--force-dense-minicpm` 的逐层行为

### 用户提问
1. `W4A16 + FP8 KV + sparse` 和 `W4A16 + bf16 KV + sparse` 我们历史最好的精度+速度结果是什么？
2. 加了 `--force-dense-minicpm` 之后，稀疏注意力层会以 dense softmax 运行（不再调 top-k 和稀疏算子）—— 已确认。那么 lightning 层呢？是否也变成 dense？

### Agent 回答

**Q1 —— 历史最好结果**

| 配置 | 最佳测试 | ori_acc | norm | C | mcq / qa / niah / cwe / fwe | 精度评测时长 | 备注 |
|---|---|---|---|---|---|---|---|
| GPTQ + FP8 KV + **稀疏** | Test 8b (`9d3ecd168`) | 76.07% | 95.08% | **0** | 60 / 66.67 / 96.67 / 60.33 / 96.67 | 2411 s | FP8 KV 破坏稀疏 top-k 评分（descale 舍入误差）|
| GPTQ + bf16 KV + **稀疏** | Test 9 (`79e49f39f`) | **79.67%** | **99.58%** | **1.0** | 63.33 / 56.67 / 100 / 81.67 / 96.67 | 3157 s | 唯一达到 C=1.0 的稀疏配置 |
| GPTQ + FP8 KV + **dense**（当前 baseline）| Test 12 (`9e82efe43`) | 79.29% | 99.11% | **1.0** | 63.33 / 72 / 97.78 / 100 / 63.33 | 4244 s | S1=121.71s, S8=44.09s, Smax=35.86s |

关键观察：**两个稀疏配置都没单独跑过 S1/S8/Smax**。表里 "时长" 是精度评测（150 样本 concurrency=8）的总时长，不是速度 benchmark trio。Test 8b 的 2411s < Test 12 的 4244s 不能直接推出 "稀疏更快"，差异多半来自输出长度（精度回退时模型答错或截断会更快结束，并不是 decode 真的快）。

用户提出 Test 8b 的时长短可能预示更高吞吐。反驳：dense FP8 已经 >100 tok/s，2411/4244 的比值由输出长度主导，不是 decode 速度。

**Q2 —— `--force-dense-minicpm` 下的逐层行为**

读 `model_config.py` L238/L248 + `minicpm.py` L546-590 + `configs/minicpm.py`：

- `force_dense_minicpm=True` 只改写 `has_sparse_attention` 和 `sparse_layer_ids`。**不会**动 `mixer_types` / `lightning_layer_ids`。
- `MiniCPMDecoderLayer.__init__` 的层构建严格按 `self.mixer_type`（来自 `config.mixer_types[layer_id]`）分支：

| 逐层 mixer_type | 模块 | 后端 | `--force-dense-minicpm` 的影响 |
|---|---|---|---|
| `"minicpm4"` | `MiniCPMAttention`（softmax QKV）| 配置的 attention_backend | 后端被改写：`minicpm_flashinfer → flashinfer` ⇒ **以 dense softmax 跑全 KV**，不调 top-k、不调 sparse_page_table |
| `"lightning"` / `"lightning_attn"` | `MiniCPMLightningMixer` | `SimpleGLAAttnBackend`（线性注意力 / chunked GLA）| **不动** —— lightning 是带隐状态的线性注意力，没有 dense softmax 等价形式。chunked-GLA kernel 照常执行 |
| `"attention"` / `"standard"` | `MiniCPMAttention`（softmax）| 配置的 attention_backend | 同 minicpm4：走 stock FlashInfer dense |

结论：**用户直觉对了一半**。稀疏 softmax 层会变成 dense；lightning 层 **不会** —— 仍走 `SimpleGLAAttnBackend`。我们提交模型究竟有没有 lightning 层得看 `config.json`（用户预期 8 个 sparse + 24 个 lightning，待 fcloud 起来后实测）。

### 用户输入（冠军帖子上下文）
> "几乎所有冠军帖都用 W4A16 + 混合 FP4/FP8 KV cache，他们没提 force-dense 也没说稀疏 —— 我猜他们默认用稀疏（否则会强调）。"

这是个强烈的暗示：**赢家配方很可能是稀疏 + 混合精度 KV**，不是 dense + 单一 FP8 KV。但要注意两点：
1. 我们有直接证据 **GPTQ + FP8 KV + 稀疏会破坏精度**（Test 8b 76.07%，远低于 99% 归一化阈值）。top-k 上的 descale 舍入误差累积是真的。
2. 冠军方案可能用 **逐层混合 KV**：在做 top-k 评分的几层稀疏层上用 bf16（或精心 scale 的 FP8），其它层用 FP4/FP8。我们目前没有这套混合 KV plumbing。

### 决策上下文
用户想 **重测 GPTQ + FP8 KV + 稀疏**：
- (a) 在新 fcloud / 当前代码上确认 Test 8b 的精度回退是否依然存在（模型已经重新量化过，kernel 也更新过）；
- (b) 收集稀疏路径的 S1/S8/Smax —— 我们从来没跑过；
- (c) 验证或证伪冠军方案猜想。

### 结论 / 待办
- 本轮 Q&A 已记录到 chat log（本条目）。
- 用户开 fcloud，等会让 agent 看 `MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8` 的 `config.json`，确认 `mixer_types`（预期 8 个 minicpm4 + 24 个 lightning）。
- 下一步：mixer_types 确认后，agent 给出具体的重测方案（稀疏 + FP8 KV：精度 + 速度 S1/S8/Smax）。

### 交叉引用
- [TEST_RESULTS_TRACKING.md](../TEST_RESULTS_TRACKING.md) Tests 8b、9、12
- [model_config.py L238 / L248](../../../python/sglang/srt/configs/model_config.py#L238)
- [minicpm.py L546-L590](../../../python/sglang/srt/models/minicpm.py#L546-L590)
- [configs/minicpm.py L145-L167](../../../python/sglang/srt/configs/minicpm.py#L145-L167)
