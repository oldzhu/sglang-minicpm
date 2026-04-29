# CHANGE_0135（续 001）— Round 13e Option-B 性能剖析结果

本文档是 `CHANGE_0135_sparse_path_cleanup_and_profile_plan.zh.md`（提案文档）的配套结果文档,记录实际剖析数据,并对提案中的决策树给出结论。

## 1. 剖析对象

- 分支 / commit: `mixed_minicpm_cudagraph` @ `f4097eef6` (CHANGE_0135 清理之后)。
- 服务器配置 (来自 `prepare_env.sh` `noquant` 分支):
  - `--trust-remote-code --disable-radix-cache`
  - `--attention-backend minicpm_flashinfer`
  - `--chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1`
  - `--max-running-requests 8 --mem-fraction-static 0.78`
  - `--schedule-conservativeness 1.0`
  - `--kv-cache-dtype fp8_e5m2 --enable-mixed-chunk`
  - **不带** `--quantization`、`--force-dense-minicpm`、`--dense-as-sparse`。
- 模型: `/root/models/openbmb/MiniCPM-SALA` (BF16, 加载约 18 GB)。
- 路由: 8 层 sparse_attention + 24 层 lightning-attn;每请求阈值 `dense_len = sparse_dense_len`(默认 512)。三个样本 prompt 长度都远高于 512,因此 8 层 sparse 全部走 **稀疏 top-k + paged-KV FlashInfer** 分支。
- KV cache: torch.float8_e5m2 (预留 12 073 962 个 token,约 46 GB)。
- Cudagraph 捕获 `bs ∈ {1, 2, 4, 8}`。
- Workload: 通过 `/generate` 单请求,`max_new_tokens=64`,从 `/root/data/perf_public_set.jsonl` 中按 `prompt_tokens` 选择最接近目标长度的样本。
- 工具: `torch.profiler`,通过 `/start_profile`、`/stop_profile` 启停。

| 跑次   | 样本 idx | prompt_tokens | wall   | GPU kernel 总时间 |
|--------|----------|---------------|--------|--------------------|
| 32k    | 129      | 31 744        |  9.7 s | 5 658 ms          |
| 64k    |  49      | 63 683        | 24.7 s | 10 344 ms         |
| 128k   | 149      | 127 732       | 49.3 s | 19 775 ms         |

GPU 活跃占比 (kernel 时间 / wall 时间): 32k 时 58% → 64k 时 42% → 128k 时 40%。即使 `--prefill-max-requests 1` 已经把调度成本压到很低,主机端开销仍然不可忽略,且随上下文增长而扩大。

完整分析输出已提交至 `docs/soar_2026_changes/profile_data/round13e_analyze.txt`。

## 2. 内核级别汇总 (Top)

三种长度下,占用第一的内核都是同一个:
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` —— 一个普通的 BF16 cuTLASS GEMM(模型的 dense 线性层 / MLP 投影)。

| 项目                                    | 32k        | 64k        | 128k        |
|-----------------------------------------|------------|------------|-------------|
| BF16 256×128 GEMM (prefill 阶段线性/MLP)| 66.7%      | 73.2%      | **76.9%**   |
| BF16 gemvx (decode 阶段线性/MLP)        | 13.4%+3.0% | 7.3%+1.6%  | 3.8%+0.8%   |
| flashinfer BatchPrefillWithPagedKV (稀疏 FA 主体) | 3.6% | 4.0% | 4.3% |
| act_and_mul (SiLU·gate)                 | 1.5%       | 1.5%       | 1.6%        |
| FillFunctor<BFloat16> (compress_k1/k2 fill,CHANGE_0133 之后) | 1.0% | 1.1% | 1.1% |
| FusedAddRMSNorm                         | 1.1%       | 1.1%       | 1.1%        |
| get_block_table_cuda_v2<96> (稀疏 top-k 元数据) | 0.7% | 0.8% | 0.8% |
| cumsum_kernel (稀疏元数据)              | 0.7%       | 0.7%       | 0.8%        |
| flatten_and_fill (稀疏元数据)           | 0.4%       | 0.4%       | 0.4%        |
| chunk_fwd_kernel_o (lightning-attn)     | 0.5%       | 0.6%       | 0.6%        |
| chunk_fwd_kernel_h (lightning-attn)     | 0.4%       | 0.4%       | 0.4%        |

类别汇总:

| 类别                        | 32k    | 64k    | 128k   |
|-----------------------------|--------|--------|--------|
| GEMM (BF16 cuTLASS)         | 67.2%  | 73.8%  | 77.7%  |
| GEMM (其它 / GEMV)          | 16.4%  | 9.0%   | 4.7%   |
| FLA / SimpleGLA / 稀疏 FA / fused norm | 7.7% | 8.3% | 8.7% |
| 其它 (fill / index / copy / topk)      | 7.7% | 7.9% | 8.1% |
| RMSNorm (QK-norm + RoPE)               | 0.6% | 0.6% | 0.6% |
| Embedding / topk-gather                | 0.4% | 0.4% | 0.3% |

## 3. 数据解读

1. **瓶颈是 BF16 GEMM,不是稀疏注意力。** 稀疏注意力本身的核 (`BatchPrefillWithPagedKVCacheKernel`) 只占 3.6–4.3% GPU 时间。稀疏元数据相关核(`get_block_table` + `cumsum` + `flatten_and_fill` + topk)合计约 2%。整个稀疏机制总计远低于 7%。

2. **为什么 GEMM 这么重?** noquant 构建下,32 层的 QKV / O / MLP 投影全部走 BF16 cuTLASS。SM120 的 BF16 吞吐(约 148 TFLOPS)只有 FP8 (约 296 TFLOPS) 的一半,Marlin INT4 的四分之一。比赛 baseline (Test 12: GPTQ + FP8 KV + dense, S₁=121.7 s) 之所以快,关键就是把这些投影路由到 Marlin/INT4 + FP8 GEMM。一旦因为 GPTQ-sparse 历史精度问题而把线性层退回 BF16,这一份红利就全部失去 —— profile 上"BF16 GEMM 全精度"成为头号开销正是这件事的直接表现。

3. **稀疏 FA 自身扩展性正常。** GEMM 时间 3.8 s → 7.6 s → 15.2 s(≈线性),稀疏 FA 0.20 s → 0.42 s → 0.85 s(同样线性,斜率相近),没有平方级爆炸。SM120 上稀疏 top-k + paged-KV 路径表现良好。

4. **CHANGE_0133 已经够。** compress_k1/k2 fill 在三种长度下都稳定在 1.0–1.1% GPU 时间(每 step 约 1720–1888 次 launch)。bounded-fill 修复之前那种"每 step GB 级"问题已经不存在。

5. **主机开销是第二大项,但不是某条代码路径。** GPU 活跃占比 32k 58% → 128k 40%。128k 时 49 s wall 中 GPU 实际工作约 30 s,剩下的是 launch / scheduler / Python 开销。`--prefill-max-requests 1 --max-running-requests 8` 已经把 CPU 成本压到很低;再做 cudagraph-over-sparse 或者把零碎 bookkeeping 内核(cumsum / fill / index)合并,**理论上限**也只是去掉那 8% "其它"尾巴,而不是 77% 的 GEMM。

## 4. 落实 CHANGE_0135 的决策树

提案中列出 4 条退出分支:

| 分支 | 触发条件                                 | 是否命中 |
|------|-------------------------------------------|----------|
| (a) 稀疏 FA 已经在 >70% 带宽上 memory-bound — 关闭稀疏线 | 稀疏 FA 占主时间 | **否** —— 稀疏 FA ≤4.3% |
| (b) 单一稀疏内核 dominate — 写专门的 SM120 优化         | 单核 >25%        | **否** —— 最高的稀疏内核仅 4.3% |
| (c) CPU launch / scheduler dominate — 修主机循环 / cudagraph | 活跃率 <30% 或 scheduler 占 wall >20% | **部分命中** —— 128k 时活跃率 40%,但优化上限只能再省 8–10% wall |
| (d) BF16 权重才是成本 — 接受 BF16+sparse 永远赢不了 INT4+dense | GEMM >50%      | **是 —— 67–77%** |

数据明确指向**分支 (d)**。这条分支提案没有显式列出,但它正好是提案目标("判断稀疏线是否还有可能赢过 dense+INT4")的隐含答案:

- 让稀疏线在 SOAR 上慢下来的不是稀疏注意力,而是它当前以 **BF16 权重精度**运行这件事本身。
- 要让稀疏线显著变快,必须复刻 dense+INT4 的量化故事 —— 即在稀疏路由下做出 **GPTQ + sparse + FP8 KV** 且精度稳定。
- 之前对 `sparse_qkv_w8` 的 GPTQ-sparse 尝试在公开集上把精度打到了 ~50%(见 `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` 与 Round 11/12 历史结果)。在没有精度修复方案之前,分支 (d) 对提交的唯一诚实结论是:**冻结 BF16+sparse 线,提交保持 dense+GPTQ+FP8 KV baseline**。

## 5. 决策与下一步建议

1. **SOAR 提交(短期):** 维持 dense + GPTQ + FP8 KV baseline (Test 12 配置) 作为提交路径。不再为 BF16+sparse profile 投入更多 kernel 侧成本。要变快的核是通用 BF16 cuTLASS GEMM —— sgl-kernel 之外没有快速的 local win。

2. **稀疏线研究(中期,可选):** 让稀疏线有竞争力的唯一现实路径是把 BF16 权重换成量化权重,**同时让稀疏注意力数值稳定**。两个方向:
   - **GPTQ + sparse + FP8 KV,重新调校 calibration**(8 个稀疏层 QKV 保留 W8A16,其它仍 W4A16;重测公开集精度,看历史上的 50% 崩溃是否可避免)。
   - **稀疏路由下使用 FP8 权重 (W8A8 / mxfp8)。** SM120 有 QMMA (mxfp8) 硬件通路;这能把 76.9% GPU 时间的 BF16 GEMM 替换为 296 TFLOPS 的 FP8 通路。需先验证稀疏注意力中 `compress_k`/top-k 在 FP8 激活下的数值稳定性。

3. **尾部 kernel 清理(低优先级):** 那 8% "其它"尾巴(主要是 launch 开销 + element-wise fill/indexing)是 cudagraph-over-sparse-path 的潜在目标,但只在第 2 项确定要保留稀疏线后再做 —— 在 67–77% GEMM 瓶颈面前,加速 8% 尾巴的性价比太低。

## 6. 验证状态

本次只做了单请求 profile,目的是定位 dominate kernel,不衡量 S₁ / S₈ / S∞ 的真实吞吐。Round 13e 之前在 concurrency=32 上的超时与本次显示的 BF16 GEMM 瓶颈完全自洽:并发上去之后,这些 GEMM-bound 内核会序列化,wall 时间随 batch 大致线性增长。

本轮没有做精度跑。dense+GPTQ submission baseline 的精度 / 速度参考数据请见 `TEST_RESULTS_TRACKING.md` Test 12。

## 7. 回滚

本轮没有需要回滚的改动 —— 只产出了 profile 数据,`--dense-as-sparse` 是 CHANGE_0135 本身已经移除的。fcloud 端临时变通(`ln -sf /usr/bin/gcc /usr/local/bin/clang && ln -sf /usr/bin/g++ /usr/local/bin/clang++`,因 Python 3.10.19 sysconfig 设置了 `CC=clang -pthread` 而 fcloud 镜像里只有 `gcc`)**尚未提交**。如果以后还要在干净 fcloud 镜像上跑 noquant 分支,应把这一步并入 `prepare_env.sh`,或者把预编译好的 pypcre wheel 打进 `submission_sim.tar`。这是后续动作,不在本次提交范围内。

## 8. 文件

- `docs/soar_2026_changes/profile_data/round13e_analyze.txt`(新增)—— 三种长度下原始 Top-25 kernel breakdown。
- `docs/soar_2026_changes/CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.en.md`。
- `docs/soar_2026_changes/CHANGE_0135_sparse_path_cleanup_and_profile_plan_001.zh.md`(本文件)。

Trace 文件 `round13e_{32k,64k,128k}.trace.json.gz`(共约 379 MB)位于 fcloud `/root/profile_round13e/`,未下载到本地;若需 TensorBoard 深入查看,可用 `profile_driver.py` 重现。
