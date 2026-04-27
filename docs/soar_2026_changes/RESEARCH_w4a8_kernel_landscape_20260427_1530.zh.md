# 研究笔记：W4A8 kernel 全景、Phase 0 微基准计划、W4A16 vs W4A8 成本分析（2026-04-27 15:30）

本文整合 2026-04-27 下午会话的问答，便于复盘与追踪。配套：`RESEARCH_w4a8_kernel_landscape_20260427_1530.en.md`。

---

## 1. 最新冠军方案（情报记录）

参考微信文章 https://mp.weixin.qq.com/s/w1g3njB24rxLCiCLxWFD7Q（最新一周冠军综述）：

- **量化**：W4A16 GPTQ（与我们 v18 基线同族）
- **KV 缓存**：混合 **NVFP4 + FP8** KV
- **其他**：未公开

对路线图的含义：冠军**当前并未使用 W4A8**。"真正的 W4A8" 在该硬件/赛事上仍是未验证路径。我们更近期的杠杆可能是 **NVFP4 KV 缓存**（已有 `ANALYSIS_nvfp4_offline_quant_*` 原型文档），而非 W4A8 kernel 工作。把 W4A8 当作研究下注，不是稳赢方案。

---

## 2. 现存 W4A8 kernel 清单（基于代码搜索核验）

### 2.1 `sgl-kernel/` 内已 vendor、构建可用

| Kernel | 文件 | 权重 | 激活 | MMA | Arch 守卫 | 用途 |
|---|---|---|---|---|---|---|
| `qserve_w4a8_per_group_gemm` | `csrc/gemm/qserve_w4a8_per_group_gemm.cu` | INT4 packed，group=128 | **INT8** 对称 | I8 IMMA 内联 PTX | `__CUDA_ARCH__ ≥ 800`（SM80+） | **稠密 GEMM**，MIT QServe |
| `qserve_w4a8_per_chn_gemm` | `csrc/gemm/qserve_w4a8_per_chn_gemm.cu` | INT4，per-channel | INT8 | I8 IMMA | SM80+ | 稠密 GEMM |
| `cutlass_w4a8_moe_mm` | `csrc/moe/cutlass_moe/w4a8/*` | INT4 packed | **FP8 e4m3** | FP8 QMMA via Hopper TMA | **仅 SM90**（`is_hopper()` 门控） | **仅 MoE 分组 GEMM**，非稠密 |

### 2.2 vllm `csrc/quantization/machete/`

- README 明确：`compute_type = a.dtype`，其中 `a` 为 BF16/FP16。**Machete 是 W4A16，不是 W4A8。**
- 是 Marlin 的 Hopper 后继，覆盖与我们 SM120 Marlin tile（CHANGE_0125）相同的问题域。
- 一些下游 fork 加了 FP8 激活，但**未进 vllm 主线**。

### 2.3 SM120 稠密 W4A8 kernel 可用性结论

| 路径 | 现有 kernel？ | SM120 可用？ | 集成工作量 |
|---|---|---|---|
| **稠密 W4-INT8** | 有 — QServe `qserve_w4a8_per_group_gemm`（已在 sgl-kernel 中） | 是 — SM80+ 内联 PTX 路径 | **低**：`gptq.py` 中 Python 接线 + 权重重打包 + 每 token INT8 量化器 |
| **稠密 W4-FP8** | **无**（sgl-kernel 仅 SM90 MoE；vllm Machete 是 W4A16） | 不适用 | **高**：从零写或移植；可能数周 |
| **自定义 CUTLASS 混合输入** | 从零开始 | 视情况 | 最高 |

这**翻转了之前的优先级**。更新建议：

- **SM120 稠密 W4-FP8 = 研究项目**。叠加第 1 节（无冠军在做），选项 A 作为首轮迭代不具吸引力。
- **稠密 W4-INT8（QServe）= 务实首选**，前提是 Phase 0 微基准显示 SM120 INT8 IMMA 吞吐与 FP8 持平。

---

## 3. Phase 0 微基准：SM120 INT8 IMMA vs FP8 QMMA vs BF16

### 3.1 我们要测什么

官方 `SM120_RTX_PRO_HARDWARE.md` 列出 FP8=296 TF、BF16=148 TF、FP4=593 TF，**未列 INT8**。需确认 SM120 INT8 IMMA 是跑在 FP8 档位（约 296 TF）还是被压到 BF16 档位（约 148 TF）。这决定 QServe W4-INT8 是否值得集成。

### 3.2 仅 PyTorch 的简易微基准（无需构建 cutlass_profiler）

PyTorch 已封装了相关张量核心路径：
- `torch.matmul(bf16, bf16)` → BF16 张量核心
- `torch._scaled_mm(fp8, fp8, ...)` → **FP8 QMMA**（你问的 FP8 简易测试）
- `torch._int_mm(int8, int8)` → INT8 IMMA

因此一个 **fcloud 上约 5 分钟的脚本**即可。无需 cutlass 构建、无需 cuBLAS 调用。

### 3.3 脚本（写到 `/root/bench_int8_vs_fp8_sm120.py`）

```python
import torch, time
torch.manual_seed(0)
DEV = "cuda"

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

# MiniCPM 代表性形状：qkv_proj/o_proj 隐藏维 4096；gate_up/down MLP=14336
shapes = [
    (4096, 4096, 4096),
    (4096, 14336, 4096),
    (4096, 4096, 14336),
]
for M, N, K in shapes:
    a_bf16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    a_fp8  = a_bf16.to(torch.float8_e4m3fn)
    b_fp8  = b_bf16.to(torch.float8_e4m3fn)
    a_i8   = (a_bf16 * 100).clamp(-128, 127).to(torch.int8)
    b_i8   = (b_bf16 * 100).clamp(-128, 127).to(torch.int8)
    sa = torch.tensor(1.0, device=DEV); sb = torch.tensor(1.0, device=DEV)

    t_bf16 = bench(lambda: torch.matmul(a_bf16, b_bf16))
    t_fp8  = bench(lambda: torch._scaled_mm(
        a_fp8, b_fp8.t().contiguous().t(),
        scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    try:
        t_i8 = bench(lambda: torch._int_mm(a_i8, b_i8))
        i8_str = f"INT8={t_i8*1e3:.3f}ms ({2*M*N*K/t_i8/1e12:.1f} TFLOPS)"
    except Exception as e:
        i8_str = f"INT8=N/A ({e})"

    flops = 2 * M * N * K
    print(f"[{M}x{N}x{K}]  "
          f"BF16={t_bf16*1e3:.3f}ms ({flops/t_bf16/1e12:.1f} TF)  "
          f"FP8={t_fp8*1e3:.3f}ms ({flops/t_fp8/1e12:.1f} TF)  "
          f"{i8_str}")
```

### 3.4 运行命令（fcloud 启动后）

```bash
# 在 python3 scripts/fcloud/fcloud_workflow.py setup 之后
python3 /root/bench_int8_vs_fp8_sm120.py 2>&1 | tee /root/phase0_int8_vs_fp8_sm120.log
# 把日志拉回本地用于追踪
```

结果会写入**新建的后续文档 `PHASE0_INT8_vs_FP8_SM120_<timestamp>.md`**（不并入本研究笔记），原始测量与研究叙事分离。

### 3.5 决策规则

| 结果 | 行动 |
|---|---|
| INT8 ≥ ~250 TF（≈ FP8 档位） | 放行 QServe W4-INT8 集成作为下一轮迭代 |
| INT8 ≈ ~148 TF（BF16 档位） | INT8 在 SM120 上无算力收益；放弃 W4-INT8 |
| FP8 < ~250 TF | 与官方规格冲突；先排查再做任何 8 位路径 |

---

## 4. W4A16（当前 Marlin）vs W4A8 — 不只是 TFLOPS

### 4.1 W4A16 已有的优势（当前基线）

Marlin W4A16 已经具备：
- **2× 权重带宽** vs FP8 权重存储（INT4=0.5 B/elem，FP8=1 B/elem；vs BF16 是 4×）。decode 阶段是**权重带宽受限**，这是主导杠杆，Marlin 已经吃到。
- BF16 MMA 在 SM120 上 148 TF → 远超 decode 实际持续速率（decode 是带宽限，不是算力限）。
- BF16 激活精度 → 精度损失极小。

### 4.2 W4A8 *可能* 在 W4A16 之上的增益

**每层微观层面**，激活从 BF16 切到 8 位会改变三件事：

| 项目 | W4A16（Marlin BF16 激活） | W4A8（INT8 或 FP8 激活） | Delta |
|---|---|---|---|
| 权重存储 | INT4（0.5 B/elem） | INT4（0.5 B/elem） | **0**（相同） |
| 激活内存流量 | BF16（2 B/elem） | INT8 / FP8（1 B/elem） | **−50% 激活带宽** |
| MMA 峰值算力 | BF16 = 148 TF | FP8 = 296 TF（INT8 若持平） | **+100% 峰值算力** |
| 输出 / 累加器 | BF16 | BF16 | 0 |
| 每 token 开销 | 无 | 每 token 激活量化（一次乘 + cast） | 小幅**新增** |

### 4.3 按工作负载的定量预期

Decode（S1）和 prefill（S8/Smax）对张量核心的负载方式不同。估计：

#### Decode bs=1（S1，基线 121.71s）
- **权重带宽受限**，不是激活或算力。
- 激活带宽下降：极小（激活只是单 token，O(hidden_size) 字节，相对 O(hidden×hidden) 权重字节微不足道）。
- 峰值算力翻倍：无关（MMA 不是瓶颈）。
- **现实 S1 收益：≤ 5%**，若激活量化新增延迟可能为负。
- 风险：每 token 量化在每层每 token 增加一次小 kernel 启动 + 一次内存遍历。bs=1 每个周期都要算账。

#### Prefill / 大 M（S8 约 44s，Smax 约 36s）
- **算力与共享内存受限**，不是纯权重带宽。
- 激活带宽下降：中等（激活随序列长度增长，长上下文时变重要）。
- 峰值算力翻倍：在这里有用；张量核心更接近峰值。
- **现实 S8/Smax 收益：若 kernel 达到 FP8/INT8 峰值的 ~70%，约 10–20%。**
- 风险：完全取决于 kernel 调优程度。QServe 是为 SM80 调的，SM120 可能要重调。

### 4.4 诚实的收益预期表

| 档位 | 基线 | 乐观 W4A8 | 悲观 W4A8 | 最可能 |
|---|---|---|---|---|
| S1 | 121.71s | ~115s（−5%） | ~125s（+3%） | ≈ 基线 ± 3% |
| S8 | 44.09s | ~36s（−18%） | ~42s（−5%） | ~40s（−9%） |
| Smax | 35.86s | ~29s（−19%） | ~34s（−5%） | ~32s（−11%） |

这些都是 *理论估计，假设 kernel 在 SM120 上正确工作且归一化精度保持 > 99%*。实际数字可能更差（kernel 未调优）。

### 4.5 战略结论

- W4A8 主要是**prefill / 长上下文**优化，不是 decode 优化。
- 官方速度集据称比我们本地集长上下文样本更多，所以 prefill 优化即使本地 Smax 不动，官方也应有收益。
- 但：冠军用 W4A16 + NVFP4 KV 已经证明**不靠 W4A8 也能竞争。** NVFP4 KV 缓存在该硬件上很可能是更高杠杆的优化。

---

## 5. 更新后的开放问题与建议序列

1. **Phase 0 微基准（廉价，fcloud 约 5 分钟）** — 任何 kernel 工作之前先跑。**等待你启动 fcloud。**
2. 若 Phase 0 INT8 通过：**选项 B = QServe W4-INT8 集成**（低工作量，kernel 已在仓库）。
3. 若 Phase 0 INT8 不过 *且* 团队有 kernel 研究带宽：搁置 W4A8，**优先 NVFP4 KV 缓存**（匹配冠军组合）。
4. 选项 A（自定义 W4-FP8 稠密 kernel）仅当 (2) 与 (3) 都耗尽时。

---

## 6. 追踪

- 创建：2026-04-27 15:30
- 作者：agent
- 配套 EN：`RESEARCH_w4a8_kernel_landscape_20260427_1530.en.md`
- 触发下一文档：`PHASE0_INT8_vs_FP8_SM120_<timestamp>.md`（fcloud 微基准跑完后创建）
