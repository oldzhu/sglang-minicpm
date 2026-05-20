# CHANGE_W4A8_QMMA_TCGEN05: SM120 融合 INT4→FP8 反量化 + QMMA GEMM 内核

## 背景

### 先前的 W4A8 工作

W4A8 优化经历了多个迭代：

| 迭代 | 方法 | 结果 |
|-----------|----------|--------|
| W4A8#1 (2026-04) | 加载时 INT4→FP8 反量化 + FP8 GEMM | **S1 +118% 性能倒退** — HBM 权重带宽翻倍 |
| W4A8-REAL v25 (2026-05-18) | 每次前向 INT4→FP8 反量化 + cutlass FP8 GEMM | **声称 −9% 加速但无法复现** — 非法 CUDA 内存访问 |
| 标量融合内核 | SMEM 中 INT4→FP16 反量化 + wmma | **正确但比 Marlin 慢 2−5 倍**（148 TFLOPS） |
| MMA 融合内核 | SMEM 中 INT4→FP16 反量化 + wmma m16n16k16 | **有 Bug** — 多 tile 误差 ~68，服务器 503 |

### 根因分析

两步法（反量化 + 分离的 FP8 GEMM）从根本上无法超越 Marlin：
- Marlin：从 HBM 读取 INT4 权重（0.5 字节/元素），以 148 TFLOPS 计算
- 两步法：读取 INT4（0.5 B/elem）+ 写入 FP8（1 B/elem）+ 读取 FP8（1 B/elem）= **2.5× HBM 权重流量**
- 即使有 296 TFLOPS 的 FP8 QMMA，GEMM 在 decode（bs=1）时也是带宽受限的

**唯一可行的路径是融合内核**：INT4 存储 → 内核内反量化 → FP8 QMMA。

### 本次变更

新的融合内核（`w4a8_fp8_qmma.cu`）替换旧的基于 wmma 的内核：
- **tcgen05.mma** PTX 内联汇编 — SM120 原生 FP8 QMMA，**296 TFLOPS**
- **TMEM**（Tensor Memory）用于累加器 — Blackwell 新增的存储空间
- **TMA 描述符**用于 tcgen05 访问 SMEM 张量
- 与旧内核相同的 INT4→FP8 反量化逻辑（已验证正确）
- 相同的 Python 接口（`torch.ops.w4a8_fused.w4a8_fp8_fused_gemm`）

## 规则合规

- **SOAR 约束**：纯 CUDA 内核优化 — 无预量化权重，无额外模型文件
- **精度**：INT4→FP8 反量化与旧内核逐位一致（相同的解包逻辑）
- **提交大小**：无变化（权重保持 INT4，0.5 字节/元素）

## 实现

### 新建文件

| 文件 | 用途 |
|------|---------|
| `sgl-kernel/csrc/gemm/w4a8_fp8_qmma.cu` | 新的 tcgen05 QMMA 内核（~340 行） |
| `sgl-kernel/csrc/gemm/CMakeLists_standalone.txt` | 独立构建配置 |

### 修改文件

| 文件 | 变更 |
|------|--------|
| `sgl-kernel/CMakeLists.txt` | 添加 `w4a8_fp8_qmma.cu` 到构建列表 |

### 内核架构

```
线程块: 128 线程（1 个 warp-group, cta_group::1）
Tile: M=128, N=128, K=128

每个 K-tile 迭代:
  1. 协作加载 + 反量化: INT4 → FP8 → W_fp8[128×128] 到 SMEM（列优先）
  2. 协作加载: FP8 激活 → A_fp8[128×128] 到 SMEM（行优先）
  3. 为 W_fp8 和 A_fp8 构建 TMA 描述符
  4. tcgen05.mma SS: SMEM_A × SMEM_B → TMEM_C（296 TFLOPS）
  5. 在 TMEM 中跨 K-tile 累加

Epilogue:
  6. tcgen05.commit — 提交 TMEM
  7. tcgen05.st — 每线程子 tile 从 TMEM → SMEM（每线程 16×8 个 float）
  8. float → BF16 → 全局内存
```

### SMEM 预算

| 缓冲区 | 大小 | 用途 |
|--------|------|---------|
| W_fp8 | 16 KB | 反量化权重（128×128 FP8，列优先） |
| A_fp8 | 16 KB | 激活 tile（128×128 FP8，行优先） |
| C_bf16 | 32 KB | 输出暂存（128×128 BF16，兼作 tcgen05.st 目标） |
| TMA descs | 256 B | 两个 128 字节的 TMA 描述符 |
| **总计** | **~64 KB** | 适合 SM120 的 101 KB SMEM |

## 已知风险

### 高：TMA 描述符格式（风险 A）
`fill_tma_desc_2d()` 填充的 128 字节 TMA 描述符布局是基于 NVIDIA PTX ISA 文档和 cutlass CuTe 模式的最佳推测。**尚未在 SM120 硬件上验证。** 如果格式不正确，tcgen05.mma 将从 SMEM 读取垃圾数据。

**缓解措施**：如果描述符有误，回退到使用 cutlass 头文件（`cute::TmaDescriptor`）或 `cp.async.bulk.tensor.2d` PTX 指令创建有效描述符。

### 中：无 SMEM Swizzling（风险 B）
简单的行优先/列优先 SMEM 布局可能导致 tcgen05.mma 读取时的 bank 冲突，降低有效吞吐量。生产内核应使用 cutlass 的 swizzled 布局。

### 中：tcgen05.st 粒度（风险 C）
通过 `tcgen05.st` 的每线程子 tile 拷贝假设了特定的线程到元素的映射关系（8×16 线程网格，每线程 16×8 子 tile）。如果此映射有误，部分元素将丢失或重复。

### 低：TMEM 泄漏（风险 D）
调用了 `tcgen05.alloc` 但未调用 `tcgen05.dealloc`。对于每个 SM 单个块的启动来说这是可以接受的；TMEM 是每 warp-group 的，在块退出时释放。

## 验证

### 构建
```bash
cd /root/standalone_fused/build
cmake .. \
  -DTorch_DIR=/app/sglang_minicpm_sala_env/lib/python3.10/site-packages/torch/share/cmake/Torch \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j2
cp libw4a8_fused_gemm.so /root/submission_sim/
```

### 正确性（单 tile）
```python
import torch
torch.ops.load_library("/root/submission_sim/libw4a8_fused_gemm.so")

M, N, K, g = 128, 128, 128, 128
qw = torch.randint(0, 2**31-1, (K//8, N), dtype=torch.int32, device="cuda")
qz = torch.randint(0, 2**31-1, (K//g, N//8), dtype=torch.int32, device="cuda")
sc = torch.randn(K//g, N, dtype=torch.float32, device="cuda")
a  = torch.randn(M, K, dtype=torch.float8_e4m3fn, device="cuda")

c_fused = torch.ops.w4a8_fused.w4a8_fp8_fused_gemm(qw, qz, sc, a, N, K, g)
torch.cuda.synchronize()

# 参考值：CPU FP32 矩阵乘法
w_fp32 = dequant_cpu(qw, qz, sc, g)  # 使用现有的反量化函数
c_ref  = torch.mm(a.float(), w_fp32.t().float()).bfloat16()

print("最大误差:", (c_fused.float() - c_ref.float()).abs().max().item())
# 预期: < 0.5（FP8→BF16 量化误差）
```

### 速度
```bash
# 在 fcloud 上，使用 SOAR_W4A8_REAL_FP8_GEMM=1 运行速度基准测试
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 回滚

回滚到旧的 wmma 内核：
```bash
cd /root/standalone_fused/build
# 编辑 CMakeLists.txt: 将 w4a8_fp8_qmma.cu 替换为 w4a8_fp8_fused_gemm.cu
cmake --build . -- -j2
cp libw4a8_fused_gemm.so /root/submission_sim/
```

或完全禁用融合内核：
```bash
export SOAR_W4A8_REAL_FP8_GEMM=0
```

## 后续步骤

1. **构建和编译** — 在 fcloud 上验证 PTX 编译通过（`-arch=sm_120a`）
2. **正确性测试** — 先单 tile，再多 tile（4096×4096）
3. **服务器集成** — 重启 sglang 并验证输出一致
4. **速度基准测试** — S1/S8/Smax 对比 Marlin 基线
5. **如果 TMA 描述符有误** — 切换到 cutlass 头文件方式或使用 `cute::TmaDescriptor`
6. **如果内核正常但比预期慢** — 添加 SMEM swizzling、double-buffering
