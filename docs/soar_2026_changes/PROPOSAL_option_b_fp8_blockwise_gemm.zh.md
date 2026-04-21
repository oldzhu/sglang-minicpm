# 提案：选项B — 通过现有 SM120 内核实现 FP8 Blockwise GEMM

**日期**: 2026-04-21  
**状态**: 等待批准  
**优先级**: 高 — 利用现有 SM120 内核实现 30-50% 加速的直接路径  
**基准**: S1=121.71s, S8=44.09s, Smax=35.86s（测试12）  
**预期结果**: S1 ~95-110s, S8 ~34-40s, Smax ~28-33s（估计）

---

## 执行摘要

sgl-kernel 已经有一个完全可用的 SM120 FP8 blockwise GEMM（`fp8_blockwise_scaled_mm` with `sm120_fp8_blockwise_dispatch_shape`）。唯一缺失的是将权重从 GPTQ W4 格式转换为 FP8 blockwise 格式，以及模型级别的分发以使用新内核。

**无需编写 CUDA 内核。** 实现完全是 Python + PyTorch。

---

## 问题陈述

当前 Marlin GPTQ W4 内核：
- 使用 SM80 `mma.sync.aligned.m16n8k16` 指令
- 有效算力约 ~100-140 TFLOPS（BF16 路径）
- SM120 FP8 硬件（296 TFLOPS）在 GEMM 期间利用不足

选项B目标后：
- 通过 `fp8_blockwise_scaled_mm` 使用 SM120 UMMA → ~200-260 TFLOPS FP8（估计）
- 保持归一化精度 > 99%，C=1.0
- 无新 CUDA 依赖

---

## 规则合规性

| 约束 | 状态 |
|------|------|
| ≤2GB 提交包 | ✅ 无新 wheel（内核已在 sgl-kernel 中） |
| 现场量化 | ✅ FP8 转换在 preprocess_model.py 运行（<<5h） |
| 正确性 C ≥ 0.96 | ✅ FP8 8位 vs W4 4位 → 应保持 >99% 精度 |
| 无禁止技巧 | ✅ 纯量化格式更改 |
| 可复现 | ✅ 确定性转换 |

---

## 架构概述

### 当前流程
```
GPTQ W4 权重（int32 打包，group_size=128）
         ↓
Marlin gptq_marlin.cu（SM80 mma.sync）
         ↓
BF16 输出
```

### 选项B后
```
FP8 E4M3 权重（K×N，列优先）+ scales_b（K/128 × N/128）
         ↓
激活量化：BF16 → FP8 E4M3 + scales_a（M × K/128）
         ↓
fp8_blockwise_scaled_mm（SM120 UMMA tcgen05.mma.ws.sync）
         ↓
BF16 输出
```

### 权重转换（在 preprocess_model.py 中一次性完成）
```
GPTQ W4 模型（group_size=128）
         ↓
每层：Marlin 反量化 W4 → FP16 块（形状 K×N）
         ↓
每个 128×128 块：找 max_abs → FP8_scale = max_abs / 448.0
         ↓
FP8_weights = clamp(FP16_block / FP8_scale, -448, 448).to(float8_e4m3fn)
         ↓
保存 FP8 权重（K×N，列优先）+ scales（K/128 × N/128）为 safetensors
         ↓
更新 config.json，加入 quantization_config: {"quant_type": "fp8_blockwise", "block_size": 128}
```

---

## 需要修改的具体文件

### 1. `preprocess_model.py` — 添加 FP8 blockwise 转换步骤

添加新模式 `fp8_blockwise`（或扩展 `gptq` 模式加 `--fp8-postprocess` 标志）。

**从 FP16 模型直接量化**（推荐，无需校准，精度最好）：
```python
def run_fp8_blockwise_quantization(src: Path, dst: Path):
    """从 FP16 模型直接量化为 FP8 blockwise 格式（SM120 UMMA）。"""
    from safetensors.torch import load_file, save_file
    import torch
    
    config = load_json(src / "config.json")
    
    for shard_file in sorted(src.glob("*.safetensors")):
        tensors = load_file(shard_file)
        new_tensors = {}
        
        for key, tensor in tensors.items():
            if tensor.dtype in (torch.float16, torch.bfloat16) and \
               tensor.ndim == 2 and \
               tensor.shape[0] % 128 == 0 and tensor.shape[1] % 128 == 0:
                # 线性层权重 → 量化为 FP8 blockwise
                w_fp8_col, scale_b = quantize_fp8_blockwise(tensor, block_size=128)
                new_tensors[key] = w_fp8_col          # (N, K) float8_e4m3fn 列优先
                new_tensors[key + "_scale"] = scale_b  # (N/128, K/128) float32
            else:
                new_tensors[key] = tensor  # 传递非线性层张量
        
        save_file(new_tensors, dst / shard_file.name)
    
    # 更新 config.json
    config["quantization_config"] = {
        "quant_type": "fp8_blockwise",
        "block_size": 128,
        "fp8_dtype": "float8_e4m3fn"
    }
    save_json(config, dst / "config.json")
```

**辅助函数 `quantize_fp8_blockwise(w_fp16, block_size=128)`**：
```python
def quantize_fp8_blockwise(w: torch.Tensor, block_size: int = 128):
    """w: (K, N) float16/bfloat16 → (N, K) float8_e4m3fn 列优先 + (N/128, K/128) float32 scales"""
    K, N = w.shape
    FP8_MAX = 448.0
    
    # 以 (K/128, 128, N/128, 128) 形状计算每块最大值
    w_blocks = w.reshape(K // block_size, block_size, N // block_size, block_size)
    max_abs = w_blocks.abs().amax(dim=(1, 3))  # (K/128, N/128)
    scales = (max_abs / FP8_MAX).clamp(min=1e-12)  # (K/128, N/128) float32
    
    # 量化
    scale_expanded = scales.unsqueeze(1).unsqueeze(3).expand_as(w_blocks)
    w_scaled = (w_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    w_fp8 = w_scaled.reshape(K, N).to(torch.float8_e4m3fn)
    
    # 转为列优先（内核要求 mat_b.stride(0) == 1）
    w_fp8_col = w_fp8.t().contiguous()  # (N, K) C-contiguous = (K, N) 列优先
    
    # scales 格式：内核要求 (K_dim/128, N_dim/128)
    # K_dim = mat_b.size(0) = N（转置后），N_dim = mat_b.size(1) = K（转置后）
    scale_b = scales.t().contiguous()  # (N/128, K/128) — 对应内核的 (K_dim/128, N_dim/128)
    
    return w_fp8_col, scale_b
```

### 2. 新文件：`python/sglang/srt/layers/quantization/fp8_blockwise.py`

新量化方法类：
```python
class FP8BlockwiseConfig(QuantizationConfig):
    """FP8 blockwise 量化配置（SM120 UMMA）。"""
    quant_type = "fp8_blockwise"
    block_size = 128
    
    def get_quant_method(self, layer, prefix=""):
        if isinstance(layer, LinearBase):
            return FP8BlockwiseLinearMethod(self)
        return None

class FP8BlockwiseLinearMethod(LinearMethodBase):
    """FP8 blockwise 线性层，使用 SM120 UMMA（fp8_blockwise_scaled_mm）。"""
    
    def create_weights(self, layer, ...):
        # 注册 weight_fp8 和 weight_fp8_scale 为层参数
        layer.weight_fp8 = nn.Parameter(...)      # (N, K) float8_e4m3fn，列优先
        layer.weight_fp8_scale = nn.Parameter(...)  # (N/128, K/128) float32
    
    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        # 激活量化：每行每 128 个 K 元素
        scales_a, x_fp8 = quantize_activation_fp8_blockwise(x)  # (M, K/128), (M, K) fp8
        
        # 调用 SM120 FP8 GEMM
        from sgl_kernel import fp8_blockwise_scaled_mm
        out = fp8_blockwise_scaled_mm(
            x_fp8,                      # (M, K) fp8，行优先
            layer.weight_fp8,           # (N, K) fp8，列优先（内核视为 K_dim×N_dim 列优先）
            scales_a,                   # (M, K/128) float32
            layer.weight_fp8_scale,     # (N/128, K/128) float32
            out_dtype=torch.bfloat16
        )
        if bias is not None:
            out = out + bias
        return out
```

**激活量化（每行每块）**：
```python
def quantize_activation_fp8_blockwise(x: torch.Tensor):
    """x: (M, K) bfloat16 → x_fp8: (M, K) float8_e4m3fn + scales_a: (M, K/128) float32"""
    M, K = x.shape
    assert K % 128 == 0
    FP8_MAX = 448.0
    
    x_blocks = x.reshape(M, K // 128, 128)  # (M, K/128, 128)
    max_abs = x_blocks.abs().amax(dim=2)  # (M, K/128)
    scales_a = (max_abs / FP8_MAX).clamp(min=1e-12)  # (M, K/128)
    
    scale_expanded = scales_a.unsqueeze(2)  # (M, K/128, 1)
    x_scaled = (x_blocks / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    
    return scales_a, x_fp8
```

### 3. `python/sglang/srt/layers/quantization/__init__.py` — 注册新方法

将 `FP8BlockwiseConfig` 添加到量化注册表。

### 4. `python/sglang/srt/models/minicpm.py` — 条件 FP8 分发

当检测到 `quantization_config.quant_type == "fp8_blockwise"` 时：
- 将 `FP8BlockwiseConfig` 作为 quant config 传递给线性层
- 所有线性层（MLP gate/up/down, QKV, O-proj）使用 `FP8BlockwiseLinearMethod`
- Lightning 层（`SimpleGLAAttnBackend`）不使用线性层 → 无需更改

### 5. `benchmark/soar/demo_sala/preprocess_model.py` — 扩展模式

添加 `fp8_blockwise` 到模式选项：
```python
choices=["copy", "gptq", "fp8_blockwise"],
```

新执行路径：
```python
elif mode == "fp8_blockwise":
    run_fp8_blockwise_quantization(src, dst)
    print(f"[preprocess] mode={mode} done - FP8 blockwise model saved to {dst}")
    return
```

---

## 实现顺序

### 第1天：权重转换脚本

1. 在 `preprocess_model.py` 中编写 `run_fp8_blockwise_quantization(src, dst)`
2. 使用 `transformers.AutoModel.from_pretrained(src, torch_dtype=torch.float16)` 加载 FP16
3. 迭代所有线性层，调用 `quantize_fp8_blockwise(layer.weight, block_size=128)`
4. 使用 `safetensors.torch.save_file` 保存，自定义命名约定
5. 本地测试单层

**测试**：`python preprocess_model.py --input /root/models/.../MiniCPM-SALA-Copy --output /tmp/minicpm_fp8 --mode fp8_blockwise`
- 应在 <30 分钟内完成（无校准，只是计算）
- 检查保存的文件大小：FP8 = 1字节/参数 vs FP16 = 2字节/参数 → 模型大小减半

**注意**：模型权重不在 2GB 提交包中，只有代码 wheels 和脚本。FP8 权重大小（约9GB vs W4的4.5GB）不影响提交大小，只影响推理显存（84GB 完全够用）和量化时间（<30分钟 vs GPTQ 几小时）。

### 第2天：量化方法类

1. 创建 `python/sglang/srt/layers/quantization/fp8_blockwise.py`
2. 实现 `FP8BlockwiseConfig` 和 `FP8BlockwiseLinearMethod`
3. 在 `__init__.py` 中注册
4. 本地单元测试：单线性层前向传播

### 第3天：模型集成

1. 修改 `minicpm.py` 检测 `fp8_blockwise` quantization_config
2. 向线性层传递正确的 quant config
3. 处理权重加载（新参数名：`weight_fp8`，`weight_fp8_scale`）
4. 在本地用 FP8 模型启动服务器
5. 验证服务器无错误启动

### 第4天：精度验证

1. 在 fcloud 上运行 `eval_model_001.py`（FP8 模型）
2. 检查归一化精度：目标 >99%（C=1.0）
3. 如果精度 <99%：研究哪些层需要 W4（例如保持 QKV 为 GPTQ，只有 MLP 用 FP8）

### 第5天：速度基准测试

1. 运行 `fcloud_workflow.py speed --variant all`
2. 与基准对比 S1/S8/Smax（测试12）
3. 用 Nsight 性能分析：检查实际 FP8 TFLOPS 利用率

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| FP8 纯量化精度 <99% | 中 | 高 | 回退：QKV 保持 GPTQ，只有 MLP 用 FP8 |
| 权重转换错误（块布局错误） | 低 | 高 | 单元测试：FP8 反量化回 FP16，与原始比较 |
| Scale 格式与内核不匹配 | 中 | 高 | 跟踪内核的 TORCH_CHECK 信息；先用小矩阵测试 |
| Decode 退化（2× 权重带宽） | 中 | 中 | 单独分析 S1；如果 decode 主导，回退或只对 MLP 用 FP8 |
| FP8 decode 比 W4 decode 慢 | 高 | 低-中 | 预期情况；S1 分数取决于基准中 prefill/decode 的比例 |

---

## 回退计划

FP8 通过 `config.json` 的 quantization_config 在模型加载时选择。回退方法：
1. 使用原始 GPTQ 模型目录（仍然保留）
2. 无需代码回退 — 内核分发通过 quantization_config 类型判断
3. 用原始模型路径重启服务器

---

## 验证命令

```bash
# 在 fcloud 上：

# 第1步：将 FP16 模型转换为 FP8 blockwise
python3 /root/submission_sim/preprocess_model.py \
    --input /root/models/openbmb/MiniCPM-SALA-Copy \
    --output /root/models/minicpm_fp8_blockwise \
    --mode fp8_blockwise

# 第2步：检查权重格式
python3 -c "
from safetensors.torch import load_file
t = load_file('/root/models/minicpm_fp8_blockwise/model-00001-of-XXXX.safetensors')
for k,v in list(t.items())[:10]:
    print(k, v.shape, v.dtype)
"

# 第3步：用 FP8 模型启动服务器
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
    --model-path /root/models/minicpm_fp8_blockwise \
    --host $HOST --port $PORT "${SGLANG_SERVER_ARGS[@]}"

# 第4步：精度验证
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 第5步：速度测试
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

---

## 预期结果

| 指标 | 基准（测试12） | 选项B后 | 目标 |
|------|--------------|---------|------|
| S1 时长 | 121.71s | ~95-110s | <100s |
| S8 时长 | 44.09s | ~34-40s | <36s |
| Smax 时 长 | 35.86s | ~28-33s | <30s |
| 归一化精度 | 99.11% | >99% | >99% |
| 正确性系数 C | 1.0 | 1.0 | 1.0 |

**注**：这些是基于 TFLOPS 利用率提升的估计。2× 权重内存导致的实际 decode 影响可能会部分抵消 prefill 收益，特别是对于 decode 主导的 S1。

---

## 关键实现注意事项

**内核的 scale 格式（重要）**：

```
内核检查：
  TORCH_CHECK(mat_a.size(0) == scales_a.size(0), ...)   → scales_a行数 = M
  TORCH_CHECK(mat_a.size(1) / 128 == scales_a.size(1), ...) → scales_a列数 = K/128
  TORCH_CHECK(mat_b.size(0) / 128 == scales_b.size(0), ...) → scales_b行数 = mat_b.size(0)/128
  TORCH_CHECK(mat_b.size(1) / 128 == scales_b.size(1), ...) → scales_b列数 = mat_b.size(1)/128

mat_b 为列优先，shape (N, K)：
  mat_b.size(0) = N（N 维度是行），mat_b.size(1) = K（K 维度是列）
  所以 scales_b shape = (N/128, K/128)

权重矩阵逻辑上是 (K_in, N_out)，存储为 (N_out, K_in) 行优先（C-contiguous）即列优先视图
  → scales_b shape = (N_out/128, K_in/128)
```

**内核列优先要求**：
```python
# 验证
assert w_fp8_col.stride(0) == 1, "B matrix must be col-major (stride[0]=1)"
```

---

## 后续步骤

选项B成功后，如果仍有差距：
1. **FP8 KV cache + FP8 attention**：KV cache 已有 FP8，但 attention GEMM 也可以用 FP8
2. **选择性精度**：MLP 用 FP8（矩阵更大，更受计算限制），QKV 用 W4（更小，decode 主导）
3. **选项C**：W4A8 融合反量化内核（保持 W4 内存 + 获得 FP8 计算）
4. **Prefill/decode 分割**：不同阶段用不同精度

---

## 参考实现文件

- 现有 SM120 FP8 内核：`sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`
- Python 绑定：`sgl-kernel/python/sgl_kernel/gemm.py`（`fp8_blockwise_scaled_mm`）
- 现有 FP8 量化工具：`python/sglang/srt/layers/quantization/fp8_utils.py`
