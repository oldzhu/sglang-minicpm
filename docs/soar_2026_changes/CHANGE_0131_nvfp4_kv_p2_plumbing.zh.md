# CHANGE 0131 —— NVFP4 KV cache P2 plumbing（dense-only）

**日期**：2026-04-28
**状态**：代码改动已提交；等待 fcloud smoke 测试。
**前序**：[SURVEY_NVFP4_KV_P1_20260428_1130.zh.md](SURVEY_NVFP4_KV_P1_20260428_1130.zh.md)
**分支**：`minicpm-src` 上的 `mixed_minicpm_cudagraph`

## 1. 背景与动机

P1 调研发现 ~80% 的 MXFP4 KV-cache plumbing 已在上游 sglang 中（server arg、dtype 解析、`MHATokenToKVPoolFP4`、`KVFP4QuantizeUtil`）。剩余缺口完全集中在我们自定义的 `python/sglang/srt/layers/attention/minicpm_backend.py` 内。P2 关闭这些必修门控，让 `--kv-cache-dtype fp4_e2m1 --force-dense-minicpm` smoke 测试能起来。

为何 MXFP4 KV 值得做：

- **相比 FP8 节省 ~44% KV 内存**（MiniCPM-SALA 形状下每 token 每 layer 1152 vs 2048 字节）。
- 直接帮 S∞：长上下文下并发更大。
- ~3 天 plumbing vs 3–4 周调优 W4-FP8 kernel —— ROI 远好于 [W4-FP8 spike RED 判定](RESULT_W4_FP8_CUTLASS_SPIKE_20260428_1300.zh.md)。

## 2. 规则合规声明

- **守 SOAR 规则**：KV-cache 压缩属推理时优化，无需重训，权重量化不变（仍是 GPTQ W4A16）。
- **守基线配置不变量**：先 dense 模式（`--force-dense-minicpm`）smoke；FP8 路径仍是生产回退。不删 FP8 分支。
- **精度门**：归一化精度 ≥ 75%（用户在 R11 设定的容忍度）。当前基线 79.29%，最大可接受回退 ~4.3pp。
- **不动 eval 脚本** —— 纯服务端 plumbing。

## 3. 详细实现计划（改动前）

### `python/sglang/srt/layers/attention/minicpm_backend.py` 中需修复的门控

| # | 位置 | 当前逻辑 | `fp4_e2m1` 下问题 | 修复 |
|---|---|---|---|---|
| A | L834 sparse-bridge | `if self.kv_cache_dtype_str.startswith("fp8")` 把 query/compressed_k cast 到 bf16 | FP4 路径同样可能返回非 bf16；dense smoke 完全跳过此分支（无 sparse），但未来重启 sparse 需要它 | 把门改为 "任意压缩 KV dtype" |
| B | L189 `use_fp8_sparse_scratch` | `kv_cache_dtype_str.startswith("fp8")` | 对 FP4 为 False → sparse scratch 保持 BF16。dense smoke 不走 sparse 路径；P2 安全 | **P2 不改**（重启 sparse 时再看） |
| C | L932, L1144 `set_kv_buffer` | 传 `layer.k_scale, layer.v_scale`（FP8 per-tensor scale） | `MHATokenToKVPoolFP4.set_kv_buffer` 在 MXFP4 quant 之前 `cache_k.div_(k_scale)` —— 当存储 dtype 是 FP4 时会错 scale K | 当 `kv_cache_dtype_str == "fp4_e2m1"` 时传 `None, None` |
| D | L952, L1173 `k_descale` 构建 | `kv_cache_dtype_str != "auto"` 时构建 `k_descale = layer.k_scale.expand(...)` | FP4 pool 的 `_get_key_buffer` 已返回 dequant 后的 BF16；再把 k_descale 喂给 FA 会双重 scale | 当 `kv_cache_dtype_str == "fp4_e2m1"` 时跳过整块（保持 `None, None`） |

### 涉及文件

- **`python/sglang/srt/layers/attention/minicpm_backend.py`** —— 唯一一个有源码改动的文件。

### 不动的文件（已核实安全）

- `prepare_env.sh` —— server arg 已支持 `fp4_e2m1`；用户在本地编辑 `SGLANG_SERVER_ARGS` 切换。
- `memory_pool.py` —— `MHATokenToKVPoolFP4` 上游原样。
- `kvfp4_tensor.py` —— `KVFP4QuantizeUtil` 上游原样。
- `model_runner_kv_cache_mixin.py` —— dtype 匹配自动路由到 FP4 pool。
- Eval 脚本 —— 永不动（按 copilot instructions）。

### 验证命令

agent 给放行 + 用户显式启动 fcloud 后，在 fcloud 上跑：

```bash
# 1. 编辑 prepare_env.sh：把 SGLANG_SERVER_ARGS 加上
#    --kv-cache-dtype fp4_e2m1 --force-dense-minicpm
# 2. 同步并重启服务
python3 scripts/fcloud/fcloud_workflow.py sync
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
# 3. 快 smoke（小并发）
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
# 4. smoke 过 → 跑精度
python3 scripts/fcloud/fcloud_workflow.py accuracy
# 5. 精度 ≥ 75% → 跑 S8/Smax
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax
```

### 成功 / 失败标准

- **Smoke 过**：server 用 `fp4_e2m1` 起来，能完成至少一个请求，无异常无垃圾。
- **精度过**：归一化 ≥ 75%（由于归一化是相对最佳玩家，绝对约 ≥73%）。
- **速度过（目标）**：S1/S8/Smax 相比 FP8 基线不退超过 5%；理想情况下 Smax 因 44% 内存节省带来更大并发而提升。
- **预期可能的失败模式**：
  - cudagraph capture 与 `KVFP4QuantizeUtil` 内的 `@torch.compile` 不兼容 → 退化到 eager 或重写为 Triton（P2.5）。
  - 提早需要重启 sparse → CHANGE_0132 处理（Gap A + Gap B）。

## 4. 实际代码改动（改动后）

所有编辑均在 `python/sglang/srt/layers/attention/minicpm_backend.py`。未动其它源文件。forward_extend 和 forward_decode 两条路径均覆盖。

### Edit 1 (L188-193) — 在 `__init__` 里加 `self.use_fp4_kv_cache` 标记

```python
self.kv_cache_dtype = model_runner.kv_cache_dtype
self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
self.use_fp8_sparse_scratch = self.kv_cache_dtype_str.startswith("fp8")
# MXFP4 KV cache flag (kv_cache_dtype = fp4_e2m1)。用于禁用 FP8 风格的 per-tensor
# scaling (k_scale/v_scale)，那样会与 MXFP4 的 per-16-element block scaling 冲突。
self.use_fp4_kv_cache = self.kv_cache_dtype_str == "fp4_e2m1"
```

### Edit 2 (L839) — Gap A：Sparse 桥接门含 FP4

```python
# 原本：if self.kv_cache_dtype_str.startswith("fp8"):
if self.kv_cache_dtype_str.startswith("fp8") or self.use_fp4_kv_cache:
    # 把 query / compressed_k / compressed_k2 cast 到 bf16
```

### Edit 3 (L940-941, L1160-1161) — Gap C：FP4 时传 None,None 给 set_kv_buffer

在两个调用 `set_kv_buffer` 的地方（forward_extend 和 forward_decode）都修了：

```python
k_scale = None if self.use_fp4_kv_cache else layer.k_scale
v_scale = None if self.use_fp4_kv_cache else layer.v_scale
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, k, v, k_scale, v_scale
)
```

为何：`MHATokenToKVPoolFP4.set_kv_buffer` 在 MXFP4 量化**之前**做 `cache_k.div_(k_scale)`。传 FP8 per-tensor scale 会让 K/V 数据被错误 scale（scale 语义不兼容）。

### Edit 4 (L964-965, L1194-1195) — Gap D：FP4 时跳过 k_descale 构造

两个点均适用：

```python
if (
    self.kv_cache_dtype_str != "auto"
    and not self.use_fp4_kv_cache       # 新增
    and layer.head_dim <= 256
    # ...
):
    if layer.k_scale is not None:
        # 构造 k_descale, v_descale
```

为何：`MHATokenToKVPoolFP4._get_key_buffer/_get_value_buffer` 已经返回 dequant 后的 BF16（通过 `KVFP4QuantizeUtil.batched_dequantize`）。再传个非 None 的 k_descale 给 FlashAttention 会双重 scale。

### 有意不改的点（及原因）

- **L189 `use_fp8_sparse_scratch`** — 保持原样。启 `--force-dense-minicpm` 后，sparse scorer 路径不被调用，所以 `scratch_only=False`（退化 BF16）对 smoke 测试是安全的。重启 sparse（CHANGE_0132）时再改。
- **`memory_pool.py` / `kvfp4_tensor.py`** — 上游原样；Survey 确认它们已经正确处理 MXFP4 布局。
- **`prepare_env.sh`** — 用户本地切换 `--kv-cache-dtype fp4_e2m1 --force-dense-minicpm` 跑 smoke。
- **Eval 脚本** — 按项目规则从不改。

### 已做的自活检查

- `python3 -c "import ast; ast.parse(...)"` → AST parse OK
- `get_errors` → 无 compile/lint 错误
- 四个修补点都 grep 验证引用了 `self.use_fp4_kv_cache`
- 审计 `kv_cache_dtype_str.startswith("fp8")` 引用：仅 L189（`use_fp8_sparse_scratch`，有意不改）与 L839（现为 `or self.use_fp4_kv_cache`）。

## 5. 结果汇总表

Smoke 测试（Round 13）—— RED。基线配置（`--force-dense-minicpm` +
`--kv-cache-dtype fp4_e2m1`）下 server 无法启动。

| 变体 | 基线（FP8 KV）| 新（FP4 KV）| Δ |
|---|---|---|---|
| S1 (s) | 121.71 | n/a — 启动被阻断 | — |
| S8 (s) | 44.09 | n/a — 启动被阻断 | — |
| Smax (s) | 35.86 | n/a — 启动被阻断 | — |
| ori_accuracy | 79.29% | n/a — 启动被阻断 | — |

### 5.1 已发现并修复的启动期 bug 链（已提交）

| # | 现象 | 根因 | 修复 commit |
|---|---|---|---|
| 1 | `AssertionError: KV4 MHA expects attention_backend ['triton','torch_native','flex_attention','trtllm_mha'], got flashinfer` | 主线 `_handle_kv4_compatibility()` 白名单没考虑 MiniCPM 自定义后端（`minicpm_flashinfer`），也没考虑 `force_dense_minicpm` 把它重写成 `flashinfer` 后的情况。 | `fd7e797ea`（minicpm 前缀绕过）+ `252cc4d64`（force_dense_minicpm 绕过）|
| 2 | `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`，来自 `torch.zeros(..., dtype=fp4_e2m1fn_x2)` | `HybridLinearKVPool` 永远使用 `MHATokenToKVPool`，忽略了文件里已经存在的 `MHATokenToKVPoolFP4`（uint8 打包 K/V + e8m0 共享指数缓冲）。 | `8a0976593`（fp4 路由到 FP4 池）|
| 3 | `KeyError: torch.float4_e2m1fn_x2`，来自 `flashinfer.decode.get_batch_decode_uri` | **架构性阻断。** `--force-dense-minicpm` 在 `_handle_model_specific_adjustments` 里把 `attention_backend = "minicpm_flashinfer"` 改写成 `"flashinfer"`。主线 FlashInfer 没有 FP4 KV 支持。 | **未修复** —— 见 §7 |

### 5.2 架构阻断（bug #3）

CHANGE_0131 的全部 plumbing 都在 `MiniCPMAttentionBackend`
（`python/sglang/srt/layers/attention/minicpm_backend.py`）。但传入
`--force-dense-minicpm` 时，`_handle_model_specific_adjustments`
会无条件把 `minicpm_flashinfer` 重写为 `flashinfer`，绕过我们的后端，
落到主线 FlashInfer 的 `BatchDecode` —— 它根本没编译 FP4 KV 的 decode
kernel。

当前生产提交 **始终** 带 `--force-dense-minicpm`（GPTQ baseline 的
`SGLANG_SERVER_ARGS` 写死），所以不进一步改造就根本碰不到 FP4 代码路径。

## 6. 回滚说明

```bash
git revert <commit-hash>
git push minicpm-src mixed_minicpm_cudagraph
```

或者把环境变量 `SOAR_FP4_KV_CACHE` 设为 0（默认）—— opt-in 开关保留 FP8
基线路径。三个 bug 修复 commit（kv4-compat 绕过 + memory-pool 路由）属于
通用加固，可以保留。

## 7. 下一步建议

启动期修复 commit（`fd7e797ea`、`252cc4d64`、`8a0976593`）是正确的，
应保留 —— 它们对未来任何 MiniCPM + FP4 工作都是通用加固。

要在生产配置上真正跑通 FP4 KV，**CHANGE_0132** 必须在以下中选一个：

- **方案 A（首选 —— 小改动）：** 当 `kv_cache_dtype == "fp4_e2m1"` 时，
  跳过 `minicpm_flashinfer` → `flashinfer` 的重写。MiniCPM 自定义后端
  在 `--force-dense-minicpm` 下本来就能处理 dense-only batch，我们只
  要继续用它（而不是主线 flashinfer），就能命中
  `minicpm_backend.py` 里的 FP4 plumbing。然后重跑 §3 验证流程。
- **方案 B（更重 —— 数周工作量）：** 给主线 `flashinfer_backend` 加
  FP4 KV 支持（需要 flashinfer 端的
  `dtype_kv = float4_e2m1fn_x2` kernel template）。比赛时间线内不可行。
- **方案 C（在方案 A 通过后开 sparse）：** CHANGE_0133 重启 sparse
  路径下的 FP4（CHANGE_0131 §3 中的 Gap A + B）。

建议下一轮直接走方案 A。
