# CHANGE 0132 — NVFP4 KV 与 `--force-dense-minicpm` 兼容

**状态**：**方案 A 不可行**，详见 §8 Round 13b 代码审查
**前置**：CHANGE_0131（P2 plumbing —— Round 13 smoke 因架构阻断 RED）
**分支**：`mixed_minicpm_cudagraph` on `minicpm-src`

## 1. 背景与动机

CHANGE_0131 在 `MiniCPMAttentionBackend`
（`python/sglang/srt/layers/attention/minicpm_backend.py`）中加入 MXFP4
KV 缓存 plumbing，并在 `prepare_env.sh` 加入 `SOAR_FP4_KV_CACHE`
opt-in 开关。Round 13 fcloud smoke 暴露三个 bug，前两个已在 commit
`fd7e797ea`、`252cc4d64`、`8a0976593` 中修复。第三个 bug 属于架构性
问题，是本提案要解决的对象。

当用户传入 `--force-dense-minicpm`（GPTQ 生产基线必带）时，
`server_args.py` 中的 `_handle_model_specific_adjustments` 会无条件把
`attention_backend = "minicpm_flashinfer"` 改写成 `"flashinfer"`。这
直接绕过我们自定义的后端，落到主线 FlashInfer 的 `BatchDecode`。主线
FlashInfer **没有编译 FP4 KV decode kernel**，所以 cudagraph capture
报：

```
File ".../flashinfer/jit/attention/modules.py", line 77, in get_batch_decode_uri
    f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
KeyError: torch.float4_e2m1fn_x2
```

由于 CHANGE_0131 plumbing 只在 `MiniCPMAttentionBackend` 里，FP4 路径
在生产提交配置上根本到不了。

## 2. 规则合规说明

- 仅修改我们自己维护的开源代码（`server_args.py`，必要时改
  `MiniCPMAttentionBackend`）。
- 无 KV 布局 / 量化方案规则违规 —— FP4 KV 格式已在 CHANGE_0131 通过审查。
- 不动评测脚本和 chat template。
- 提交包形态不变（开关通过环境变量 opt-in；默认保持 FP8 e5m2 KV）。

## 3. 详细实施方案（变更前）

### 方案 A（原提案）—— **不可行**（证据见 §8）

原提案：在 `_handle_model_specific_adjustments` 里，当
`kv_cache_dtype == "fp4_e2m1"` 时跳过 `minicpm_flashinfer → flashinfer`
重写，假设 MiniCPM 自定义后端有一条 dense 路径能走。

**Round 13b 的代码审查（§8）表明该假设是错的**。自定义后端是
`MiniCPMSparseBackend`，在 `has_sparse_attention=False`（这正是
`force_dense_minicpm=True` 产生的状态）时会直接
`raise ValueError`。根本没有 dense 路径。连 `forward_decode`
也是无条件调 `get_topk_for_sparse`。跳过重写不会生效。

### 方案 A′（修订）—— 重构 MiniCPMSparseBackend

在 `MiniCPMSparseBackend` 里加一条真正的 dense 路径：

1. 放宽 `has_sparse_attention` 门限，让 backend 能在
   `force_dense_minicpm` 下初始化。
2. 在 `forward_decode`/`forward_extend` 中用一个新的
   `is_dense_run` 标志分支：开启时不调 `get_topk_for_sparse`、
   不走 `sparse_kernel_extension`，直接用完整 `page_table` 过
   `BatchDecodeWithPagedKVCacheWrapper`（已被CHANGE_0131 贴着为 FP4 感知）。
3. `init_cuda_graph_state` 要额外分配 dense（完整 page_table）的
   buffer。
4. 元数据构造器等同步调整。

工作量估计：几百行 + 小心的 cudagraph 重验证。不是一天能收到底的。

### 方案 B —— 重型替代

给主线 FlashInfer 的 decode wrapper 加 FP4 KV 支持。需要 FlashInfer
端为 `dtype_kv = float4_e2m1fn_x2` 加 kernel template。比赛时间线内
不可行。

### 方案 C（推荐）—— 冻结 CHANGE_0131/0132

保留四个启动期 bug 修复 commit，作为通用加固（`SOAR_FP4_KV_CACHE=0`
默认下它们在运行期不产生任何代价）。FP4 KV 实验冻结，除非：

- 其他代码路径需要该 KV 内存节省（例如面对官方超长上下文速度集出现 OOM）。
- 主线 FlashInfer 原生加了 FP4 KV 支持（方案 B 自动过期）。

转轻是优先走 `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md` 里高 ROI 的优化。

### 验证流程（仅当采用方案 A′ 时才需要）

```bash
# 1. 同步 + 重启
git push minicpm-src mixed_minicpm_cudagraph
python3 scripts/fcloud/fcloud_workflow.py sync
SOAR_FP4_KV_CACHE=1 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 2. Smoke S1（1 样本）
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1

# 3. 精度
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 4. 精度 ≥ 75% 则全套速度
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 4. 实际代码改动（变更后）

待打补丁后填。

## 5. 结果汇总表

| 变体 | 基线（FP8 KV）| 新（FP4 KV）| Δ |
|---|---|---|---|
| S1 (s) | 121.71 | TBD | TBD |
| S8 (s) | 44.09 | TBD | TBD |
| Smax (s) | 35.86 | TBD | TBD |
| ori_accuracy | 79.29% | TBD | TBD |
| KV 内存 / token / layer | 2048 B | 1152 B | −44% |

## 6. 回滚说明

```bash
git revert <commit-hash>
git push minicpm-src mixed_minicpm_cudagraph
```

或者直接不设 `SOAR_FP4_KV_CACHE` / 设为 `0`（默认）。这样
`KV_CACHE_DTYPE_ARG` 走 `fp8_e5m2`，方案 A 的分支永不被触发。

## 7. 下一步建议

鉴于 §8，推荐 **方案 C —— 比赛时间线内冻结 FP4 KV**：

1. 保留 R13 的四个加固 commit（`d4608f170`、`fd7e797ea`、
   `252cc4d64`、`8a0976593`）。`SOAR_FP4_KV_CACHE=0` 时它们是闲置的。
2. 转轻其他目录项（Marlin tile 调优、调度、推测解码变体等）。
3. 仅在以下情况重反 FP4 KV：
   - 主线 FlashInfer 加了 FP4 KV decode kernel（方案 B 自动过期）。
   - 有证据表明官方长上下文速度集 FP8 KV 会 OOM（内存压力压过
     方案 A′ 的 fork 成本）。

## 8. Round 13b 代码审查 —— 证据方案 A 不可行

在 commit `49a7ed5f4` 上对
`python/sglang/srt/layers/attention/minicpm_backend.py` 逐行核实。

### 发现 1 —— 后端硬要求 `has_sparse_attention=True`

[L221-230](../../python/sglang/srt/layers/attention/minicpm_backend.py#L221-L230)：

```python
self.has_sparse_attention = hf_config is not None and getattr(
    hf_config, "has_sparse_attention", False
)
if not self.has_sparse_attention:
    raise ValueError(
        "MiniCPM model must have sparse attention enabled. "
        "Please ensure the model config has 'has_sparse_attention=True'."
    )
```

同时 [`model_config.py` L238 / L248](../../python/sglang/srt/configs/model_config.py#L238)
上 `force_dense_minicpm=True` 会把 `has_sparse_attention` 覆盖为
False。所以生产配置下该后端 **根本初始化不了**。

### 发现 2 —— `forward_decode` 没有 dense 分支

[L1130-1300](../../python/sglang/srt/layers/attention/minicpm_backend.py#L1130-L1300)
无条件执行：

```python
topk_idx = self.get_topk_for_sparse(
    q_reshaped.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
    1, layer, forward_batch, False,
)
sparse_page_table = sparse_kernel_extension.get_block_table_v3(...)
```

没有 `if not is_sparse_layer: ...` 分支，也没有 `force_dense_minicpm`
短路。所有 decode 调用都走 sparse top-k 路径。

### 发现 3 —— `forward_extend` 的 `else` 分支也是 sparse 形状

[L989-1056](../../python/sglang/srt/layers/attention/minicpm_backend.py#L989-L1056)
中 `max(seq_lens) >= self.dense_len` 走完整 sparse top-k，`else`
分支仍走 `metadata.sparse_page_table`、`sparse_cache_seqlens_int32` 以及
压缩 key 分配器。短序列也跳不开 sparse 管道。

### 发现 4 —— config 层 `force_dense_minicpm` 的效果

[`model_config.py` L237-238](../../python/sglang/srt/configs/model_config.py#L238)
和 [L247-248](../../python/sglang/srt/configs/model_config.py#L248)
强制 `has_sparse_attention → False`、`sparse_layer_ids → []`。
发现 1 的 `raise` 加上空 `sparse_layer_ids` 让所谓“隐式方案 A”不可行。

### 结论

生产提交要求的组合（`GPTQ + --force-dense-minicpm +
--kv-cache-dtype fp4_e2m1`）**没有便宜路径** 可以走到现有 CHANGE_0131
plumbing。可行实现只有：

- 方案 A′：在 `MiniCPMSparseBackend` 里 fork 出一条真正的 dense 路径
  （重，几百行 LoC + cudagraph 重验证）。
- 方案 B：主线 FlashInfer FP4 KV decode kernel（超出范围）。

两者都不适合比赛时间线。推荐 **方案 C：冻结**。

