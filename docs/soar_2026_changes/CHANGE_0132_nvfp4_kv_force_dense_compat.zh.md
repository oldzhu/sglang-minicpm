# CHANGE 0132 — NVFP4 KV 与 `--force-dense-minicpm` 兼容

**状态**：提案（待批准）
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

### 方案 A —— 首选（小改动、低风险）

在 `python/sglang/srt/server_args.py` 的
`_handle_model_specific_adjustments` 中，找到 `force_dense_minicpm` 时
把 `minicpm_flashinfer` 改写成 `flashinfer` 的代码块，**当
`kv_cache_dtype == "fp4_e2m1"` 时跳过该重写**：

```python
# server_args.py（示意）
if self.force_dense_minicpm:
    if self.kv_cache_dtype == "fp4_e2m1":
        # 保留 MiniCPM 自定义后端，使 MiniCPMAttentionBackend 中的
        # FP4 KV plumbing 可达。该后端本身已通过 force_dense_minicpm
        # 内部分支支持 dense-only batch。
        pass
    else:
        if self.attention_backend == "minicpm_flashinfer":
            self.attention_backend = "flashinfer"
        # ... 既有重写
```

为什么这样可行：`MiniCPMAttentionBackend` 已有 dense-only 路径（它内部
会查 `force_dense_minicpm`），CHANGE_0131 已用 `self.use_fp4_kv_cache`
门控 FP4 逻辑。FP4 时保留 `attention_backend == "minicpm_flashinfer"`，
plumbing 就会跑起来。

### 方案 B —— 兜底（重）

给主线 FlashInfer 的 decode wrapper 加 FP4 KV 支持。需要 FlashInfer
端为 `dtype_kv = float4_e2m1fn_x2` 加 kernel template。比赛时间线内
不可行。

### 验证流程（方案 A）

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

1. 方案 A 在 dense 上 smoke 绿、精度 ≥ 75% 后：
   - 量化释放出来的 KV 内存能带来多少吞吐 / max-running-requests 提升。
   - 增益 < 5%：放下，转 Marlin tile 工作。
   - 增益 ≥ 5%：开 CHANGE_0133，把 sparse 路径下的 FP4 重启
     （CHANGE_0131 §3 Gap A + B）。
2. 方案 A 精度不过：怀疑 MiniCPMAttentionBackend 的 dense 路径在
   `minicpm_flashinfer` 与主线 `flashinfer` 之间存在分歧 —— 调 per-layer
   输出对照。
