# CHANGE_0152 — 应用 OpenBMB PR #10 `_init_rope` 修复以兼容 transformers ≥ 4.43 / 5.x

## 背景与动机

OpenBMB 在 HuggingFace 讨论区
[`openbmb/MiniCPM-SALA/discussions/10`](https://huggingface.co/openbmb/MiniCPM-SALA/discussions/10)
（commit
[`f28de5e4`](https://huggingface.co/openbmb/MiniCPM-SALA/commit/f28de5e488b065a06bec8526d9683c14fe83bf7b)）
官方公告中发布了对 `modeling_minicpm_sala.py` 的官方修复，用以解决新版
transformers 上的硬加载失败：

```
ValueError: Unknown RoPE scaling type default
```

根因：`transformers>=4.43` 在配置加载阶段统一了 `rope_scaling` 字段，会把缺失/None
自动填成 `{"rope_type": "default", "factor": 1.0}`。原版 `_init_rope()` 只识别原
始的 `None`、`"linear"`、`"dynamic"`、`"longrope"`，遇到新增的 `"default"`
直接抛错。当 `gptqmodel` 7.0.0 把 `transformers` 拉到 5.8.0 后，量化与推理
两条路径都会因此挂掉。

我们曾在 `preprocess_model.py` 里通过内存补丁
（`_install_gptqmodel_minicpm_rope_patch`）在 `_init_rope` 之前把
`config.rope_scaling` 强制清成 `None` 来绕过该问题。但这个钩子只覆盖离线量化
路径——sglang 推理时没有装这个钩子，所以量化后的模型用 `trust_remote_code=True`
加载时同样会触发 `ValueError`，因为 transformers 5.x 在每次加载 config 时都会
重新填回 `"default"` 标记。

官方修复纯粹是 trust-remote-code 模型文件的修改（不会改变行为，因为我们的
config 没有 `rope_scaling`），需要在所有加载点都同步生效。

## 规则合规说明

- 修改对象是 **模型自带的 trust-remote-code Python 文件**，由模型作者
  （OpenBMB）官方公开发布。
- 不动 eval 脚本，不做按任务的生成参数覆盖，无任何禁用技巧。补丁与
  上游一字不差。
- 当 `config.rope_scaling` 缺失时（我方场景），新旧代码路径都选用
  `MiniCPMRotaryEmbedding`，行为完全一致。
- 提交约束：零运行时开销，零额外权重，对 2 GB 提交包大小无影响。

## 详细实现计划（修改前）

`modeling_minicpm_sala.py` 中存在两个 `_init_rope()`：
- `MiniCPMAttention._init_rope`（约第 879 行）
- `LightningAttention._init_rope`（约第 2144 行）

二者结构几乎一致。上游修复（PR #10）做了：

1. 把 `rope_scaling = self.config.rope_scaling` 提取为局部变量；
2. 把 `None` 与 `scaling_type in (None, "default")` 都视作 **不缩放**
   （走 `MiniCPMRotaryEmbedding` 分支）；
3. 同时接受旧版 `"type"` 与新版 `"rope_type"` 键；
4. `LongRoPE` 分支里的 `self.config.rope_scaling[...]` 下标读取改为读取
   局部变量 `rope_scaling[...]`。

模型文件由 HuggingFace `trust_remote_code=True` 下载到模型目录，并不在本仓库
里。我们因此在 SOAR 提交链路里唯一会触碰模型文件的入口
`preprocess_model.py` 中**就地**修补该模型目录下的文件。

修补位置：
- 三种模式（`copy`、`gptq`、`nvfp4`）完成后，对 `dst`（输出）目录修补——
  覆盖 sglang 推理路径。
- GPTQ 量化的加载路径已被既有的 `_install_gptqmodel_minicpm_rope_patch`
  保护，因此不去动用户提供的 `src` 目录。

幂等性：补丁会写入标记注释
`transformers>=4.43 standardizes rope_scaling`；如果文件已包含该标记，
patcher 直接跳过。

## 实际代码改动（修改后）

- `benchmark/soar/demo_sala/preprocess_model.py`
  - 新增 `_patch_modeling_init_rope_inplace(model_dir, label)` 与
    `_INIT_ROPE_PATCH_MARKER`、`_INIT_ROPE_OLD_HEADER`、
    `_INIT_ROPE_NEW_HEADER`、`_INIT_ROPE_OLD_ELSE`、`_INIT_ROPE_NEW_ELSE`
    常量，与上游 PR #10 一字不差。
  - 在 `gptq`、`nvfp4`、`copy` 三种模式各自 finalize 之后调用
    `_patch_modeling_init_rope_inplace(dst, ...)`。

## 验证

本地测试（已执行——见提交日志）：

```python
# 往返测试：从上游 post-fix 文件出发反推一份 pre-fix，运行 patcher，比对
# 是否与上游 post-fix 字节一致。同时验证幂等性。
```

输出：

```
[preprocess][init-rope-patch] post-fix idempotency: ... already patched; skip
OK idempotent on already-patched file
[preprocess][init-rope-patch] synth pre-fix: patched ... (replaced 2 _init_rope headers, 2 else-branches)
EXACT MATCH after patch on synthetic pre-fix
```

量化期验证（下一轮 fcloud，NVFP4-FOS iter-4）：

```bash
# preprocess 运行后
grep -n "transformers>=4.43 standardizes rope_scaling" \
  /root/models/MiniCPM-SALA-NVFP4-FOS/modeling_minicpm_sala.py
# 期望：2 处命中（每个 _init_rope 各一处）
```

运行时验证：在 `transformers==5.8.0` 下 sglang server 干净启动，没有
`Unknown RoPE scaling type default` 报错。

## 结果摘要

| 指标                                  | 修改前                     | 修改后                              |
|---------------------------------------|----------------------------|-------------------------------------|
| 在 `transformers>=4.43` 下加载        | 量化时 `ValueError`        | 干净加载                             |
| 通过 sglang `trust_remote_code` 加载 | 有 `ValueError` 风险       | 模型文件已修补                       |
| 补丁来源                              | 本地内存 hack              | 上游 PR #10 一字不差                |
| 行为变化（`rope_scaling=None`）       | 无                          | 无——仍走 `MiniCPMRotaryEmbedding` |
| 幂等性                                | 无                          | 标记位检测，可重复执行               |

## 回滚

```bash
git revert <commit>
```

patcher 是纯增量逻辑且由标记位守护；revert 即同时移除函数与所有调用点，
后续的 preprocess 不会再修改 `modeling_minicpm_sala.py`。

## 下一步建议

1. NVFP4-FOS iter-4：将 `SOAR_NVFP4_MAX_CALIB_SEQ_LEN` 设为 4096，沿用
   iter-2 的分层 90（qa、mcq、cwe + FOS=1）校准集，使用 iter-1 的 Tier-1
   调度运行精度评测，保留 < 70 % abort 闸门。
2. iter-4 精度确认后，考虑在 fcloud 上把同一补丁同步到 HF cache 里的拷贝
   （`~/.cache/huggingface/modules/...`）以覆盖任何绕过 preprocess 的代码路径。
