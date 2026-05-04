# Phase A 设计 — NVFP4 基线（暂不含 FourOverSix、不含 Medusa）

**日期**：2026-05-04
**作者**：Agent（任何源码改动前等待用户批准）
**父提案**：[PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md) §4 Phase A
**前置版本**：v22（`SOAR_TORCH_COMPILE_MAX_BS=24`，commit `234f3fed8`）

---

## A0. Phase A 目标

在 fcloud SM120 上把 **NVFP4 权重 + FP8_e5m2 KV + dense + Marlin/Cutlass FP4 GEMM** 全链路打通，单一环境变量切换；本地 **acc ≥ 78%（保守底线）**、**Smax ≤ 35s**。**不**做 FourOverSix、**不**做 Medusa。仅验证 FP4 权重链路健康。

通过则发版 **v23**；未通过则 unset 一个环境变量即可回到 v22 字节等价。

## A1. 为什么用 NVFP4（而非 MXFP4 / modelopt_fp8）

| 格式 | 块大小 | scale 数据类型 | 取值集合 | sglang 加载器 | SM120 GEMM TFLOPS | 选用？ |
|---|---|---|---|---|---|---|
| MXFP4 | 32 | E8M0 | E2M1 | `mxfp4` | cutlass mxfp4 | 否 — 权重精度更差（scale 仅 8-bit 二的幂） |
| **NVFP4** | **16** | **FP8（E4M3）** | **E2M1，±{0,0.5,1,1.5,2,3,4,6}** | `modelopt_fp4` | **593 TF（FP4 张量核心）** | **是** |
| GPTQ W4A16（当前）| 128 | FP16 | INT4 | `gptq_marlin` | 148 TF | 否 — 留 4× 余量 |
| modelopt_fp8 | per-tensor | FP32 | E4M3 | `modelopt_fp8` | 296 TF | 否 — FP4 的一半 |

NVFP4 也是第 7 周冠军 / arXiv:2512.02010 使用的格式；FourOverSix（Phase B）是 NVFP4 上的逐 block scale 选取细化，因此格式必须一致。

## A2. 量化管线选型 — **NVIDIA Model Optimizer + sglang `modelopt_fp4` 加载器**

Phase A **不**扩展 `gptqmodel` 让其输出 NVFP4。原因：

1. `prepare_env.sh` 里固定的 `gptqmodel 5.7.0` 没有原生 NVFP4 导出；现写一个非平凡，恰好就是 Phase B（GPTQ 内嵌 FourOverSix）的事。两件事不能混在一起。
2. sglang `modelopt_quant.py` 已支持读取 `hf_quant_config.json` / `quantization_config.quant_algo == "NVFP4"`。我们只要产出符合该格式的 checkpoint。
3. nvidia-modelopt 是 Apache 2.0 — **合规**。
4. modelopt 的 NVFP4 校准是 E2M1 格点上的 MSE round-to-nearest，恰好对应 FourOverSix 论文的 M=6-only 基线（FoS 在 Phase B 引入 M=4 备选）。

### A2.1 Phase A 安装行（追加到 `prepare_env.sh`）

```
uv pip install --no-deps "nvidia-modelopt[hf]==0.31.0" "scikit-learn"
```
0.31.0 是已知对 cu128 稳定的 NVFP4 导出版本，固定精确版本，`--no-deps` 防止覆盖 torch/transformers。

> 若 fcloud 无法装上（网络），自动走 A2.2 的回退路径。

### A2.2 modelopt 不可用时的回退方案

我们在仓内预留一个小型 NVFP4 量化器 `benchmark/soar/demo_sala/nvfp4_quantize.py`（~120 行，由我们署名 Apache 2.0），逻辑：
```
对每个 Linear：
  对每行的每个 16 元素 block：
      block_max = max(|w|)
      scale_amax = block_max / 6.0          # 6.0 = NVFP4 最大幅值
      block_scale_fp8 = quant_to_fp8_e4m3(scale_amax)
      w_int4 = round_to_lattice(w / dequant(block_scale_fp8), {0,±.5,±1,±1.5,±2,±3,±4,±6})
      两个 int4 打包成一个 uint8
输出 hf_quant_config.json，quant_method=modelopt，quant_algo=NVFP4，group_size=16
```
这正是 Phase B 必须复用的代码（FourOverSix 是在该例程上做 `min(err_M=6, err_M=4)`），所以即便 modelopt 路走得通，这部分代码**不浪费**。

**决策**：先试 modelopt；安装失败 → 自动回退仓内实现。preprocess 时由 `import nvidia_modelopt` try/except 决定。

## A3. 模块包含/排除策略

与现行 GPTQ 一致 — `lightning-attn` 的 `o_gate`、`z_proj` 是极小的门控投影，4-bit 量化下 acc 损失不成比例，**必须**排除。

| 模块 | 处理 |
|---|---|
| `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | NVFP4 |
| `mlp.gate_proj`, `up_proj`, `down_proj` | NVFP4 |
| `self_attn.o_gate`, `z_proj` | **排除**（保持 BF16） |
| `lm_head` | **排除**（保持 BF16） |
| 所有 norm、所有 embedding | **排除**（保持 BF16） |

`hf_quant_config.json` 的 `exclude_modules` 字段携带此列表 — sglang 的 `ModelOptFp4Config.from_config` 会读取（在 modelopt_quant.py:1012-1017 验证）。

**注意**：本步骤**取消**当前的 `sparse_qkv_w8` 混合精度（sparse-attn 层 QKV 走 W8）。Phase A 故意采用**统一 NVFP4**。如果 acc 跌破 78%，下一轮（Phase A.1）再恢复 attn QKV 的 W8 混合。

## A4. 服务端接线（sglang 不改一行）

`sglang.launch_server` 已能通过以下任一方式自动识别 NVFP4：
```
config.json
└── quantization_config: {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", ...}
```
或独立的 `hf_quant_config.json`。`python/sglang/srt/layers/quantization/modelopt_quant.py`（已在仓内，1700 行，原封不动）的 `modelopt_fp4` 加载器会处理。

我们只需**去掉** `--quantization gptq`，让自动识别接管；如有显式参数，覆写为 `--quantization modelopt_fp4`。

KV cache 保持 `fp8_e5m2`（v22 默认）。FP4 KV（`SOAR_FP4_KV_CACHE=1`，CHANGE_0131）在 Phase A **保持关闭** — 那是另一个独立坐标轴。

## A5. 改动文件清单（仅 Phase A）

| 文件 | 改动类型 | 行数估计 |
|---|---|---|
| `benchmark/soar/demo_sala/prepare_env.sh` | 新增 `SOAR_QUANT_PROFILE` 环境变量（默认 `gptq`）；当 `nvfp4` 时把 `SGLANG_SERVER_ARGS` 量化标志切到 `modelopt_fp4` 并固定不同 `MODEL_PATH`；追加 `nvidia-modelopt` 安装 | ~30 |
| `benchmark/soar/demo_sala/preprocess_model.py` | 新增 `run_nvfp4_quantization(...)`；`main()` 按 `SOAR_QUANT_PROFILE` 分发；复用 `_patch_chat_template_for_mcq` 与 include/exclude 逻辑；输出 `hf_quant_config.json`（modelopt 路径）或我们手写（回退路径） | 新增 ~150 + 分发 ~20 |
| `benchmark/soar/demo_sala/nvfp4_quantize.py` | **新文件**，仓内回退量化器（仅当 modelopt 导入失败时使用） | ~120 |
| `docs/soar_2026_changes/CHANGE_0150_phase_a_nvfp4_baseline.{en,zh}.md` | **新建**，结果文档，fcloud 测试后再填 | 各 ~80 |

Phase A **不**触碰：
- `gptqmodel_minicpm_sala.py`（仅 `SOAR_QUANT_PROFILE=gptq` 时使用）
- `python/sglang/srt/` 下任何文件（modelopt_fp4 上游已支持）
- 评测脚本（仓库铁律）

## A6. 单一开关规范

`prepare_env.sh` 增加：

```bash
# Phase A：权重量化档位选择。
#   gptq      = 当前 v22 基线（sparse_qkv_w8 GPTQ W4A16）
#   nvfp4     = 统一 NVFP4 权重（modelopt 优先，回退到仓内实现）
#   nvfp4_fos = NVFP4 + FourOverSix 自适应 M=6/M=4 [Phase B]
export SOAR_QUANT_PROFILE="${SOAR_QUANT_PROFILE:-gptq}"
```

`preprocess_model.py`（离线，on-site 量化）和服务参数块都读这一个变量：
- preprocess：分发到 `run_gptq_quantization` vs `run_nvfp4_quantization`。
- server：当 `nvfp4*` 时把 `--quantization gptq` 替换为 `--quantization modelopt_fp4`（或依赖自动识别；两条路都验一遍）。

**默认仍是 `gptq`**，v22 字节等价。

## A7. 验证方案

### A7.1 本地小模型自检（不上 fcloud）
1. 开发容器装好环境，`python -c "from nvidia_modelopt.torch.quantization import quantize"` 验证安装。
2. `preprocess_model.py --mode nvfp4` 跑一个 10 层 stub 模型 → 检查输出 `hf_quant_config.json` schema 与 sglang 期望一致。

### A7.2 fcloud 量化跑通
1. `start-instance`（征得用户同意）。
2. `sync` 推送代码。
3. fcloud 上：`cd /root/submission_sim && SOAR_QUANT_PROFILE=nvfp4 bash prepare_model.sh --input <bf16> --output <out>` — 时长目标 ≤ 3h（与现行 GPTQ 同量级）。
4. 检查输出：体积、`quantize_config.json` / `hf_quant_config.json`、NVFP4 vs BF16 模块计数。

### A7.3 fcloud 上线
1. `SOAR_QUANT_PROFILE=nvfp4 python3 scripts/fcloud/fcloud_workflow.py restart-server`。
2. `wait-server` 健康检查。
3. 冒烟：`curl /v1/models` 返回 `quantization_method=modelopt_fp4`。
4. `accuracy` 跑通。判通过：ori_accuracy ≥ 78%。
5. `speed --variant all`。判通过：Smax ≤ 35s。
6. `pause-instance`。

### A7.4 量化器算术正确性自检（离线）
随机抽 5 个 Linear 权重，反量化 NVFP4 → BF16 与原 BF16 计算 MSE，**要求 < 5e-3**。这是脱离模型的量化器正确性校验。

## A8. 通过 / 失败 / 决策矩阵

| 结果 | 动作 |
|---|---|
| 校准 ≤ 3h，acc ≥ 78%，Smax ≤ 35s | **发版 v23**。写 CHANGE_0150。进入 Phase B。 |
| 校准完成，acc 76-78% | 恢复 attn QKV 的 `sparse_qkv_w8` 混合（Phase A.1，半天工作量）。 |
| 校准完成，acc < 76% | Phase A 暂停。检查排除清单（lm_head？embedding？gate norm？）。最坏回到 v22。 |
| 校准完成，Smax > 35s | NVFP4 GEMM 没真正生效；查 `--quantization` 参数 / cutlass kernel 调度；可能需要 `--enable-flashinfer-cutlass-fp4` 之类。 |
| 校准崩溃 / OOM | 排查每 block 内存；校准 batch 调到 1；持续失败 → **自动回退 A2.2**。 |
| modelopt 装不上 | **自动回退 A2.2**。 |

## A9. 回滚

```bash
unset SOAR_QUANT_PROFILE   # 或 SOAR_QUANT_PROFILE=gptq
```
重启 server，v22 字节等价。

如已落盘 NVFP4 权重，无需清理 — `profile=gptq` 时不会被加载。

## A10. 工作量与依赖顺序（不给具体日历）

1. 用户批准 Phase A 设计。
2. 实现 env 开关 + 分发器 + modelopt 路径 + 回退量化器（同一 PR，本分支）。
3. 本地 stub 自检（A7.1 + A7.4）。
4. 用户批准 fcloud 跑测。
5. fcloud 跑测（A7.2 + A7.3）。
6. 写 CHANGE_0150.{en,zh}.md。
7. 通过 → v23 打包 + 提交窗口。
8. 失败 → 诊断；A.1 子迭代或回退。

## A11. 等用户回答的几个问题

1. **`prepare_env.sh` 里允许装 `nvidia-modelopt` 吗？** 它是 Apache 2.0、PyPI 上有。如果你不想多依赖，直接走 A2.2 仓内实现（额外 ~120 行，全部我们可控）。**我的建议**：先试 modelopt，它是参考实现，能省调试量化算术的麻烦。
2. **Phase A 是否保留 `--force-dense-minicpm`？** 默认保留（与 v22 对齐）。如果你想顺便切原生 sparse 模式，那是另一个坐标轴，应放到 Phase A.2。
3. **Phase A 是否带上 MXFP4 KV cache（`SOAR_FP4_KV_CACHE=1`）？** 默认关（v22 是 FP8 e5m2）。Phase A 通过后再 A/B。
4. **打包**：v23 tarball 会多 `nvfp4_quantize.py`（~5KB）和稍长的 `prepare_env.sh`。**不要**预量化打包 NVFP4 权重，量化必须在 on-site 跑（规则 §3）。请确认。

## A12. 交叉引用

- 父提案：[PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md](PROPOSAL_nvfp4_fouroversix_and_medusa_20260504.zh.md)
- 调研：[RESEARCH_week7_champion_review_20260504.zh.md](RESEARCH_week7_champion_review_20260504.zh.md)
- sglang NVFP4 加载器：`python/sglang/srt/layers/quantization/modelopt_quant.py:863-1100`
- 既有 FP4 KV 接线（独立坐标轴）：[CHANGE_0131_nvfp4_kv_p2_plumbing.zh.md](CHANGE_0131_nvfp4_kv_p2_plumbing.zh.md)
- 硬件：[SM120_RTX_PRO_HARDWARE.md](SM120_RTX_PRO_HARDWARE.md)（FP4 = 593 TF，4× BF16）
- nvidia-modelopt：https://github.com/NVIDIA/TensorRT-Model-Optimizer（Apache 2.0）
- FourOverSix 论文（Phase B 参考）：arXiv:2512.02010

---

## 等待用户回复

回复 **approve A-design** 按上述方案开始实施。
回复 **A11 各题答案** 调整范围（例如：跳过 modelopt，直接仓内实现）。
回复 **adjust** 提结构性变更。

建议：**approve A-design**，A11 取默认（先试 modelopt、保 dense、FP8 KV、on-site 量化）。这是最低风险、最贴近冠军方案的路径。
