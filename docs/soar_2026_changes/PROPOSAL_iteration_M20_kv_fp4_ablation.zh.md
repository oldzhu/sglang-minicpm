# 提案：迭代 M2.0 — 逐层 NVFP4 KV 缓存敏感度消融实验

**日期**：2026-04-26
**状态**：提案 — 任何代码修改前需用户批准
**前置文档**：[`PLAN_post_v18_baseline_M2_kv_mixed_20260426_1029.md`](./PLAN_post_v18_baseline_M2_kv_mixed_20260426_1029.md)
**基线**：commit `8d1e4d12b`（v18 外科式回退，2026-04-26）

---

## 1. 背景与动机

### 1.1 战略背景

SOAR 第 5 周冠军（智算一队半决赛）在他们的 MiniCPM-SALA 提交中报告了 **混合 FP8 + NVFP4 KV 缓存** 的以下数据：

| KV 策略 | 他们的精度 | 结论 |
|---|---|---|
| 纯 FP8 KV | ~80% | 基线 |
| 纯 NVFP4 KV | ~75% | 低于可接受阈值 |
| **混合：首末层 FP8、中间层 NVFP4** | ~80% | **接近 FP4 的带宽 + 接近 FP8 的精度** |

他们的方法：**逐层敏感度消融** — 把单层 KV 从 FP8 切到 FP4，测精度变化，对所有层重复一遍，然后保留敏感层为 FP8、把鲁棒层切到 FP4。

### 1.2 为什么这能打中我们的弱点

- 我们官方最差的档位是 **Smax**（当前 2746-2917s）— 长上下文 decode。
- 官方速度数据集：约 68% 输入是 32K-512K token；长上下文 decode 是 **KV 读取带宽受限**。
- 把 KV 缓存带宽减半（FP8 → FP4）直接打中 Smax 主要瓶颈。
- 冠军已经在 **同一个模型族**（MiniCPM-SALA）上验证了这个方案 — 适用性高置信度。

### 1.3 对先前认知的重要纠正

| 误解 | 纠正 |
|---|---|
| "我们已经测过 NVFP4 失败了" | Test 21 测的是 **W4A4**（权重+激活都是 FP4）；KV 缓存仍是 FP8。NVFP4 KV 缓存是 **未测过的轴**。 |
| "Catalog 写着 NVFP4 NOT VIABLE" | Catalog 措辞仅适用于权重量化。KV 缓存 FP4 还没烧过。 |
| "需要从零写 FP4 显存池" | SGLang 上游 **已经有** `MHATokenToKVPoolFP4`（[memory_pool.py:1085](../../python/sglang/srt/mem_cache/memory_pool.py#L1085)）和 `--kv-cache-dtype fp4_e2m1` 标志。 |
| "消融需要 32 次运行" | MiniCPM-SALA 只有 **8 个标准注意力层**（`mixer_type==minicpm4`）；24 个 lightning（SimpleGLA）层用循环状态、没有分页 KV 缓存。消融是 **8 次运行**。 |

---

## 2. 规则合规性检查

| 规则 | 状态 |
|------|------|
| 用 v18 作基线 | ✅ 分支 HEAD = `8d1e4d12b`（v18 外科式回退） |
| 只 push 到 `minicpm-src` | ✅ 所有 commit 推到 `minicpm-src/mixed_minicpm_cudagraph` |
| 永不修改 `eval_model_001.py` | ✅ M2.0 不动评测脚本 |
| 保持精度 ≥ 78%（C ≥ 0.92） | ✅ M2.0 行为可逆：仅按层翻 KV 数据类型；通过环境变量随时关掉 |
| 提交包 ≤ 2GB | ✅ M2.0 不增加任何编译产物 |
| 量化在现场 | ✅ KV 缓存数据类型仅运行时 |
| 所有服务参数走 `prepare_env.sh` | ✅ M2.0 配置走 `SGLANG_SERVER_ARGS` |

---

## 3. 目标与非目标

### 3.1 目标
1. 实证测出 8 个标准注意力层中、每层 KV 从 FP8 翻到 NVFP4 的 **精度差**。
2. 找到一个 **鲁棒子集**：这些层一起切到 FP4 后精度仍 ≥ 78%（高于 C=0.92 底线 ≥1pt）。
3. 给出 **GO/NO-GO 决策**：是否要投入 M2.1（真实的逐层混合池实现）。
4. 估算如果鲁棒子集落地后的 **预期带宽节省**，确认实现成本值得。

### 3.2 非目标
- M2.0 **不会** 交付真实混合池实现（那是 M2.1+）。
- M2.0 **不会** 修改注意力后端、显存池内部或任何 kernel。
- M2.0 **不会** 改提交包 — 输出仅限文档 + 逐层 CSV。

---

## 4. 方案选择：**先 Option B（仿真量化），如有需要再 Option A（真 FP4）**

### 4.1 Option A — 通过现有上游池真实 FP4 存储
- 全局设 `--kv-cache-dtype fp4_e2m1` 跑精度。如果灾难性（与冠军 75% 接近），说明池工作正常。然后需要 **逐层覆盖钩子** —— 但这需要修改 [`model_runner_kv_cache_mixin.py:583`](../../python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L583)，让两个池（FP8+FP4）共存且每个注意力层指向正确池。
- 优点：数值就是 M2.1 最终交付的；带宽节省可观测。
- 缺点：消融数据还没拿到、就要先动上游代码，非取小。

### 4.2 Option B — BF16 存储里仿真 FP4（M2.0 推荐）
- 在注意力 forward 加薄钩子：当某层启用 FP4 仿真时，`K = (K / scale).round().clamp(-6, 6) * scale`，V 同理。
- 存储仍是 BF16，**但数值效果与真实 FP4 round-trip 完全一致**。
- 优点：代码面极小（~30 行单文件）；环境变量完全可逆；不需要动池创建代码；精度信号与 Option A 等价。
- 缺点：实际带宽不会减少，无法直接确认 Smax 增益 — 但这是 M2.1 的活。

**决定**：M2.0 用 Option B。M2.0 唯一需要的是精度信号，Option B 以最小风险给出。Option A 推到 M2.1（真实池集成时）。

### 4.3 逐层消融前的合理性检查
先做 **一次** 预跑：**全部 8 个标准层** 都用仿真 FP4，验证：
- 精度落在冠军纯 FP4 数字（~75%）附近；
- 仿真量化钩子真的在执行（不是默默无效）；
- 模型不崩、不出乱码。

如果预跑给 ~78%+（即 MiniCPM-SALA 比冠军模型更耐 FP4），可能不需要逐层消融了 — 8 层全 FP4 即可。
如果预跑给 ≤70%，我们模型比冠军更敏感；消融必须精细。

---

## 5. 详细实施计划

### 5.1 文件改动（仅 M2.0）

| 文件 | 改动 | 行数 | 类型 |
|------|------|------|------|
| `python/sglang/srt/layers/attention/minicpm_flashinfer.py`（或标准注意力 forward 所在文件） | 加 `SGLANG_KV_FP4_SIM_LAYERS` 环境变量门控的仿真量化钩子 | ~30 | 修改 |
| `benchmark/soar/demo_sala/m20_ablation_runner.py` | 新增：编排脚本 | ~150 | 新增 |
| `benchmark/soar/demo_sala/m20_ablation_results.csv` | 新增：逐层结果表 | 输出 | 新增（生成） |

### 5.2 钩子设计（伪代码）

```python
# 在该层的注意力 forward 中、计算完 K 和 V 之后：
fp4_sim_layers = os.environ.get("SGLANG_KV_FP4_SIM_LAYERS", "")  # 例如 "0,5,7" 或 "all"
if fp4_sim_layers:
    layer_set = parse_layer_set(fp4_sim_layers, layer_id, num_attn_layers=8)
    if layer_id in layer_set:
        K = fake_quant_fp4_e2m1(K)  # 量化到 FP4 网格、再反量化回 BF16
        V = fake_quant_fp4_e2m1(V)
```

其中 `fake_quant_fp4_e2m1` 实现：按块（如 block_size=16）absmax 缩放 + 在 FP4 E2M1 网格 `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` 上四舍五入，再反量化。

### 5.3 消融运行器设计

```python
# m20_ablation_runner.py — 伪代码
STANDARD_LAYER_IDS = read_from_config_json("mixer_types")  # mixer_type=="minicpm4" 的 8 个层

results = []
# Phase 0: 基线（无 FP4）+ 全 FP4 合理性
for label, env_val in [("baseline_fp8", ""), ("all_8_fp4_sim", "all")]:
    set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=env_val)
    acc = run_accuracy_eval()
    results.append({"label": label, "layers_fp4": env_val, "acc": acc})

# Phase 1: 逐层消融（8 次）
for lid in STANDARD_LAYER_IDS:
    set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=str(lid))
    acc = run_accuracy_eval()
    results.append({"label": f"only_layer_{lid}_fp4", "layers_fp4": str(lid), "acc": acc})

# Phase 2（可选，仅当 Phase 1 显示有清晰敏感集时）：
# 子集测试 — 验证"所有鲁棒层同时 FP4"仍达标
robust_set = [lid for lid, acc in phase1 if acc >= 78.0]
set_env_and_restart_server(SGLANG_KV_FP4_SIM_LAYERS=",".join(robust_set))
acc = run_accuracy_eval()
results.append({"label": "robust_subset_fp4", ...})

write_csv(results)
```

### 5.4 验证命令

```bash
# 用户批准后，本地仓库：
git checkout -b m20_kv_fp4_ablation 8d1e4d12b
# ... 实现钩子 + 运行器 ...
git push minicpm-src m20_kv_fp4_ablation

# fcloud 上：
python3 scripts/fcloud/fcloud_workflow.py sync
# 每次（共 10 次：1 基线 + 1 全 FP4 + 8 逐层 + 1 鲁棒子集）：
SGLANG_KV_FP4_SIM_LAYERS="<value>" python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

总计算时长：约 10 × (3 分钟重启 + 5 分钟精度) ≈ **80 分钟**。

---

## 6. 决策矩阵（消融后）

| 场景 | 结论 | 下一步 |
|------|------|--------|
| 全 FP4 已经 ≥78% | MiniCPM-SALA 高度耐 FP4 | 跳过 M2.1 钩子复杂度；直接 8 层全 FP4，进入真实池集成 |
| 子集 ≥78%、全 FP4 < 78% | 混合精度是正确答案 | **GO** M2.1，使用发现的子集 |
| 最佳子集 76-78% | 边缘风险，速度增益若大可能值得 | 条件 GO；M2.1 dry-run 确认 Smax 增益后再决定 |
| 最佳子集 < 76% | KV FP4 路径死路 | **NO-GO**；转向 catalog 中其他项（SM120 native MMA、FP8 权重等） |

---

## 7. 结果总结模板（运行后填写）

| 运行 | FP4 层 | acc_ori | mcq | qa | cwe | fwe | niah | 备注 |
|------|--------|---------|-----|-----|-----|-----|------|------|
| 0 | 无（基线） | TBD | | | | | | |
| 1 | 全部 8 标准层 | TBD | | | | | | |
| 2 | 仅 L_a | TBD | | | | | | |
| ... | ... | | | | | | | |
| 9 | 仅 L_h | TBD | | | | | | |
| 10 | 鲁棒子集 | TBD | | | | | | |

（层 ID 待从 fcloud 上 `config.json` 读出后填入。）

---

## 8. 回滚说明

M2.0 改动均为环境变量门控且增量：

```bash
# 关闭：直接 unset 环境变量（默认行为不变 = FP8）
unset SGLANG_KV_FP4_SIM_LAYERS

# 完整撤销（删除钩子代码）：
git revert <m20_hook_commit>

# 删分支：
git branch -D m20_kv_fp4_ablation
git push minicpm-src --delete m20_kv_fp4_ablation
```

提交包 **永不** 受 M2.0 影响（分支只到本地，直到 M2.1 落地）。

---

## 9. 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| MiniCPM-SALA 完全 FP4 敏感（无鲁棒子集） | 中 | M2 死得便宜：仅烧 ~80 分钟 fcloud 时间 |
| 仿真量化与真实 FP4 数值不完全匹配（边缘 case） | 低 | M2.0 末尾另跑 1 次真实 FP4（Option A）与全 8 仿真做对比；若误差 ≤0.5pt，仿真量化可信 |
| 8 个标准层的 KV 带宽只占 decode 总带宽的小部分（vs 24 个 SimpleGLA 的 state I/O） | 中 | 基线时跑 SGLang 自带 profiler 测；若标准注意力 KV 读取 < 30%，M2 ROI 缩水，应不投 M2.1 |
| 钩子在禁用 FP4 时也带来延迟 | 极低 | 环境变量在服务启动时读、层集合冻结；每 token 无开销 |

---

## 10. 后续建议（M2.0 之后）

| 阶段 | 触发条件 | 工作量 |
|------|---------|--------|
| **M2.1** | M2.0 GO 决策 | 3-5 天：`model_runner_kv_cache_mixin.py` 中加逐层数据类型配置；双池共存；注意力层索引修复 |
| M2.2 | M2.1 上线 | 2-3 天：fcloud 完整速度测试（真实 FP4 存储）；预期 +15-30% Smax |
| M3（并行） | 独立 | 1-2 天：重建本地 `speed_{s1,s8,smax}.jsonl`，用长上下文输入匹配官方分布 |
| M4（拉伸目标） | 仅当 M2 表现不及预期 | 2+ 周：SM120-native MMA / mxfp8 GEMM（按 CHANGE_0125_001 建议） |

---

## 11. 待用户确认

1. **确认 M2.0 范围**：仅 Option B（仿真量化），结果出来后再决定 M2.1 — 同意？
2. **分支策略**：建独立分支 `m20_kv_fp4_ablation`，还是直接在 `mixed_minicpm_cudagraph` 上提？
3. **仿真量化保真度验证**：M2.0 末尾加 1 次真实 FP4 交叉检验（额外 ~10 分钟）—— 加吗？
4. **KV 带宽预先剖析**：开始消融前先 `nsys` profile 一下，量化标准注意力 KV 占总带宽的比例 —— 做吗？

---

## 附录 A — 为什么 BF16 中的仿真量化与真实 FP4 给出相同精度

真实 FP4 KV 缓存流程：
```
写：  K_bf16 → quantize_to_fp4(K_bf16, scale) → 存为打包 uint8 + scale
读：  load 打包 uint8 + scale → dequantize_to_bf16 → K_bf16'
```

仿真量化流程：
```
写：  K_bf16 → quantize_to_fp4 → dequantize_to_bf16 → 存为 bf16（同一个 K_bf16'）
读：  load bf16（已经是 K_bf16'）
```

`K_bf16'`（反量化后的表示）在两条流程中 **逐位相同**。因此注意力下游计算产生相同的 logits、相同的 token ID、相同的精度。唯一差别是带宽/存储 — 而 M2.0 不测这个。

参考：这是 PyTorch QAT 的标准 fake-quant 技术（`torch.fake_quantize_per_tensor_affine`），数学严格。
