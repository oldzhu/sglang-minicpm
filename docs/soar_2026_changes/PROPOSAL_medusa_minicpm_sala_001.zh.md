# 提案 — 在 MiniCPM-SALA 上落地 Medusa 推测解码（iter 001）

状态：**提案 — 尚未实施；待批准**
日期：2026-05-09
分支：`mixed_minicpm_cudagraph`
基线：GPTQ sparse_qkv_w8 + FP8_e5m2 KV + dense + Tier1 + flashinfer + torch.compile bs=24（commit `ac91b1afe`）
当前最佳本地数（今日回测 `GPTQ-FP8-DENSE-retest-newinst`）：ori_acc=77.47%、norm=96.83%、**S1=110.68 / S8=40.33 / Smax=32.53**。

## 1. 动机

SOAR 第七周冠军"香草小张"在文章 https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q 中报告：在 MiniCPM-SALA 上集成 **Medusa 推测解码 + GLA 状态分叉**，在所有并发层级下都能稳定提升 decode 端到端吞吐，K=1 verify 单步开销 **≈0.39 ms**。Medusa 主要打击我们计分中权重最高的 **S₁（40%）**——并发=1 下 decode 受 weight bandwidth 制约，正是 Medusa 最擅长的场景。

提交得分：
```
最终分 = (S₁·0.40 + S₈·0.30 + S∞·0.30) × C
S_N = (Duration_best / Duration_player) × 100
```
当前本地 S1=110.68 s。我们已知第 5 名 v18-A 官方 S1 是 426 s，冠军更快——所以本地 S1 每减少 1 秒，在 C 维持 1.0 的前提下都直接换算成可计量的总分。

## 2. 为什么 Medusa 比目录里其他速度方向更值得做

| 方向 | 现状 / 结果 | 结论 |
|---|---|---|
| Marlin tile 重新调优（CHANGE_0125） | Test 27 中性 | 已饱和 |
| W4A8 / FP8-blockwise GEMM（CHANGE_W4A8_001） | Test 27a：S1 +118% | 反向 |
| 当前 HEAD 上的 sparse 路径 | Round 13d、R13e、CHANGE_0136：挂死 / 旧 bug | 阻塞 |
| 激进调度（Tier1） | v22 已默认开启 | 已饱和 |
| `torch.compile` bs 扫描（#2A bs=24） | Smax −3.2%，accuracy 不变 | 已上线 |
| **Medusa K=1/K=2** | **冠军已验证；我们仓库未触碰** | **应推进** |
| 训练好的 EAGLE3 draft | Test 22 用随机 draft 已 C=0；训练版未试 | 备选（详见 RESEARCH 文档） |

收益估算：Medusa K=1 在 accept_rate p≈0.5 时每个 decode step 等价于 1+p 个 token → S1 预期 **−25–33%**，S8/S∞ 收益更小。25% 的 S1 缩短（110.68→83 s）按 score 的反比近似可换 **≈ +10 分**（S₁ 只占 40% 权重，但绝对降幅大）。

## 3. 规则合规检查（SOAR 2026）

参 `.github/copilot-instructions.md`：

- ✅ **"Speculative heads allowed (count toward 2GB)"** — Medusa heads（3 × MLP，≤100 MB）远小于 2 GB。
- ✅ **"Code: Apache 2.0, reproducible, explainable"** — sglang 本身 Apache-2.0；我们新增的（model wrapper + worker glue + 训练脚本）全部 Apache-2.0。
- ✅ **"Quantization + evaluation time ≤ 5 hours"** — head 训练**离线**完成；只有 quant + eval 计时。
- ✅ **"All files ≤ 2GB total"** — 当前 tarball ~731 MB；+100 MB heads 仍远低于 2 GB。
- ✅ **"Submission is reproducible"** — head 权重随 tarball 一起提交；不需要在线下载。
- ✅ **正确实现下无损** — Medusa verify 会拒绝任何与 base model argmax/sample 不一致的 draft，所以与 no-spec baseline 的精度差异只可能来自：(a) verify 数值噪声；(b) GLA state-fork 的 bug。

**核心正确性风险。** MiniCPM-SALA 包含 24 个 GLA（Lightning Attention）层，递推态 `h_t = exp(−γ)·h_{t−1} + k_t·v_tᵀ`。tree-verify 中 K 条候选路径必须每条 sibling 都从**父节点的** GLA 状态分叉，否则递推态会把别的分支的历史污染进来 → 隐式精度下降。这就是 **CHANGE_0072 / Test 22 EAGLE3** 失败的根因（`accept_rate=0.26`, `S1 +65 %`, `C=0`）。GLA-fork 是本提案最核心的技术任务。

## 4. 我们距冠军还差什么

| 模块 | 冠军 | 我们今天 |
|---|---|---|
| 基础量化 | NVFP4 FP4 + FourOverSix | GPTQ W4A16 sparse_qkv_w8 |
| GLA 状态分叉 | 已实现 | **缺失** |
| Medusa heads（基于评测分布加权数据训练） | 已实现 | **缺失** |
| Tree-attention mask 黏合 | 已实现 | sglang 在 EAGLE 路径有；需要适配 |
| K（heads 数） | 文中报告 K=1，K≥2 暗示 | 待定 |
| Verify 开销 | K=1 ≈0.39 ms | 待测 |

注：冠军同时叠了 NVFP4 + Medusa。我们本地的 NVFP4-FOS 单跑（CHANGE_0151_007）显示 NVFP4 单独使用 **S1 反而 +57%**，因为 FP4 路径缺 `flashinfer` decode kernel，只能 fallback 到 cuTLASS BF16。所以我们将在 **GPTQ 之上** 做 Medusa（GPTQ 路径下 SM120 decode kernel 已优化）。如果将来 NVFP4 路径修好，Medusa 框架可复用。

## 5. 分四阶段推进（每阶段都需独立审批）

**任何代码改动都不会在 R1 批准前发生。**

### Phase R1 — 框架打通（随机权重，不训练）

目标：tree-verify + GLA-fork 在 SALA 上跑通且不 crash、不掉精度。

预计文件：
- 新增：`python/sglang/srt/speculative/medusa_info.py`（树结构 dataclass，参考 `eagle_info.py`）
- 新增：`python/sglang/srt/speculative/medusa_worker.py`（继承 `base_spec_worker.BaseSpecWorker`；复用 `eagle_utils.py` 的 tree-mask 工具）
- 新增：`python/sglang/srt/models/minicpm_medusa.py`（带 K 个 Medusa head 的 MiniCPM-SALA；MLP + 分类器：`p_t^(k) = softmax(W₂^(k)·(SiLU(W₁^(k)·h_t) + h_t))`；`W₁` 全零初始化）
- 修改：`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` — GLA 状态读写支持 "branch_id" 参数
- 修改：`python/sglang/srt/layers/attention/fla/{chunk.py,fused_recurrent.py}` — 递推路径接受并尊重 per-token branch index
- 修改：`python/sglang/srt/server_args.py` — 注册 `MEDUSA` 枚举（与现有 `EAGLE`/`EAGLE3`/`NEXTN`/`STANDALONE`/`NGRAM` 并列）
- 修改：`benchmark/soar/demo_sala/prepare_env.sh` — 增加 `SOAR_SPEC_MEDUSA=1` opt-in（默认 0；保持 v22 字节等价）

验证：
- 本地正确性 diff：相同 prompt + 相同 seed，分别用 `SOAR_SPEC_MEDUSA=0` 和 `=1` 跑（`accept_threshold=1.0` 即只接受 argmax）→ 至少 10 个代表样本（mcq + niah + cwe + qa）token 序列必须 byte-identical。
- fcloud 烟囱测试：server 启动正常，单 `/generate` 返回有效完成。测一次 verify 单步延迟。
- **中止条件**：如果 `accept_threshold=1.0` 下精度与 baseline 不一致，停。说明 GLA-fork 实现有 bug。

预估工作量：2–3 天编码 + 1 次 fcloud（约 1 h）。

### Phase R2 — GLA 状态分叉正确性

目标：把 R1 的 K=1 确定性 verify 扩展到真正的树分叉，确认 verify 开销与冠军 0.39 ms 在同量级。

任务：
- 实现 branch-id 感知的 GLA 状态 buffer，shape `(batch, branch, heads, ...)`。每个 verify step 之前把 parent state 广播到所有 sibling 分支。
- 把 sglang 已有的 `tree_mask` 构造工具（`srt/speculative/eagle_utils.py`）接到 Medusa worker 上。
- 验证 harness：CPU 参考实现，把递推路径与 verify 树路径分别走一遍小 prompt 集，逐 accepted token 断言 hidden-state 相等。

验证：
- 50 样 mcq + niah + cwe，`accept_threshold=1.0` 下 byte-identical。
- 150 样全量 eval（默认采样温度），norm 精度与今日 96.83% 相差 ±0.5 pt 以内。
- Speed：测 K=1 verify 开销，目标 ≤1 ms。

**中止条件**：如果 K=1 lossless 模式（`accept_threshold=1.0`）至少 50 prompt 都 byte-identical 做不到，GLA-fork 结构性错误，不进 R3。

预估：3–5 天编码 + 2 次 fcloud。

### Phase R3 — 在评测分布上训练 Medusa heads

目标：把 accept_rate 拉到 0.5–0.7。

输入：
- 训练语料：对 `perf_public_set.jsonl` 中 5 个任务类型（qa/mcq/cwe/fwe/niah）做分层采样。冠军明确指出：**评测分布加权采样优于随机采样**。
- Heads：K∈{1,2,3}；每个 head 一份 MLP（结构见 §4）。
- Loss：标准 Medusa 下一 k token cross-entropy；主模型冻结。
- 计算：head 权重 ≈ (`hidden_size`)² × 2 + (`hidden_size`·vocab) × K。SALA hidden_size≈4096 vocab≈73440 K=2 → ~2 × 32 MB MLP + 2 × 600 MB classifier = **~1.2 GB**，**naive 实现会破 2 GB 上限**。缓解：**直接复用主模型 `lm_head` 当 classifier**（文章其实就是这么做的——只有 `W₁`、`W₂` 是 head-specific MLP，最终分类器是主模型已有的 LM head）。复用后 ~2 × 32 MB ≈ 64 MB，完全够。

验证：
- Held-out slice 上的 head loss 曲线。
- fcloud 复评：norm 精度 ≥ 96.83%，accept_rate ≥ 0.45。

预估：训练 1 天（单台 H100/RTX 6000）+ pipeline & 数据治理 2 天 + 集成 1 天。

### Phase R4 — Bench + 方差探针 + 提交包

- 跑两次 accuracy + speed（S1, S8, Smax）做方差。
- 把实测加速率写进 `OPTIMIZATION_CATALOG_GPTQ_FP8_DENSE.md`。
- 打 tarball；验证 ≤2 GB；验证 `prepare_model.sh` 加载 heads 后仍在时间预算内。
- 双跑共识规则：两次都 C ≥ 0.96 且 S1 改善 ≥ 10% 才提交。

## 6. 文件 / 行数估算（R1–R2）

```
python/sglang/srt/speculative/medusa_info.py          +200（新）
python/sglang/srt/speculative/medusa_worker.py        +400（新）
python/sglang/srt/models/minicpm_medusa.py            +250（新）
python/sglang/srt/layers/attention/fla/chunk.py       +50  （branch_id 参数）
python/sglang/srt/layers/attention/fla/fused_recurrent.py  +30
python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py  +80
python/sglang/srt/server_args.py                      +5
python/sglang/srt/speculative/eagle_utils.py          +20  （re-export tree_mask）
benchmark/soar/demo_sala/prepare_env.sh               +20  （opt-in env）
benchmark/soar/demo_sala/preprocess_model.py          +10  （head 权重透传）
```

R3 增 `tools/train_medusa_heads.py`（~300 行）和训练好的 `.safetensors`。R4 仅增打包胶水。

## 7. 测试命令

R1 之后：
```
# fcloud
cd /root/submission_sim
source prepare_env.sh
export SOAR_SPEC_MEDUSA=1
python3 -m sglang.launch_server --model-path "$MODEL_PATH" "${SGLANG_SERVER_ARGS[@]}"
# 期望：启动 OK，单 /generate 第一个 token 与 SOAR_SPEC_MEDUSA=0 完全一致
```

R2 之后：
```
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
# 在温度=0 下 diff predictions.jsonl，norm Δ 须在 ±0.5pt 以内
```

## 8. 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| GLA-fork bug 隐蔽，静默掉精度 | 高 | 速度结论之前必须先过 `accept_threshold=1.0` byte-identity 闸 |
| `torch.compile` 图爆炸（verify forward ≠ decode forward） → 启动时间暴涨 | 中 | `SOAR_SPEC_MEDUSA=1` 时先关闭 torch.compile；R3 后再视情况打开 |
| Head 训练塌缩到 base 分布（accept_rate 低） | 中 | 用冠军的评测分布加权数据；先 K=1，K=1 ≥0.5 才上 K=2 |
| 提交包超 2 GB | 低 | 复用 `lm_head` 作分类器；预算已核 ≤100 MB |
| Fcloud SM120 显存不足以同时跑 verify+main | 低 | verify 与主 forward 共权重；只有 state buffer 增大 |

## 9. 回滚

所有改动由 `SOAR_SPEC_MEDUSA=0`（默认）门控。回滚 = 不导出该 env。新增文件是 addition；修改的文件在 env 关时早返回原路径。

## 10. 总工作量估计

| 阶段 | 工时 | fcloud 轮次 | 累计 |
|---|---|---|---|
| R1 plumbing | 2–3 d | 1 | 3 d |
| R2 GLA-fork | 3–5 d | 2 | 8 d |
| R3 训练 heads | 4 d | 1 | 12 d |
| R4 bench + package | 1 d | 1 | 13 d |

## 11. 决策门

每阶段（R1 spike、R2 fork、R3 training、R4 packaging）开始前都会请你审批。每阶段产出一份双语 CHANGE_NNNN 报告。

## 12. 下一步具体动作（待你"go"）

只批 Phase R1。我会：
1. 在 `mixed_minicpm_cudagraph` 分支落 R1 文件 deltas。
2. 本地跑 syntax / 单测一遍。
3. fcloud 恢复后做 fcloud 烟囱测试（单 `/generate`、verify 单步延迟）。
4. 写 `CHANGE_0153_medusa_phase_r1.{en,zh}.md`，提交并 push。

**显式批准前，本提案仅作为文档存在，不会动代码。**

## 参考

- 冠军原文：https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- Medusa 论文：Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" — ICML 2024（https://arxiv.org/abs/2401.10774）
- 配套综述：[RESEARCH_speculative_decoding_survey_001.zh.md](RESEARCH_speculative_decoding_survey_001.zh.md)
- 之前失败的 EAGLE3 spike：[TEST_RESULTS_TRACKING.md](TEST_RESULTS_TRACKING.md) Test 22-acc
- 早期对 EAGLE3 / GLA 阻塞的分诊：[CHANGE_0072_three_path_optimization_research.zh.md](CHANGE_0072_three_path_optimization_research.zh.md)
