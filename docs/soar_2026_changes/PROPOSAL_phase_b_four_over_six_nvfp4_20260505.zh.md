# 提案 —— Phase B：FourOverSix 自适应 NVFP4（每块 M=6/M=4 量级选择）

日期：2026-05-05
分支：`mixed_minicpm_cudagraph`
状态：**仅提案 —— 未经批准前不动代码**
依赖：`CHANGE_0150_phase_a_nvfp4_baseline.{en,zh}.md`

## 0. 摘要

Phase A 已经确认：
- SM120 上 FP4 cutlass 内核相对 BF16 在 prefill/chunk 形状上有 ~4× 提速。
- 统一 NVFP4 模型质量崩塌（mcq 输出乱码 "unedoc.com/abc/u.com…"）。

Phase B 套用公开冠军博客的 **FourOverSix** 技术：对每个 16 元素的 NVFP4 块，
按块级重建误差自动在 M=6 / M=4 两种量级中选一个。这样能恢复精度而保留 NVFP4 的存储
布局 —— 因此**不需要内核改动**：Phase A 验证过的 `flashinfer.mm_fp4` cutlass 路径
仍然适用。

目标：先通过冒烟测试（短 prompt 给出连贯回答），再在 fcloud 上跑全量精度 + S1/S8/Smax，
然后决定是否提交、或在其上再叠加 FP8 KV / 融合 norm。

## 1. 背景

### NVFP4 的存储 & "M" 到底是什么

NVFP4 每个 16 元素块里存**两样**量化产物：

1. **每元素的 4-bit 码** —— 共 16 个，取值在固定格点
   `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` 上。两个码打包进一个字节，所以
   `(N, K)` 的权重张量变成 `uint8` 形状 `(N, K/2)`。
2. **每块标量 `s`** —— 每 16 元素块一个 FP8 E4M3 数，张量形状 `(N, K/16)`、
   dtype 为 `float8_e4m3fn`。反量化就是 `w ≈ code · s`。

格点本身由 NVFP4 标准固定（也是 `flashinfer.mm_fp4` /
`cutlass_scaled_fp4_mm` 在硬件里解码的对象）。**`M` 不出现在存储里**，它只是
**校准时挑 `s` 的规则**：

```
M = 6 规则：  s = max(|w_block|) / 6     # 块内最大值映射到格点极值 ±6
M = 4 规则：  s = max(|w_block|) / 4     # 块内最大值映射到格点 ±4；
                                          # 大于 4·s 的部分被截到 ±4·s
```

格点一样，存储布局一样，只是 `s` 不同。内核完全不知道也不关心是哪个规则算出来的，
只看到一个块标量。

### 何时 M=6、何时 M=4 占优？

由块级 MSE 自己决定；直觉两边都成立：

- **M=6 占优**：块里只有一个离群点、其余值都很小。`s` 取得小，能保住零附近的分辨率；
  那个孤立的离群点照样被四舍五入到 ±6。
- **M=4 占优**：块"顶部稠密"—— 大量值聚在块最大值附近。M=6 把格点拉伸到 ±6，但 ±5、±6
  几乎用不到，多数活跃值在 {±2, ±3, ±4} 这种很疏的点之间四舍五入。M=4 把格点
  `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4}` 完全压在活跃区里，中段值能舍到更近的点。代价是
  落在 `4·s_M4` 与 `max` 之间的值会被截到 ±4·s_M4；如果"块最大值本身"就是唯一这种值，
  截断误差很小。

冠军博客在 LLaMA / Qwen 上观察到 ~40–43 % 的块挑 M=4；MiniCPM-SALA-90 上的比例需要实测。

**FourOverSix** = 校准时对每块独立挑选 `M ∈ {4, 6}`，使块级重建误差
`||round_to_lattice(w/s) · s − w||²` 最小。

得到的结果仍然是标准 NVFP4 存储（uint8 打包码 + 每块 FP8 E4M3 标量），所以
`flashinfer.fp4_quantize` / `mm_fp4` / `cutlass_scaled_fp4_mm` 都不用改。
**只是在校准阶段挑选的标量值变了。**

### 规则合规

- Apache-2.0、可复现、可解释：是 —— 全过程在场量化、确定性。
- 无私有数据：仅使用公开的 90 条分层校准集。
- 量化 + 评测 ≤ 5 h：FourOverSix 每块多算 O(16)，校准时间可忽略。
- 提交包 ≤ 2 GB：与 Phase A 同一 NVFP4 存储布局。

## 2. 目标与预期收益

### 成功的判定

| 指标 | GPTQ v22 基线 | Phase A（统一 NVFP4） | Phase B 目标 |
|---|---|---|---|
| 冒烟测试（单条短 prompt） | 连贯 | 乱码 | **连贯** |
| ori_accuracy（90 条 mcq+qa+niah+cwe+fwe） | 79.29 % | 无（超时） | **≥ 78 %** |
| 归一化精度 | 99.11 %（C=1.0） | 0（淘汰） | **≥ 99 %**（C=1.0）目标，≥ 97 %（C≥0.92）硬底线 |
| S1 / S8 / Smax（本地集） | 121.71 / 44.09 / 35.86 s | 全部超时 | **≥ 0.5×** 基线（即 S1 ≤ 240 s） |
| 官方分（粗估） | 30.04（#22） | 0 | 若 FP4 GEMM 4× 在官方长上下文集上兑现，会有明显跳升 |

**速度**收益的上限就是 Phase A 微基准里量到的：prefill/chunk 上 ~4×、decode 上 ~1×
（甚至更慢）。能否在 harness 里兑现取决于 prefill/decode 占比、注意力主导度和并发数。
Phase B 的首要目标是**让我们能开始测速度**，而不是榨更多速度。

### 现实地讲，可能的失败模式

- 即便加了 FourOverSix，MiniCPM-SALA-90 是一个针对长上下文重 fine-tune 的模型，
  权重分布未必和冠军在 LLaMA / Qwen 上验证的一致。掉 2 pp 以上是有可能的。
- 离群层（例如 MoE 风格 gating、过 rotary 的投影）可能必须保留更高精度。
  我们现有的排除列表已经把 `o_gate`、`z_proj`、`lm_head`、`norm`、`embed_tokens`
  保留为 BF16；可能还要加。
- decode 主导的 benchmark 可能相对 GPTQ 基线**回退**，因为 Phase A 微基准显示
  FP4 decode-1 = 0.49–0.92× BF16。如果官方集 Smax 是 decode 主导的，Phase B 可能
  在 Smax 上掉分，即使精度无问题。

## 3. 规则合规检查

| 规则 | 合规情况 |
|---|---|
| 在场量化 | ✅ 通过 `prepare_model.sh --input … --output …`，与 Phase A 同流程 |
| 提交包 ≤ 2 GB | ✅ 与 Phase A 相同 NVFP4 布局（~2.7–3 GB safetensors，加 `lm_head`、`embed_tokens` BF16 后还是合规）—— 量化完成后实测 |
| 量化 + 评测 ≤ 5 h | ✅ FourOverSix 校准开销 < 1 % |
| 不泄露私有/评测数据 | ✅ 沿用 GPTQ 用过的 90 条分层校准集 |
| Apache-2.0、可复现 | ✅ 给定校准集后是确定性的 |
| 精度 > 97 %（C ≠ 0） | ⚠ 必须实测；这就是 Phase B 自身的过关条件 |
| 不绕过 `--flush-cache` / 固定并发 | ✅ 没有运行时改动 |
| eval 脚本完整性 | ✅ 不动 `eval_model_001.py` |

## 4. 实施计划（**未改动代码 —— 待批**）

### 4.1 两个候选实现路径

**两条路径产出的 checkpoint 完全相同**（带 FourOverSix 标量的 NVFP4），只是**在调用栈
里的位置不同**。FourOverSix 只需要权重本身（不需要激活），因此放在 modelopt 校准循环
里并不能多拿到任何信息。

**(B1) 改写 modelopt 的 NVFP4 量化器（深侵入）**
- 子类化 `modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor`（0.43 上具体路径
  实现时再确认），替换其标量选择逻辑。
- 优：天然复用 modelopt 的校准数据。
- 缺：与 modelopt 内部紧耦合，小版本升级可能挂；难以靠肉眼审查。

**(B2) 事后改写标量（浅侵入） —— 推荐**
- 让 modelopt 完整跑完它默认的 NVFP4 校准（与 Phase A 完全一样）。
- 在调用 `export_hf_checkpoint` 之前，遍历每个被量化的 Linear 模块，读取原 BF16 权重，
  按 FourOverSix 重新算每块标量，**就地把 FP8 E4M3 标量张量和 uint8 4-bit 码张量
  一起改写**，再 export。（码也要改写，因为 `s` 一动，每个权重要四舍五入到的格点就变了。
  存储布局仍是标准 NVFP4。）
- 优：完全在我们自己的代码里，独立、好测、好回滚。
- 缺：每块多做一倍计算（两个 M 都算 MSE），但只是权重一遍 pass，毫秒级。

**决定：先做 B2。** 跑通了就根本不用 B1。如果纯 MSE 选法不够（比如需要激活感知），
那时再考虑 B1。

### 4.2 算法（B2）—— 伪代码

```python
# 对每个被 NVFP4 量化的 Linear 层（即 weight_quantizer 已启用）
W = layer.weight  # 原 BF16 权重，形状 (N, K)
B = 16            # NVFP4 块大小

# 重排成 (N, K/B, B)，每行切成 K/B 个 16 元素块
Wb = W.view(N, K // B, B)

block_max = Wb.abs().amax(dim=-1)  # (N, K/B) 每块的最大绝对值

# 两个候选标量
s6 = block_max / 6.0
s4 = block_max / 4.0

# 每个候选标量下的量化误差
def err(scale):
    q = round_to_nvfp4_lattice(Wb / scale.unsqueeze(-1))
    deq = q * scale.unsqueeze(-1)
    return ((deq - Wb) ** 2).sum(dim=-1)  # (N, K/B)

e6 = err(s6)
e4 = err(s4)

pick_m4 = e4 < e6  # (N, K/B) bool

new_scale = torch.where(pick_m4, s4, s6)

# NVFP4 存储要求每块标量是 FP8 E4M3
new_scale_fp8 = new_scale.to(torch.float8_e4m3fn)

# 用新标量重新量化权重，写回打包 uint8 码
new_codes_uint8 = pack_nvfp4(round_to_nvfp4_lattice(Wb / new_scale_fp8.unsqueeze(-1).to(torch.float32)))

# 覆写 modelopt 量化器内部状态
layer._weight_quantizer.amax = ...   # modelopt 用以推导标量的属性
layer.weight = new_codes_uint8       # modelopt 量化后存放权重的属性
# (具体属性名实施时验证)
```

modelopt 0.43 内部属性名要看具体实现，需要在 fcloud 上开一个短 jupyter 会话先探一下，
这一步是下面"实施步骤 1"。

### 4.3 涉及文件

1. `benchmark/soar/demo_sala/preprocess_model.py`
   - 新增辅助函数 `_apply_four_over_six(model)`，遍历所有 NVFP4 化的 Linear 层并按上面
     算法改写标量。
   - 在 `run_nvfp4_quantization` 里，`mtq.quantize(...)` 与 `export_hf_checkpoint(...)`
     之间调用它。
   - 加环境变量开关 `SOAR_NVFP4_FOUR_OVER_SIX=1`（`nvfp4_fos` profile 默认 ON，
     `nvfp4` 默认 OFF）以便 Phase A 的"统一 NVFP4"仍能复现。

2. `benchmark/soar/demo_sala/prepare_env.sh`
   - `SOAR_QUANT_PROFILE=nvfp4_fos` 分支已经在校验里被接受。
   - 让 `nvfp4_fos` 在调 preprocess 前设置 `SOAR_NVFP4_FOUR_OVER_SIX=1`。
   - 服务侧参数：与 `nvfp4` 完全一致（`--quantization modelopt_fp4`）。

3. （sglang 源码不动）

4. （sgl-kernel 不动）

### 4.4 实施步骤（批准后执行）

| 步骤 | 内容 | 位置 | 验证 |
|---|---|---|---|
| 1 | 探 modelopt 0.43 NVFP4 内部 | fcloud jupyter 临时 exec | 打印一个量化后的 Linear 的属性，把属性名写进实现 PR |
| 2 | 写 `_apply_four_over_six`（B2） | 本地编辑 `preprocess_model.py` | 在合成 (256, 4096) 张量上做单元测试：MSE 下降、标量仍是 FP8 E4M3、解码往返 OK |
| 3 | 接 `nvfp4_fos` profile | `prepare_env.sh` + 参数派发 | `bash -n prepare_env.sh`；打印 SGLANG_SERVER_ARGS |
| 4 | 同步到 fcloud，跑量化 | `python3 scripts/fcloud/fcloud_workflow.py sync` + 手动 `prepare_model.sh` | 量化跑完，输出目录大小合理 |
| 5 | 冒烟测试：单条短 prompt | curl 一条 mcq | 回答连贯，不出现 "unedoc.com" 类乱码 |
| 6 | 全量精度评测 | `fcloud_workflow.py accuracy` | 读 `predictions.jsonl`；对比 ori_accuracy |
| 7 | 速度 S1/S8/Smax | `fcloud_workflow.py speed --variant all` | 记入 `TEST_RESULTS_TRACKING.md` |
| 8 | 暂停实例 | `fcloud_workflow.py pause-instance` | （强制省钱规则） |

### 4.5 测试 / 基准命令

```bash
# 本地：纸上验证
cd /home/oldzhu/sglang/benchmark/soar/demo_sala
python3 -c "
import preprocess_model as p
import torch
W = torch.randn(256, 4096, dtype=torch.bfloat16) * 0.05
W[0, 0] = 5.0  # 一个块里塞个离群点
# 调辅助函数；断言形状、并验证有部分块挑了 M=4
"

# fcloud（同步后）：
exec on fcloud:
cd /root/submission_sim
SOAR_QUANT_PROFILE=nvfp4_fos bash prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/MiniCPM-SALA-NVFP4-FOS

# 验证量化跑了
ls -la /root/models/MiniCPM-SALA-NVFP4-FOS

# 重启服务
SOAR_QUANT_PROFILE=nvfp4_fos source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path /root/models/MiniCPM-SALA-NVFP4-FOS \
  --host 0.0.0.0 --port 30000 \
  "${SGLANG_SERVER_ARGS[@]}"

# 冒烟
curl -s http://localhost:30000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'

# 全量精度
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 速度
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### 4.6 结果汇总表（待跑完填写）

| 指标 | GPTQ v22 基线 | Phase A（统一 NVFP4） | Phase B（FourOverSix） |
|---|---|---|---|
| ori_accuracy | 79.29 % | 无（超时） | TBD |
| 归一化 | 99.11 % | 0 | TBD |
| S1 (s) | 121.71 | 超时 | TBD |
| S8 (s) | 44.09 | 超时 | TBD |
| Smax (s) | 35.86 | 超时 | TBD |
| 输出目录大小 | 4.4 GB（GPTQ + lm_head BF16） | 6.2 GB | 预计 ~6.2 GB |
| 备注 | 提交基线 | 仅打通 plumbing | 决策点：提交 / 退回 |

### 4.7 回滚

- Phase B 全部改动由 `SOAR_NVFP4_FOUR_OVER_SIX=1` / `nvfp4_fos` profile 控制。
- 回滚：把 `SOAR_QUANT_PROFILE` 改回 `gptq`（v22 生产基线）—— 一个环境变量，不用回退代码。
- 如果想把源码也退回：`git revert <phase-B-commit>`，统一 `nvfp4`（Phase A）和 `gptq`
  （v22）两个 profile 都仍能用。

## 5. 风险（汇总）

| 风险 | 缓解 |
|---|---|
| 纯 MSE 选块过粗 → 精度仍掉 | 留 B1（modelopt 内部改）作为 fallback；也可以用激活幅值 / Hessian 加权 MSE |
| 离群层需要更高精度 | 已经排除 5 个模式；可按层名正则继续加 |
| modelopt 0.43 下个小版本属性名变 | `nvidia-modelopt==0.43.0` 已在 `prepare_env.sh` 钉死，并在代码里 assert |
| Smax 可能回退 | 记录；若是真问题，下一阶段提"FP4 ffn + GPTQ attn"混合精度提案 |
| 提交包 > 2 GB | `lm_head` + `embed_tokens` BF16 是大头；如模型支持可走 `tie_word_embeddings`（实施时验证） |

## 6. 后续建议（Phase B 落地后）

1. 精度 OK 且速度持平/略好：
   - 叠加 FP8 KV cache（`--kv-cache-dtype fp8_e5m2`）—— 与 Phase B 正交。
   - 重新打开 `--enable-fused-qk-norm-rope` 和 `--enable-mixed-chunk`（已经在
     `prepare_env.sh` 的 SGLANG_SERVER_ARGS 里，确认不与 `modelopt_fp4` 冲突即可）。
2. 精度 OK 但 Smax 回退：
   - 混合精度：注意力 GPTQ-W4A8、FFN NVFP4。需要运行时分发逻辑，新提案。
3. 加了 FourOverSix 精度仍崩：
   - 激活感知 FourOverSix（权重 MSE × 激活幅值）。
   - 敏感度分析：每层与 BF16 ref 的精度差，自动排除前 K 敏感的层。

## 7. 申请批准

请批准 **B2（事后改写标量）** 这条实现路线。批准后我会：
1. 在 fcloud 上开一段短 jupyter exec，确认 modelopt 0.43 NVFP4 量化器的属性名。
2. 在 `preprocess_model.py` 里写 `_apply_four_over_six`，并接通 `nvfp4_fos`。
3. push 到 `minicpm-src/mixed_minicpm_cudagraph` 后跑上面的验证流。
4. 结果落地后写对应的 `CHANGE_0151_phase_b_four_over_six.{en,zh}.md`，然后
   暂停实例。
