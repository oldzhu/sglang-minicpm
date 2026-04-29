# CHANGE_0136 — `sparse_dense_len` 运行时可调阈值(每请求 dense/sparse 路由开关)

## 状态: 提案(待批准)

## 1. 背景与动机

CHANGE_0135_001 单请求 profile (Round 13e Option-B) 已经证明,在 BF16 + 原生 sparse + FP8 KV 路径下:

- BF16 cuTLASS GEMM 在 32k / 64k / 128k 都占 67–77% GPU 时间。
- sparse FA 自身只占 **3.6–4.3%**,线性扩展。
- compress_k1/k2 fill (CHANGE_0133 之后) 已降到 1.0–1.1%,可控。

那次跑用 BF16(无 GPTQ)是因为之前 `GPTQ + sparse + FP8 KV` 在公开集上崩精度(Tests 5/6/8)。但 profile 也告诉了我们一件有意思的事:**稀疏机制本身又快又稳**,在 SOAR 上让 BF16+sparse 慢下来的是 BF16 GEMM 的成本,不是稀疏算法。

这暗示了一个**还没测过**的组合,理论上有可能赢过 Test 12:
> `--quantization gptq_marlin` (Marlin INT4 GEMM 全程,跟 Test 12 一样)
> + `minicpm_flashinfer` 后端(对长请求**启用**稀疏路由)
> + FP8 KV
> + **高 `dense_len` 阈值**:只有长度高于阈值的请求才走 sparse,短请求走 dense + Marlin INT4 全速。

Test 12 通过 `--force-dense-minicpm` 让**所有**请求都走 dense。CHANGE_0136 保留 Test 12 在短请求上的行为,但允许长请求(那些大部分时间花在 attention I/O 上的)享受 sparse top-k。

GPTQ+sparse 历史上崩精度的原因可能是**所有**层一直走 sparse。阈值高了之后,只有少数长样本走 sparse — 把失败模式的爆炸半径压下来。

## 2. 规则合规

- 仅 server-side 运行时旋钮。不改模型文件(`config.json` / 权重 / tokenizer 都不动)。
- 不改 eval 脚本。
- 不需要重建 sgl-kernel。
- 与提交打包兼容(CLI flag / env var 进 `prepare_env.sh` `SGLANG_SERVER_ARGS`,正是官方支持的客制化界面)。
- 不设环境变量时行为与今天完全一致(走 `hf_config.sparse_dense_len`)。

## 3. 实现计划(改动前视图)

### 3a. 当前 `sparse_dense_len` 读取点

三处:

- [`python/sglang/srt/layers/attention/minicpm_backend.py:238-239`](../../python/sglang/srt/layers/attention/minicpm_backend.py#L238-L239)(每 batch 路由判断,**真正的 dispatch 点**):
  ```python
  self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
  self.config_dense_len = hf_config.sparse_dense_len
  ```
- [`python/sglang/srt/layers/attention/minicpm_sparse_utils.py:958`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L958):
  ```python
  dense_len = hf_config.sparse_dense_len
  ```
- [`python/sglang/srt/configs/model_config.py:281-283`](../../python/sglang/srt/configs/model_config.py#L281-L283) — 默认 512 的 accessor。
- [`python/sglang/srt/configs/minicpm.py:47,83,95`](../../python/sglang/srt/configs/minicpm.py) — config plumbing(默认 512,可被 `config.json` 的 `sparse_config.dense_len` 覆盖)。

### 3b. 每请求决策逻辑

`MiniCPMAttentionBackend.dense_len` 是单一的每请求阈值:`seq_len < dense_len` 则在 sparse 层内部走 dense FlashInfer,否则走 sparse top-k + paged-KV。所以在 server-arg 层面覆盖 `dense_len` 已经足够,不必碰模型。

## 4. 提议改动(改动后视图)

**单一 env var,在 backend 构造时读。不改 CLI parser(把改动面控到最小)。**

`python/sglang/srt/layers/attention/minicpm_backend.py` ~L238:

```python
import os

self.dense_as_sparse = model_runner.server_args.dense_as_sparse
_env_override = os.environ.get("SOAR_SPARSE_DENSE_LEN")
if _env_override is not None and not self.dense_as_sparse:
    try:
        _override = int(_env_override)
        if _override < 0:
            raise ValueError("must be >= 0")
        self.dense_len = _override
        self.config_dense_len = _override
        logger.info(
            f"SOAR_SPARSE_DENSE_LEN override: dense_len={_override} "
            f"(model config default was {hf_config.sparse_dense_len})"
        )
    except (ValueError, TypeError) as e:
        logger.warning(
            f"SOAR_SPARSE_DENSE_LEN={_env_override!r} invalid ({e}); "
            f"falling back to config value {hf_config.sparse_dense_len}"
        )
        self.dense_len = hf_config.sparse_dense_len
        self.config_dense_len = hf_config.sparse_dense_len
else:
    self.dense_len = 0 if self.dense_as_sparse else hf_config.sparse_dense_len
    self.config_dense_len = hf_config.sparse_dense_len
```

并在 [`minicpm_sparse_utils.py:958`](../../python/sglang/srt/layers/attention/minicpm_sparse_utils.py#L958) 镜像同样的覆盖,或者(更清爽)从 backend 把已解析的值传过去而不是重读 config。打补丁时挑更干净的那条。

`benchmark/soar/demo_sala/prepare_env.sh`(env 不设时行为完全不变):

```bash
# 可选:覆盖每请求的稀疏路由阈值。
# seq_len < SOAR_SPARSE_DENSE_LEN 走 dense FlashInfer;长于此值的走 sparse top-k。
# 默认 = 不设 → hf_config.sparse_dense_len (512)。
# 推荐 sweep 值: 524288 (sanity 全 dense)、65536、32768、16384。
if [[ -n "$SOAR_SPARSE_DENSE_LEN" ]]; then
    export SOAR_SPARSE_DENSE_LEN
fi
```

提交跑次的 `SGLANG_SERVER_ARGS` 不变,完全由 env 控制。

## 5. 验证计划(在 Test 12 baseline 上做 3 步矩阵)

每步 server config 都与 **Test 12** 完全一致:
GPTQ + FP8_e5m2 KV + `--attention-backend minicpm_flashinfer`(**不带** `--force-dense-minicpm`)+ chunk=32K,prefill-max-req=1,running=24,sched-cons=1.0,mixed-chunk,torch.compile bs=8。

**重要:** 本实验**必须去掉** `--force-dense-minicpm`,因为该 flag 把 dispatch 短路掉,会让阈值失效。路由完全由 `SOAR_SPARSE_DENSE_LEN` 控制。

| 步骤 | `SOAR_SPARSE_DENSE_LEN` | 预期行为                                  | 通过标准 |
|------|-------------------------|--------------------------------------------|----------|
| **a (sanity)** | `524288` | 所有请求走 dense → 必须复现 Test 12 数据 | ori_acc 与 Test 12 ±1pt 内;S₁ 与 121.71s ±2% 内 |
| **b (保守)**   | `65536`  | 只有最长尾样本走 sparse                   | norm_acc ≥ 99% (C=1.0);S₁/S₈/S∞ 各档与 Test 12 ±2% 内 或更好 |
| **c (激进)**   | `16384`  | 大部分长上下文样本走 sparse               | norm_acc ≥ 99% (C=1.0) **且** S₁/S₈/S∞ 中至少一档比 Test 12 快 > 2% |

如果步骤 (a) 与 Test 12 不一致 → 补丁有 bug,先修。
如果步骤 (b) norm_acc < 99% → 立即回滚,这条路不安全(对应历史上的 GPTQ-sparse 崩),不跑 (c)。
如果步骤 (b) 精度通过但所有档速度都没收益 → 公开集长样本太短/太少,记下来停。
如果步骤 (c) 精度+速度都通过 → 视作新 baseline 候选,再用 `--max-concurrent 32` 复跑确认在官方风格负载下也成立。

## 6. 风险分析

- **代码面:** ≤ 30 行 Python,单函数。无 kernel 重建。
- **默认行为:** env 不设时不变,Test 12 路径完全没碰。
- **提交兼容性:** `prepare_env.sh` 是官方客制化界面,env 门控旋钮是最干净的加法。
- **精度风险:** 中等 — 触发的路径(`GPTQ + 长请求走 sparse`)正是 Tests 5/6/8 崩过的那条。**缓解:步骤 (b) 把 norm_acc ≥ 99% 当门;不达标就在跑 (c) 之前中止。**
- **速度风险:** 相对 Test 12 无 — 步骤 (a) 完全复现 Test 12,步骤 (b)/(c) 只在长请求上分流。
- **CHANGE_0133 依赖:** 稀疏路径现在依赖 CHANGE_0133 的 bounded compress_k fill。该修复已在 HEAD。本实验前如果 CHANGE_0133 被回滚,先恢复它再跑。

## 7. 结果汇总表(占位 — 跑完填)

| 步骤 | dense_len | ori_acc | norm_acc | C   | S₁ (s) | S₈ (s) | S∞ (s) | 备注 |
|------|-----------|---------|----------|-----|--------|--------|--------|------|
| a    | 524288    |    —    |    —     |  —  |   —    |   —    |   —    | sanity |
| b    | 65536     |    —    |    —     |  —  |   —    |   —    |   —    | 保守 |
| c    | 16384     |    —    |    —     |  —  |   —    |   —    |   —    | 激进 |

Test 12 参考: ori_acc=79.29%, norm=99.11%, C=1.0, S₁=121.71s, S₈=44.09s, S∞=35.86s。

## 8. 验证命令

```bash
# 步骤 (a) sanity
SOAR_SPARSE_DENSE_LEN=524288 python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# 步骤 (b)
SOAR_SPARSE_DENSE_LEN=65536  python3 scripts/fcloud/fcloud_workflow.py restart-server
# ... 同样的 accuracy + speed

# 步骤 (c) — 仅 (b) 通过时
SOAR_SPARSE_DENSE_LEN=16384  python3 scripts/fcloud/fcloud_workflow.py restart-server
# ... 同样的 accuracy + speed
```

(`fcloud_workflow.py restart-server` 已经会 source `prepare_env.sh`,后者按 §4 处理 env 变量。)

## 9. 回滚

纯 Python 改动,单文件,env 门控:

```bash
git checkout python/sglang/srt/layers/attention/minicpm_backend.py
git checkout python/sglang/srt/layers/attention/minicpm_sparse_utils.py  # 如果也改了
git checkout benchmark/soar/demo_sala/prepare_env.sh
```

或者直接 `unset SOAR_SPARSE_DENSE_LEN`,行为回到 Test 12。

## 10. 后续建议

- 如果 CHANGE_0136 步骤 (b) 或 (c) 在速度+精度上都赢,自然的下一步是用 sparse-aware GPTQ calibration 重新量化,把安全阈值再压低(比如 dense_len=4096),把更多 workload 迁到 sparse 路径。
- 如果步骤 (b) 精度不过,那就证实了"GPTQ + sparse 任何阈值都会崩精度"这条历史结论,SOAR 提交关闭这条线。结论是:dense + GPTQ + FP8 KV (Test 12) 在当前量化质量下是全局最优。
- 与 CHANGE_0136 独立,如果 `--attention-backend flashinfer` (Round 13f-1 smoketest) 被证明只是 dense 路径的别名且无回退,可考虑把 `prepare_env.sh` 简化成官方默认。
