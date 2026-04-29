# 提案 — Round 13f-2:在显式 `flashinfer` 后端路径上恢复 Test 12 准确率

## 状态:PROPOSAL(待审批)

## 背景

Round 13f-1(`SOAR_BACKEND_VARIANT=flashinfer`,提交 `6a070110b`)结果:

| 指标 | Test 12 baseline | Round 13f-1 | Δ |
|---|---|---|---|
| ori_accuracy | 79.29% | **76.91%** | −2.38pt |
| 归一化 | 99.11%(C=1.0) | ~96.1% | C=0(淘汰) |
| S₁ | 121.71s | **110.76s** | −9.0% |
| S₈ | 44.09s | **40.50s** | −8.1% |
| S∞ | 35.86s | **33.66s** | −6.1% |

速度收益真实可观,但准确率落到 C=0 截断之下,当前配置无法直接提交。

研究文档
[`RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md`](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md)
已确认:

- Test 12(`--attention-backend minicpm_flashinfer --force-dense-minicpm`)
  在 `server_args.py:1525` 内部把 `attention_backend=flashinfer`,所以
  **Test 12 与 Round 13f-1 std-attn 跑的是同一份 stock FlashInfer kernel**。
- Lightning mixer `recurrent_threshold` 两路径均为 128(`prepare_env.sh:129`
  无条件 export `=128`)。
- 剩余结构差异是 `model_config.force_dense_minicpm`:
  - True(Test 12):`has_sparse_attention=False`、`sparse_layer_ids=[]`。
  - False(Round 13f-1):`has_sparse_attention=True`、layer ids 非空。

`has_sparse_attention` 在 6 处热路径被消费,影响 KV cache pool 类型、
scheduler chunking、每请求 slot 分配
(`schedule_batch.py:1473,1525,1983,2036`、
`model_runner_kv_cache_mixin.py:369,408`、
`minicpm_backend.py:222`)。stock FlashInfer 不读 compress_k1/k2,但该 flag 改变
了调度器 / KV pool 设置,进而改变批组成与数值。

## 假设

2.4pt 的准确率回归来自 `has_sparse_attention=True` 在 scheduler/KV pool 层的
副作用,**不是** std-attn kernel 也**不是** Lightning mixer。在保留字面后端
字符串 `flashinfer` 的同时恢复 `force_dense_minicpm=True`,应当:

- 在本地噪声(±2pt)内复现 Test 12 acc。
- 要么保住 Round 13f-1 的速度(则我们获得了一个"显式 `flashinfer` 字符串"的
  新可用 baseline);要么回归到 Test 12 速度(则速度收益归功于稀疏调度路径,
  而该路径会损害准确率,意味着整条 Round 13f 路线宣告失败)。

## 计划

单轮实验:**Round 13f-2** = Round 13f-1 重新加上 `--force-dense-minicpm`。

### 方案 1:最小化 `prepare_env.sh` patch(推荐)

为 `SOAR_BACKEND_VARIANT=flashinfer` 增加一个子开关,**不**清除
`FORCE_DENSE_ARG`。默认行为不变,通过 `SOAR_BACKEND_KEEP_FORCE_DENSE=1` 显式开启:

```bash
# diff 针对 benchmark/soar/demo_sala/prepare_env.sh,gptq 分支
	if [[ "$SOAR_BACKEND_VARIANT" == "flashinfer" ]]; then
		BACKEND_ARG=" --attention-backend flashinfer"
-		FORCE_DENSE_ARG=""
+		# Round 13f-2: 默认仍丢弃 --force-dense-minicpm(13f-1 行为)。
+		# 设置 SOAR_BACKEND_KEEP_FORCE_DENSE=1 则保留 --force-dense-minicpm,
+		# 让 model_config 暴露 has_sparse_attention=False / sparse_layer_ids=[],
+		# 用于隔离这两个 flag 是否拥有 13f-1 的 2.4pt acc 下降。
+		if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
+			# 保持 FORCE_DENSE_ARG = " --force-dense-minicpm"
+			# (来自 SOAR_SPARSE_MODE!=1 分支的设置)
+			:
+		else
+			FORCE_DENSE_ARG=""
+		fi
		DENSE_AS_SPARSE_ARG=""
-		echo "[prepare_env] SOAR_BACKEND_VARIANT=flashinfer -> using stock flashinfer backend, dropping --force-dense-minicpm and --dense-as-sparse"
+		echo "[prepare_env] SOAR_BACKEND_VARIANT=flashinfer KEEP_FORCE_DENSE=${SOAR_BACKEND_KEEP_FORCE_DENSE:-0} -> using stock flashinfer backend, FORCE_DENSE_ARG='${FORCE_DENSE_ARG}', dropping --dense-as-sparse"
	else
```

无需改动 sglang 源码。提交信息建议:`prep_env: 13f-2 add SOAR_BACKEND_KEEP_FORCE_DENSE switch`。

### 风险评估

- **启动风险**:极低。`--attention-backend flashinfer --force-dense-minicpm`
  组合**内部上**就是 Test 12 已有的运行状态(经 `server_args.py:1525` 重写)。
  唯一区别在于到达该状态的路径:Round 13f-2 走显式字符串,Test 12 走重写。
  任何读取 `attention_backend` 的代码都会看到 `"flashinfer"`(与 Test 12
  重写后一致)。无新代码路径被触达。
- **准确率风险**:低。最差情况 acc ≈ Test 12(79.29%)→ C=1.0。因为没有引入
  任何新组件,所以相对 Test 12 没有回归向量。
- **速度风险**:未知。如果 13f-1 的速度收益来自移除 `force_dense_minicpm` 的
  scheduler 副作用,那么 13f-2 会回归到 Test 12 速度。这仍然是一次成功的实验
  (告诉我们 `flashinfer` 字符串本身无关紧要,速度收益归属于稀疏调度路径,
  而该路径会损失准确率 → 整条 Round 13f 路线宣告失败)。

### 规则合规

- 不改模型量化;仍是 GPTQ + FP8_e5m2 KV。
- 不改 eval 脚本。
- 无架构改动;仅尝试一个此前未测过的 server flag 组合。
- 满足提交约束(≤2GB,≤5h,无 prefix cache)。

## 测试命令(用户批准后)

```bash
# 1. 同步 prepare_env.sh 改动到 fcloud
python3 scripts/fcloud/fcloud_workflow.py sync

# 2. 用新环境变量启动:
python3 scripts/fcloud/fcloud_workflow.py restart-server \
  --env SOAR_BACKEND_VARIANT=flashinfer \
  --env SOAR_BACKEND_KEEP_FORCE_DENSE=1
python3 scripts/fcloud/fcloud_workflow.py wait-server

# 3. 校验 cmdline:
python3 scripts/fcloud/fcloud_exec.py exec "pgrep -af sglang.launch_server"
# 必须包含: "--attention-backend flashinfer --force-dense-minicpm"
# 必须不含: "--dense-as-sparse" "minicpm_flashinfer"

# 4. 先跑 accuracy(gating):
python3 scripts/fcloud/fcloud_workflow.py accuracy
# 通过标准: ori_accuracy ≥ 78.5%(Test 12 本地噪声下界 ±0.8pt)

# 5. acc 通过后,逐个跑 S1/S8/Smax(--variant all 已知 bug):
python3 scripts/fcloud/fcloud_workflow.py speed --variant s1
python3 scripts/fcloud/fcloud_workflow.py speed --variant s8
python3 scripts/fcloud/fcloud_workflow.py speed --variant smax

# 6. 关机.
python3 scripts/fcloud/fcloud_workflow.py shutdown
```

## 结果决策矩阵

| 结果 | acc | S₁ | 决策 |
|---|---|---|---|
| **A** | ≥78.5% | ≤115s | **胜出 — 提升为 v20 候选。**Round 13f flashinfer 线成为新提交 baseline。更新 prepare_env.sh 默认 + 写入优化目录。 |
| **B** | ≥78.5% | ≥118s(Test 12 水平) | **中性。**速度收益完全归功于丢弃 force-dense;显式 flashinfer 字符串无额外作用。归档 Round 13f,回到 dense+GPTQ 优化方向。 |
| **C** | <78.5% | 任意 | **失败。**`has_sparse_attention=True/False` 切换不是 13f-1 2.4pt 回归的主因。下一步跑 Exp D(重开 compile),或先重跑一次 plain 13f-1 验证 76.91% 是否可复现(可能只是一次坏样本)。 |

## 回滚

变更是 opt-in(`SOAR_BACKEND_KEEP_FORCE_DENSE=1`),取消环境变量即恢复。
完全回滚:revert prepare_env.sh diff。

## 交叉引用

- 研究文档: [RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md](RESEARCH_flashinfer_vs_minicpm_flashinfer_codeflow.zh.md)
- 13f-1 chat: [chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md](chat/CHAT_round13f_change0136_validation_20260429_1410.zh.md)
- 13f-1 测试行: TEST_RESULTS_TRACKING.md `R13f1-flashinfer`。
- Parked 13f 兄弟工作: [CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md](CHANGE_0136_minicpm_sparse_dense_len_flag.zh.md)、 [CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md](CHANGE_0137_sparse_prefill_page_table_off_by_one.zh.md)。
- 代码锚点:
  - [python/sglang/srt/server_args.py](../../python/sglang/srt/server_args.py#L1521-L1525)
  - [python/sglang/srt/configs/model_config.py](../../python/sglang/srt/configs/model_config.py#L236-L248)
  - [python/sglang/srt/managers/schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py#L1473)
