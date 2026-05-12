# CHANGE_0162 — Force-Dense 默认值：K1/K2 池崩溃类的结构性修复

**类型**: 结构性 / 配置  
**状态**: 已验证 — 2026-05-12  
**优先级**: 高（阻断 MEDUSA 崩溃类，零成本）  
**相关变更**: CHANGE_0158、CHANGE_0159、CHANGE_0161（同一根本原因的补丁级修复）

---

## 1. 背景与动机

MEDUSA Stage 3a（K=1 零初始化推测解码）在速度基准测试期间崩溃，报错
`available_size > max_total_num_tokens`。根本原因追溯至
`req_to_sparse_k1_token` / `req_to_sparse_k2_token` 数组保留了上一个更长请求
的过期非零槽位 ID。CHANGE_0161 在补丁层面修复了此问题（在 `cache_finished_req`
释放后将 K1/K2 行置零）。

然而，更深层的架构问题浮现：**既然 flashinfer 后端从不使用 K1/K2 表，为何在
使用 flashinfer 时仍然分配这些表？**

### 池类型选择路径

```python
# python/sglang/srt/model_config.py:238
def has_sparse_attention(self):
    return getattr(self.hf_config, "has_sparse_attention", False) \
           if not self.force_dense_minicpm else False

# python/sglang/srt/model_runner_kv_cache_mixin.py:369
if self.minicpm_hybrid_config is not None:
    if self.model_config.has_sparse_attention:
        self.req_to_token_pool = MiniCPMHybridReqToTokenPool(...)  # 含 K1/K2 表
    else:
        self.req_to_token_pool = HybridReqToTokenPool(...)          # 无 K1/K2 表
```

当 **不存在** `--force-dense-minicpm` 时：
- `force_dense_minicpm=False` → `has_sparse_attention=True`（来自 hf_config）
- 池类型：`MiniCPMHybridReqToTokenPool` — 分配 K1/K2 表
- K1/K2 表**被分配**，尽管 flashinfer 从不写入它们

当 **存在** `--force-dense-minicpm` 时：
- `force_dense_minicpm=True` → `has_sparse_attention=False`
- 池类型：`HybridReqToTokenPool` — 无 K1/K2 表
- `cache_finished_req` 中的 `isinstance(pool, ...)` → `False`
- K1/K2 分支**永远不执行** → 崩溃类在结构上不可能发生

### force-dense 为何缺失

当 v20（Round 13f）采用 `SOAR_BACKEND_VARIANT=flashinfer` 时，`prepare_env.sh`
的 flashinfer 分支默认清除了 `FORCE_DENSE_ARG=""`。理由是："stock flashinfer 不
使用 minicpm 后端，因此 force-dense 无关紧要。"该决定对池类型选择
（`has_sparse_attention` → 池分配 → K1/K2）的副作用被忽视了。

---

## 2. 规则合规说明

- 模型权重未更改
- 影响准确性的代码未更改
- 仅更改 `prepare_env.sh` 的默认环境变量（`SOAR_BACKEND_KEEP_FORCE_DENSE=1`）
- `--force-dense-minicpm` 标志在 Stage 2 直通提交和生产基线中已使用——此处
  恢复该行为
- 回滚方式：设置 `SOAR_BACKEND_KEEP_FORCE_DENSE=0`

---

## 3. 实施计划（变更前）

1. 在 `prepare_env.sh` 中添加
   `export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"`
2. flashinfer 分支中现有的条件判断已正确处理此情况：
   ```bash
   if [[ "$SOAR_BACKEND_KEEP_FORCE_DENSE" == "1" ]]; then
       # 保留上面设置的 FORCE_DENSE_ARG (" --force-dense-minicpm")
       :
   else
       FORCE_DENSE_ARG=""
   fi
   ```
   默认值为 `1` 时，`FORCE_DENSE_ARG` 被保留。

---

## 4. 实际代码变更

### `benchmark/soar/demo_sala/prepare_env.sh`

在 `SOAR_BACKEND_VARIANT` 导出行之后添加 `SOAR_BACKEND_KEEP_FORCE_DENSE` 默认
导出（含注释块共 18 行）：

```bash
# CHANGE_0162 (2026-05-12): 默认 SOAR_BACKEND_KEEP_FORCE_DENSE=1，
# 确保 flashinfer 后端始终启用 --force-dense-minicpm。
# 结构性修复路径：--force-dense-minicpm → model_config.has_sparse_attention=False
# → HybridReqToTokenPool（无 req_to_sparse_k1_token）→ K1/K2 崩溃类结构上不可能。
export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"
```

---

## 5. 验证命令

```bash
# 1. 验证 prepare_env.sh 在 SGLANG_SERVER_ARGS 中输出 --force-dense-minicpm
source benchmark/soar/demo_sala/prepare_env.sh 2>&1 | grep "SGLANG_SERVER_ARGS"
# 预期：包含 "--force-dense-minicpm"

# 2. 验证服务器以 force_dense_minicpm=True 启动
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py wait-server
python3 scripts/fcloud/fcloud_exec.py exec 'grep -i "force_dense" /tmp/sglang_server.log | head -3'

# 3. 准确性测试
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 4. 速度测试
python3 scripts/fcloud/fcloud_workflow.py speed --variant all

# 5. 回滚测试
SOAR_BACKEND_KEEP_FORCE_DENSE=0 source benchmark/soar/demo_sala/prepare_env.sh 2>&1 | grep "SGLANG_SERVER_ARGS"
# 预期：不包含 "--force-dense-minicpm"（旧 flashinfer 行为）
```

---

## 6. 结果汇总

| 指标 | Stage3a（无 force-dense） | Stage3a-force-dense | 变化 |
|------|--------------------------|---------------------|------|
| 崩溃 | 0（CHANGE_0161 后） | **0** | — |
| K1/K2 已分配 | 是（MiniCPMHybridReqToTokenPool） | **否**（HybridReqToTokenPool） | 结构性 |
| S1 | 202.96s | **202.70s** | −0.1%（噪声） |
| S8 | 61.65s | **61.60s** | −0.1%（噪声） |
| Smax | 43.40s | **43.29s** | −0.3%（噪声） |
| 准确性（原始） | 76.04% | **77.87%** | +1.83pt（噪声范围） |
| 准确性（归一化） | ~95.05% | **~97.34%** | C=0 → C=0.92 |
| C | 0 | **0.92** | 改善 |

**关键发现**：将 `--force-dense-minicpm` 设为 flashinfer 后端的默认值，对速度
**零可测量影响**，同时通过从根本上防止 K1/K2 表被分配，**消除了整个 K1/K2
崩溃类**。

> 注：Stage 3a 速度（202/61/43s）仍慢于 Stage 2 基线（118/43/35s）。这是预期
> 的——K=1 零初始化 Medusa heads 的 accept_rate=0（每个草稿都被拒绝）。
> Stage 3b（训练好的 heads）是实现加速的必要条件。

---

## 7. 与 CHANGE_0158/0159/0161 的关系

| 修复 | 类型 | 何时生效 |
|------|------|---------|
| CHANGE_0158 | 补丁：K1/K2 释放的 `ne(0)` 过滤器 | 存在 `MiniCPMHybridReqToTokenPool` 时 |
| CHANGE_0159 | 补丁：主 KV 释放的 `ne(0)` 过滤器 | 始终 |
| CHANGE_0161 | 补丁：释放后对过期 K1/K2 行置零 | 存在 `MiniCPMHybridReqToTokenPool` 时 |
| **CHANGE_0162** | **结构性：完全防止 K1/K2 分配** | flashinfer 后端 |

启用 CHANGE_0162 后，CHANGE_0158 和 CHANGE_0161 成为死代码
（`isinstance(pool, MiniCPMReqToTokenPool | MiniCPMHybridReqToTokenPool)` 分支
永远不会触发）。CHANGE_0159（主 KV 零过滤器）仍然有效且有用。

---

## 8. 回滚说明

```bash
# 选项 1：环境变量覆盖（仅用于测试，非持久性）
export SOAR_BACKEND_KEEP_FORCE_DENSE=0

# 选项 2：在 prepare_env.sh 中恢复默认值
# 将：export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-1}"
# 改为：export SOAR_BACKEND_KEEP_FORCE_DENSE="${SOAR_BACKEND_KEEP_FORCE_DENSE:-0}"
```

---

## 9. 后续步骤

1. **Stage 3b：训练好的 Medusa heads** — 实现相对 Stage 2 基线实际加速的必要条件。
   K=1 零初始化 heads 的 accept_rate=0，Stage 3a 比 Stage 2 慢约 70%。训练好的
   heads 目标 accept_rate≥0.60 将得到 `spec_accept_length≈1.60` → 相比非推测
   解码约 20-30% 的加速。
2. **mcq 准确性**：Stage 3a 两种配置均显示 mcq≈43-47%，低于 Stage 2 的约 57%。
   根本原因正在调查中——可能是 MEDUSA 验证开销与长思考链上 max_tokens 计数的
   相互作用。
3. **官方提交**：带 CHANGE_0162 的 Stage 3a 仍不是可行的提交候选
   （速度慢于 Stage 2 + C=0.92）。提交 Stage 2 直通版本（commit `46553947b`）
   或继续推进 Stage 3b。
