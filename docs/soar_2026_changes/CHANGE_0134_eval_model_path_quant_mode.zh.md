# CHANGE_0134 — 修正 eval `--model_path` 按 `quant_mode` 选择；澄清稀疏路径激活规则

## 状态：已应用 + 已诊断

Workflow 修复已 push（commit `ae119f5aa`）。2026-04-29 上机诊断确认：
**Round 13e 超时不是 model_path 不匹配造成的。**详见下面
[《fcloud 上机验证（2026-04-29）》](#fcloud-上机验证2026-04-29)。
修复仍然保留（是干净卸载，防止以后两个模型目录出现行为差异时
静默踩坑）。

提交（待 push）：`scripts/fcloud/fcloud_workflow.py` —— 新增
`_resolve_model_path()` 与 `--quant-mode` / `--model-path` CLI 参数，
作用于 `accuracy`、`quick-accuracy`、`full` 三个子命令。

## 背景

Round 13e Test 1（BF16 + 稀疏 + FP8 KV + CHANGE_0133 已应用，
concurrency=32）在 85/150 处中止，伴随大量 3000 s 读超时。Decode
吞吐稳定在 ~295–330 tok/s（8 个长上下文请求，约 37 tok/s/req）。
用户提了两个问题：

1. 官方 toolkit 默认 `--attention-backend flashinfer` 且没有
   `--force-dense-minicpm`——是不是表示稀疏路径默认就是开的？
   稀疏是否「`--attention-backend minicpm_flashinfer` 或
   `--dense-as-sparse` 二者其一」即可激活？
2. fcloud 自动化在测 BF16 模型时，eval 命令仍然写死
   `--model_path GPTQ_MODEL`。这种 server 与 eval 分别加载不同模型
   会不会导致跑飞？

本文记录答复，并落地问题 #2 的修复。

## 稀疏路径激活规则（澄清）

MiniCPM-SALA 模型上稀疏路由开启需**同时**满足：

1. `--attention-backend minicpm_flashinfer`（或
   `minicpm_flashattn`）。只有自定义 `MiniCPMAttentionBackend` 知道
   `mixer_types == ["sparse_attention", "lightning_attn"]` 这种分层
   分发。stock `flashinfer` / `triton` / `flash_attn` backend 都不会
   按 `is_sparse_layer` 分支，所以那 8 个稀疏层会被当作普通 dense
   attention 跑。参考：
   [`python/sglang/srt/layers/attention/minicpm_backend.py`](../../python/sglang/srt/layers/attention/minicpm_backend.py)
   L335 `elif attention_backend == "minicpm_flashinfer":`。
2. **不**带 `--force-dense-minicpm`。这个开关做两件事：
   - 把 `attention_backend` 由 `minicpm_flashinfer` 重写为
     `flashinfer`（参考 CHANGE_0070 / CHANGE_0131），并且
   - 在 config 层把 `has_sparse_attention=False`、清空
     `sparse_layer_ids`（参考 CHANGE_0132 §「Finding 4」）。

`--dense-as-sparse` **不是激活开关**。它只在稀疏已经激活（同时满足
规则 1 + 2）的前提下生效：在稀疏后端内部把 `self.dense_len = 0`
（默认是 `hf_config.sparse_dense_len`），意思是连那些原本会短路成
dense 的小批次也走稀疏 compute 路径。它扩大稀疏 compute 的覆盖率，
不开稀疏。

### 对官方 toolkit 默认值的含义

toolkit 页面上默认 `SGLANG_SERVER_ARGS = --disable-radix-cache
--attention-backend flashinfer --chunked-prefill-size 32768`。
用 `--attention-backend flashinfer`（违反规则 1），**官方默认配置
对稀疏层等价于 dense**。所以「原始模型 + 默认参数在 fcloud 跑得
顺畅」并不能反驳 Round 13e 的超时——官方 baseline 也没在跑自定义
稀疏 backend。

## 问题 #2 的修复：eval `--model_path` 不匹配

### Bug

`eval_model_001.py --model_path X` 在客户端被 eval harness 用于：

- 加载 tokenizer（提示词 / chat template 渲染），以及
- 加载 `GenerationConfig` 推导 `eos_token_id` → `stop` 词。

它**不**告诉 server 应该 serve 哪个权重。
`fcloud_workflow.py.step_accuracy()` 此前固定
`--model_path {MODEL_PATH}`（GPTQ 路径），不管 server 实际跑的是哪个。
Round 13e server 跑的是非量化模型
（`/root/models/openbmb/MiniCPM-SALA`），但 harness 从 GPTQ 目录
（`/root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8`）
读 tokenizer + GenerationConfig。

如果两个目录的 `tokenizer_config.json`、`chat_template.jinja`、
`generation_config.json` 不一致（很可能不一致——`preprocess_model.py`
在 GPTQ 构建里可能改写 chat template / generation config），结果：

- chat template 错 → server 看到的 prompt 格式错 → 模型不会按预期
  emit EOS → 一路写到 `max_tokens=65536` → 3000 s 读超时（这正是
  Round 13e Test 1 看到的现象）。
- stop tokens 错 → 同上。
- special tokens / BOS 处理错 → 静默掉点。

不论稀疏 attn 本身是否慢，这都是一个真 bug。

### 修复

`scripts/fcloud/fcloud_workflow.py`：

1. 新增 helper `_resolve_model_path(quant_mode, model_path)`，与
   `step_restart_server()` 同样的方式从 `*_MODEL_PATH` 常量挑路径。
2. `step_accuracy()` 与 `step_quick_accuracy()` 加上 `quant_mode` /
   `model_path` 形参，先解析出 `eval_model_path` 再传给
   `--model_path`。
3. `workflow_full()` 接收 `quant_mode` / `model_path`，分别透传给
   `step_restart_server()` 与 `step_accuracy()`。
4. CLI：`accuracy`、`quick-accuracy`、`full` 三个子命令加
   `--quant-mode {gptq,fp8_blockwise,noquant}` 与 `--model-path PATH`。
   默认仍是 `gptq`，保持对当前 dense 提交基线（Test 12）的
   向后兼容。

### 验证

逻辑层：
- `python3 scripts/fcloud/fcloud_workflow.py accuracy --help` 列出新
  参数。
- `python3 scripts/fcloud/fcloud_workflow.py full --help` 列出新
  参数。

实操（下次上机，重测前先做 diff）：

```bash
diff -u /root/models/openbmb/MiniCPM-SALA/tokenizer_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/tokenizer_config.json
diff -u /root/models/openbmb/MiniCPM-SALA/generation_config.json \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/generation_config.json
diff -u /root/models/openbmb/MiniCPM-SALA/chat_template.jinja \
        /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8/chat_template.jinja 2>/dev/null || true
```

任意一个 diff 非空 → Round 13e 超时至少部分由 tokenizer 不匹配解释，
值得用修复后的 workflow 重测一次。三个 diff 全为空 → 排除 tokenizer
不匹配，超时确实是稀疏 attn 自身慢。

## 下一步（提议）

1. （本次提交）push workflow 修复。
2. 用户启动 fcloud。
3. 跑上面三条 diff 确认是否存在 tokenizer 不匹配。
4. 若 diff 非空 → 用修复后的 workflow 重跑 Round 13e Test 1
   （`accuracy --quant-mode noquant`），CHANGE_0133 已应用。
   - 通过条件：1 h 内跑完，accuracy ≥ 78 %，且没有读超时。
5. 若 diff 为空 → 关闭稀疏线方向，记录最终结论，继续 dense 路径
   优化。

## 风险

- **对 dense 路径无风险** —— 修改只在自动化脚本层；
  默认 `--quant-mode gptq` 保持旧行为，所有已存在的 dense 测试不变。
- **对提交包无风险** —— `eval_model_001.py` 是官方 harness 的本地
  副本，**不**进提交 tarball。我们也**没有**改 eval 脚本（遵守
  eval 脚本完整性规则）。

## 回滚

```bash
git checkout scripts/fcloud/fcloud_workflow.py
```

## 交叉引用

- CHANGE_0070 —— `--force-dense-minicpm` 把 `minicpm_flashinfer`
  重写为 `flashinfer`。
- CHANGE_0131 —— KV4 backend 白名单与 `force_dense_minicpm` 绕过。
- CHANGE_0132 —— `force_dense_minicpm` 在 config 层对
  `has_sparse_attention` 的影响。
- CHANGE_0133 —— 稀疏 decode `compress_k1/k2` 过填修复（本轮已应用；
  必要但不充分，不足以让稀疏路径在 concurrency=32 长上下文下可用）。

## fcloud 上机验证（2026-04-29）

在实例上跑了三个 diff 与一个端到端的 tokenizer 对比。

### 文件级别的 diff

`tokenizer_config.json` 不一致：

- GPTQ 多了 `"add_prefix_space": null`。
- GPTQ 多了 `"extra_special_tokens": {}`。
- `"tokenizer_class"` 由 `LlamaTokenizer`（BF16）变为
  `LlamaTokenizerFast`（GPTQ）。
- GPTQ JSON 里**删了** `"chat_template"` 键。GPTQ 目录下另有一个
  独立的 `chat_template.jinja`。
- GPTQ 末尾多了 `"_commit_hash": null`。

`generation_config.json` 不一致：

- GPTQ 多了 `"do_sample": true`。
- `transformers_version` 4.56.1 → 4.57.1。
- `eos_token_id`、`pad_token_id` 不变。

`chat_template.jinja`：BF16 目录**没有**这个文件（模板嵌在
`tokenizer_config.json` 里）；GPTQ 目录**有**这个独立文件。

### 行为层对比

- BF16 嵌入的 chat template vs GPTQ 独立 `chat_template.jinja`：
  **字节级别一致**（sha256 `accdbc3c45c7ee51`，都是 8 803 字节）。
- 两个目录执行 `AutoTokenizer.from_pretrained(...)` 都返回
  `LlamaTokenizerFast` 实例（HF 自动把 BF16 的 `LlamaTokenizer`
  声明升级为 Fast）。
- `eos_token_id`、`eos_token`、`bos_token`、推导出的 `stop_words`
  （`['</s>', '<|im_end|>']`）：完全一致。
- 同一个示例 prompt 走 `apply_chat_template`：字节级别一致（76 字符），
  `tok.encode(...)` 结果也完全一致（17 个 token id）。
- `eval_model_001.py` 在初始化时 pop 掉 `do_sample`
  （`self.generation_kwargs.pop('do_sample', None)`），所以 GPTQ 多
  出的 `do_sample: true` 对 harness 无影响。

### 结论

workflow 里的 model_path 不匹配是个真 bug（修复保留），但**它不能
解释** Round 13e Test 1 的长上下文超时：两个目录在 tokenizer、
chat template、EOS / stop-word 推导上目前表现完全一致。

因此 Round 13e Test 1 在 concurrency=32 下的卡顿与读超时是**稀疏
attention 在长上下文（32K–128K）上本身就慢**造成的，不是
tokenizer / template 漂移。**仅靠 CHANGE_0134、原配置重测几乎不会改变
结果**（没有新证据）。待定选项：

- 选项 A：暂关稀疏线，继续 dense 路径优化（现在提交基线 Test 12
  仍是最佳）。
- 选项 B：在再投人之前，先 profile 稀疏 decode kernel（top-k 评分 +
  稀疏 FlashAttention）在 bs=8 × max_seq=128K 的热点。
- 选项 C：调低 `max-running-requests`（例如 4 而非 8）试稀疏，
  看看每步代价是否随 bs 亚线性，以及低并发下的稀疏能不能超过 dense。
