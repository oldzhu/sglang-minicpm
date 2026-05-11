# CHAT — NGRAM 探路定位 Stage 3b cuda-graph 前置修复点

- **日期**: 2026-05-11 10:50（开始）→ 10:55（已暂停）
- **分支 / HEAD**: `mixed_minicpm_cudagraph` @ `580d86582`
- **fcloud 实例**: `ai-e7e98a7c52`

## 1. 用户请求 / 决定

- 上一轮用户已批准跑 NGRAM 探路 —— 零代码改动，仅 `SOAR_SPEC_NGRAM=1 SOAR_SPEC_MEDUSA=0` —— 作为低成本去风险手段，用来验证我们这套运行时配置（stock flashinfer + GPTQ + FP8 KV + dense + mixed-chunk + torch.compile + 16 bs cuda-graph buckets）能否承载任何 TARGET_VERIFY 工作流，再决定是否动手写 Medusa Stage 3a 代码。
- 本会话指令：「破例一下，把 medusa 一直推到能在 fcloud 跑测试」。本次探路属于这条主线。

## 2. Agent 行动

1. 第一次 `restart-server` 报 `unrecognized arguments: --extra-env`，查源码确认正确 flag 是 `--env KEY=VAL`（action='append'，可重复）。重试成功：
   ```
   python3 scripts/fcloud/fcloud_workflow.py restart-server \
     --env SOAR_SPEC_MEDUSA=0 --env SOAR_SPEC_NGRAM=1
   ```
2. `wait-server` 两次 300s 超时 —— 查 server 日志。
3. **结果**：服务启动跑到 cuda-graph capture，在 bs=24 verify-shape bucket 崩溃：
   ```
   File "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 515, in _capture_metadata
       if forward_mode.is_target_verify() and spec_info.topk > 1:
   AttributeError: 'NgramVerifyInput' object has no attribute 'topk'
   ```
4. grep 确认 `MedusaInput`（`python/sglang/srt/speculative/medusa_info.py`）**同样既没有 `topk` 也没有 `draft_token_num`** —— 所以 Stage 3b（Medusa + cuda-graph）会撞到一模一样的崩溃。
5. 暂停 fcloud（`pause-instance` 第一次返回 504，5 秒后重试成功）。

## 3. 结论

- **探路 ROI 极高**：约 5 分钟 fcloud 时间，零新代码，准确定位了将来 Stage 3b cuda-graph 速度收益的拦路石。
- **Stage 3a（eager）不受影响** —— 崩溃发生在 cuda-graph capture 里，而 `SOAR_SPEC_MEDUSA_EAGER=1` 把 cuda-graph 关了，所以不会触发。可以安全地写。
- **Stage 3b 前置修复点已锁定**：`hybrid_linear_attn_backend.py` 第 515 / 570 / 575 行约 5 行的容错补丁 —— `getattr(spec_info, "topk", 1) > 1` 和 `getattr(spec_info, "draft_token_num", None) or <static_K>`。NGRAM + Medusa + 未来任何 linear-verify 算法都对称受益。风险低。
- **v23 提交包安全** —— Stage 2 medusa_worker.py 是纯 pass-through，运行时根本进不到 TARGET_VERIFY，所以 hybrid backend 的这个缺口对 v23 没影响。
- **cuda-graph 之前的基础设施已验证**：模型加载、KV 分配、hybrid pool 初始化、FP8 KV dtype、`--enable-fused-qk-norm-rope`、`--quantization gptq_marlin` —— 全部正常。Stage 3a 只需补上 draft+verify 逻辑。

## 4. 经验记录

- 「写 speculative 代码前先跑便宜的端到端探路」再次奏效 —— 与提案 §11（NotImplementedError 假警报）同样的教训，又被印证一次。
- 探路还顺手抓到一个内部工具 bug：agent 用了不存在的 `--extra-env`，应该是 `--env`。已在会话中修正用法。

## 5. 交叉引用

- 提案更新：`docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.{en,zh}.md` §12（新增）。
- 测试记录：`docs/soar_2026_changes/TEST_RESULTS_TRACKING.md` → 在 Medusa-Stage2-cgraph 之前新增 `NGRAM-probe` 一行。
- 记忆笔记：`/memories/repo/hybrid_backend_spec_info_shape.md`（新建）。

## 6. 下一步

- **Stage 3a 编码**（无基础设施阻塞）：把 `medusa_worker.py` 从 Stage 2 pass-through 改成 draft+verify，用 `SOAR_SPEC_MEDUSA_EAGER=1` 跑。目标：zero-init heads + greedy 下与 v22 字节等价 → ori_accuracy ≥ 80%。
- **Stage 3b**（3a 通过后）：给 `hybrid_linear_attn_backend.py` 打 5 行容错补丁，关 EAGER，测速。
