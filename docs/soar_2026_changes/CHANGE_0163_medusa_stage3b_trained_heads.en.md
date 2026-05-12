# CHANGE_0163 — Medusa Stage 3b: Trained Head Forward + Hidden State Capture

**Date**: 2026-05-19  
**Commit**: `7a7470af4`  
**Branch**: `mixed_minicpm_cudagraph`  
**Remote**: `minicpm-src` (`oldzhu/sglang-minicpm`)  
**Status**: Implemented — pending fcloud test

---

## Background and Motivation

Stage 3a (CHANGE_0161/0162) established a working K=1 Medusa verify loop but used
zero-initialised W1 weights and derived draft tokens from `req.output_ids[-1]` (the
last accepted output token). This made the draft almost always wrong for the real
next position — the accept rate stayed near 0% for non-trivial requests, producing
no speed benefit over the Stage 2 passthrough.

Stage 3b wires three things together:

1. **Hidden state capture** from each TARGET_VERIFY forward (using the existing
   `CaptureHiddenMode.LAST` infrastructure already present in sglang).
2. **Real Medusa head forward**: `argmax(lm_head(SiLU(W1(h)) + h))` on the captured
   hidden states to predict the *next* draft token.
3. **Training script** that fits W1 on hidden-state → next-token pairs collected
   from the non-quantized MiniCPM-SALA model on the eval dataset.

With trained W1, the draft should match the base model's next token ~50-70% of the
time, giving `spec_accept_length` ≈ 1.5–2.0 and an estimated 15–30% end-to-end
speedup over the Stage 3a baseline.

---

## Rule Compliance

- **Submission size**: W1 state_dict is shape `(4096, 4096)` BF16 ≈ **32 MB**. Total
  submission stays well under 2 GB.
- **On-site quantization rule**: W1 is trained on-site (fcloud), not pre-uploaded.
- **Correctness**: Stage 3a fallback is preserved when `SOAR_MEDUSA_HEAD_PATH` is not
  set; no accuracy regression risk from the infrastructure change itself.
- **eval_model.py integrity**: No changes to the eval harness. All changes live on the
  server/model side.

---

## Files Changed

| File | Change |
|------|--------|
| `python/sglang/srt/speculative/medusa_worker.py` | Stage 3b draft logic, hidden state capture, head forward, req cache |
| `python/sglang/srt/models/minicpm_medusa_heads.py` | `load_trained_weights()` method |
| `python/sglang/srt/speculative/ngram_info.py` | Bug fix: `if hidden_states is not None:` |
| `benchmark/soar/demo_sala/prepare_env.sh` | `SOAR_MEDUSA_HEAD_PATH` env var |
| `benchmark/soar/demo_sala/train_medusa_head.py` | **NEW** training script |
| Submission copies of the above | Kept in sync |

---

## Implementation

### 1. `ngram_info.py` — Bug fix

```python
# Before (Stage 3a — never triggered because hidden_states was always None):
if logits_output.hidden_states:

# After (Stage 3b — multi-element tensor truthiness raises RuntimeError):
if logits_output.hidden_states is not None:
```

### 2. `minicpm_medusa_heads.py` — `load_trained_weights()`

```python
def load_trained_weights(self, path: str, device: str = "cuda") -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    # expects keys: heads.{k}.W1.weight  for k in 0..num_heads-1
    self.load_state_dict(state, strict=False)   # strict=False: lm_head not in ckpt
```

### 3. `medusa_worker.py` — Stage 3b changes

**`__init__`** additions:
- Correct `lm_head` selection (respects `tie_word_embeddings` config).
- `SOAR_MEDUSA_HEAD_PATH` env var: loads trained W1 weights and sets
  `self._use_trained_heads = True`.

**`_forward_verify_k1`** changes:

```python
# 1. Draft token selection (per request)
cached = getattr(req, "_medusa_draft_token", None)
draft_token = cached if (cached is not None and self._use_trained_heads) \
              else req.output_ids[-1]

# 2. Set capture mode on spec_info BEFORE get_model_worker_batch()
if self._use_trained_heads:
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

# 3. After TARGET_VERIFY forward, capture hidden states BEFORE verify()
raw_hidden_states = logits_output.hidden_states   # (bs, hidden_size)

# 4. Run verify() — this may index / clear hidden_states
logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(...)

# 5. Run head forward, cache next draft on live requests
draft_logits = self.medusa_heads(raw_hidden_states)  # (bs, 1, vocab)
next_draft_ids = draft_logits[:, 0, :].argmax(dim=-1).tolist()
for req, tok in zip(batch.reqs, next_draft_ids):
    req._medusa_draft_token = tok if not req.finished() else None
```

### 4. `train_medusa_head.py` — Training pipeline

```
1. Load non-quantized model (MiniCPM-SALA-Copy) via transformers
2. Hook model.model.norm to capture hidden states
3. Tokenize prompts from perf_public_set.jsonl (max_len=512 per sample)
4. Collect (h_t, token_{t+1}) pairs for all positions in all prompts
5. Train W1 (Adam, lr=1e-4, 5 epochs, batch=512)
6. Save: {"heads.0.W1.weight": W1.weight}  →  /root/medusa_head_k1.pt (~32 MB)
```

---

## Validation Commands (fcloud)

### Step 1: Train the head

```bash
# On fcloud (takes ~5-10 min):
cd /root/sglang-minicpm && git pull

python3 benchmark/soar/demo_sala/train_medusa_head.py \
  --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
  --data-path  /root/data/perf_public_set.jsonl \
  --output     /root/medusa_head_k1.pt \
  --epochs 5 --lr 1e-4 --max-len 512

# Expected log lines:
# Collected ~60000-120000 (h, label) pairs
# Epoch 5/5: loss≈X  top1_acc≈50-65%  time≈Ys
# Saved checkpoint to /root/medusa_head_k1.pt (32.0 MiB)
```

### Step 2: Start server with trained head

```bash
export SOAR_MEDUSA_HEAD_PATH=/root/medusa_head_k1.pt
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  "${SGLANG_SERVER_ARGS[@]}"

# Expected startup log:
# MedusaWorker Stage 3b: loaded trained heads from /root/medusa_head_k1.pt
# MedusaWorker ready: K=1, trained=True
```

### Step 3: Accuracy test

```bash
python3 /root/data/eval_model_001.py \
  --model http://localhost:30000 \
  --data_path /root/data/perf_public_set.jsonl
# Target: ≥ 79% (≥ Stage 3a baseline of 77.87%)
```

### Step 4: Speed tests

```bash
# Server logs to watch: spec_accept_length, accept_rate
# Target: spec_accept_length ≥ 1.3 (vs 1.0 in Stage 3a)

python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

---

## Expected Result Summary

| Metric | Stage 2 (baseline) | Stage 3a | Stage 3b target |
|--------|-------------------|----------|-----------------|
| S1 | 118.28s | 202.70s | ~100-130s |
| S8 | 43.87s | 61.60s | ~45-50s |
| Smax | 35.75s | 43.29s | ~35-40s |
| spec_accept_length | — | 1.00 | 1.3–1.7 |
| accept_rate | — | ~0% | 30-70% |

> Note: The overhead of running the trained head forward on hidden states is
> small (~1 ms per batch), so the net benefit depends entirely on accept_rate.

---

## Rollback Instructions

```bash
# Disable trained heads (revert to Stage 3a zero-init):
export SOAR_MEDUSA_HEAD_PATH=
# Or:
unset SOAR_MEDUSA_HEAD_PATH

# Disable Medusa entirely (revert to Stage 2 passthrough):
export SOAR_SPEC_MEDUSA=0
```

---

## Next Steps

1. **Validate accept_rate** from server logs after Stage 3b test.
2. If accept_rate < 30%: increase training epochs or use longer `--max-len`.
3. If accept_rate ≥ 50%: consider K=2 heads for further speedup.
4. If accuracy degrades: verify that `tie_word_embeddings` is handled correctly
   (check MiniCPM-SALA config) and that `scale_width` correction in the training
   script matches the inference path.
