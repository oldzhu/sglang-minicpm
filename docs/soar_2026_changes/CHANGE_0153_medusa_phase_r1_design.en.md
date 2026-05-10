# CHANGE_0153 — Medusa Phase R1 Design (plumbing spike)

Status: **DESIGN — awaiting approval before code changes**
Date: 2026-05-10
Branch: `mixed_minicpm_cudagraph`
Parent: [PROPOSAL_medusa_minicpm_sala_001.en.md](PROPOSAL_medusa_minicpm_sala_001.en.md)
Companion: [RESEARCH_speculative_decoding_survey_001.en.md](RESEARCH_speculative_decoding_survey_001.en.md)

## 0. TL;DR

Two findings from a deep read of the sglang speculative subsystem materially **simplify R1 vs the original proposal**:

1. **The "GLA-fork problem" is already solved upstream for Mamba2/GDN.** `HybridLinearAttnBackend.update_mamba_state_after_mtp_verify` (hybrid_linear_attn_backend.py L1373–1440) writes a per-step intermediate state into `mamba_pool.intermediate_ssm[layer, request, step]` during verify, then after `verify_tree_greedy` returns the accepted prefix length per request, scatters the right step's state back into the persistent `ssm_states` cache. We need to **port the same pattern to `SimpleGLAAttnBackend`** — not invent one. This eliminates the largest design risk in the original proposal (R2 GLA-fork).
2. **Tree-verify infra is fully reusable.** `eagle_utils.build_tree_kernel_efficient` + `verify_tree_greedy_func` operate on draft topologies regardless of who produces the candidates. Medusa's tree (one root, K levels, top-s candidates per level) is a special case of EAGLE's tree topology.

Net effect: R1 becomes ~600–800 LOC (not 1500–2000) and concentrates the engineering on a single new mechanism — **`SimpleGLAAttnBackend.update_simple_gla_state_after_verify`** — plus glue.

## 1. Architecture (R1)

```
                  ┌─────────────────────────────────────┐
                  │ MedusaWorker (TpModelWorker subclass)│
                  └──────────────────┬──────────────────┘
                                     │ forward_batch_generation()
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ 1. main fwd → hidden + base logits  │
                  │ 2. K Medusa heads → K candidate sets│
                  │ 3. build_tree_kernel_efficient      │
                  │    (reuse from eagle_utils)         │
                  │ 4. main verify-fwd over tree        │
                  │ 5. verify_tree_greedy → accepted L  │
                  │ 6. update_simple_gla_state_after... │
                  │    (NEW: scatter accepted state)    │
                  └─────────────────────────────────────┘
```

R1 deliberately keeps **K=1, top_s=1, num_draft_tokens=2** (1 base + 1 draft) → tree degenerates to a chain. This isolates the state-fork mechanism from tree-verify edge cases. R2 will scale K and top_s after R1 is byte-identity-clean.

## 2. New & modified files (concrete diff plan)

### 2.1 New files

```
python/sglang/srt/speculative/medusa_info.py                   ~150 LOC
python/sglang/srt/speculative/medusa_worker.py                 ~350 LOC
python/sglang/srt/models/minicpm_medusa_heads.py               ~120 LOC
```

**`medusa_info.py`** — datastructures:
```python
class MedusaInput(SpecInput):
    """Output of K Medusa heads, ready for tree-verify."""
    spec_input_type = SpecInputType.MEDUSA_VERIFY  # NEW enum

    draft_token_ids: torch.Tensor      # (bs, num_draft_tokens)
    parent_index: torch.Tensor          # (bs, num_draft_tokens) — parent node in tree
    retrieve_index: torch.Tensor        # (bs, num_draft_tokens) — flatten index
    tree_mask: torch.Tensor             # (bs, num_draft_tokens, num_draft_tokens)
    positions: torch.Tensor             # (bs, num_draft_tokens) — abs positions
    accept_threshold: float = 1.0       # R1 default = byte-identity gate

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (self.draft_token_ids.shape[1], 1)


@dataclass
class MedusaVerifyOutput:
    verified_id: torch.Tensor                    # (sum_accepted_per_req,)
    accept_length_per_req_cpu: List[int]
    last_hidden_state: torch.Tensor              # (bs, hidden) for next-step heads
```

**`medusa_worker.py`** — control flow per generation step:
```python
class MedusaWorker(TpModelWorker):
    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        # 1. Run main model forward over current input
        logits, hidden = self.target_worker.forward_with_hidden(batch)
        # 2. K Medusa heads → top-s tokens each
        head_logits = self.heads(hidden[:, -1])     # (bs, K, vocab)
        # 3. Build tree (chain at K=1) using eagle_utils
        spec = self._build_tree(head_logits, hidden)
        # 4. Verify forward through main model with tree mask + positions
        verify_logits, verify_hidden = self.target_worker.verify(batch, spec)
        # 5. Greedy verify (R1: accept_threshold=1.0 byte-identity)
        verified_id, accept_len = verify_tree_greedy_func(...)
        # 6. Scatter accepted-prefix GLA state back into persistent cache
        self.attn_backend.update_simple_gla_state_after_verify(
            accepted_steps=accept_len, ...
        )
        return GenerationBatchResult(next_token_ids=verified_id, ...)
```

**`minicpm_medusa_heads.py`** — head module:
```python
class MedusaHeads(nn.Module):
    """K residual MLPs producing logits via the SHARED main model lm_head.

    Per-head:  p^(k) = softmax( lm_head( SiLU(W1^k · h) + h ) )
    With W1 zero-init (paper §3.2), at random init each head ≡ base model,
    giving accept_rate = 1.0 trivially → R1 byte-identity gate passes.
    """
    def __init__(self, hidden_size, num_heads, lm_head_module):
        super().__init__()
        self.num_heads = num_heads
        self.W1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(num_heads)
        ])
        self.lm_head = lm_head_module       # WEIGHT-SHARED; not stored separately

    def forward(self, h):                   # h: (B, hidden)
        outs = []
        for k in range(self.num_heads):
            outs.append(self.lm_head(F.silu(h @ self.W1[k]) + h))
        return torch.stack(outs, dim=1)     # (B, K, vocab)
```

**Why this keeps R1 trivially correct:** with `W1=0`, `SiLU(0·h) + h = h`, so each head's output is exactly `lm_head(h)` — identical to the base model's argmax. At `accept_threshold=1.0`, the verifier accepts iff draft == base argmax → **always**. This is the byte-identity gate that R1's correctness rests on.

### 2.2 Modified files

| File | Change | LOC |
|---|---|---|
| `python/sglang/srt/speculative/spec_info.py` | Add `MEDUSA = auto()` to `SpeculativeAlgorithm`; `is_medusa()`; `create_worker()` branch; add `MEDUSA_VERIFY` to `SpecInputType` | +20 |
| `python/sglang/srt/server_args.py` | Accept `--speculative-algorithm MEDUSA`; `--speculative-num-medusa-heads K` (default 1) | +15 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | Add `SimpleGLAAttnBackend.update_simple_gla_state_after_verify` (mirror of `update_mamba_state_after_mtp_verify`); init `intermediate_ssm` buffer for SimpleGLA in `init_cuda_graph_state` | +110 |
| `python/sglang/srt/models/minicpm.py` | When `SOAR_SPEC_MEDUSA=1`, instantiate `MedusaHeads` on top of `MiniCPMSALAForCausalLM`; expose `forward_with_hidden` returning the pre-norm last hidden state | +40 |
| `python/sglang/srt/managers/tp_worker.py` | Plumb `forward_with_hidden` through worker layer | +15 |
| `benchmark/soar/demo_sala/prepare_env.sh` | Opt-in env: `SOAR_SPEC_MEDUSA=0` (default), when 1 append `--speculative-algorithm MEDUSA --speculative-num-draft-tokens 2 --speculative-num-medusa-heads 1` | +15 |
| `benchmark/soar/demo_sala/preprocess_model.py` | When heads weights file is present in model dir, copy under output; otherwise no-op | +20 |

**Total**: ~635 LOC across 3 new + 7 modified files.

## 3. The state-scatter mechanism (the core technical work of R1)

### 3.1 Existing reference: `update_mamba_state_after_mtp_verify` (Mamba2/GDN)

For each layer:
- During **verify forward** the backend writes the per-step state into `intermediate_ssm[layer, request, step]` (shape `(num_layers, max_requests, max_draft_tokens, state_dim)`).
- After `verify_tree_greedy` produces `accepted_steps[req] ∈ {-1, 0, 1, ..., K-1}`:
  - For each request with `accepted_steps[req] >= 0`, scatter `intermediate_ssm[:, req, accepted_steps[req]]` → `ssm_states[:, req_pool_idx[req]]`.
  - Requests where every draft token rejected (steps == -1) keep their original state (the one before draft entered) — naturally correct because we never overwrote it.

### 3.2 R1 port: `update_simple_gla_state_after_verify`

```python
class SimpleGLAAttnBackend(MambaAttnBackendBase):
    def update_simple_gla_state_after_verify(
        self,
        accepted_steps: torch.Tensor,      # (bs,) int, -1 if no accept
        layer_id: int,
    ):
        """After verify, write the accepted-prefix's GLA state into temporal cache."""
        request_number = accepted_steps.shape[0]
        valid = accepted_steps >= 0
        dst = self.forward_metadata.mamba_cache_indices[:request_number][valid]
        src = torch.arange(request_number, device=dst.device)[valid]
        steps = accepted_steps[valid].to(torch.int64)

        layer_cache = self._get_layer_cache(layer_id)
        # `intermediate_ssm` is layer-private and indexed [request, step, ...]
        layer_cache.intermediate_ssm  # NEW field (see §3.3)
        layer_cache.temporal[dst.to(torch.int64)] = (
            layer_cache.intermediate_ssm[src, steps].to(layer_cache.temporal.dtype)
        )
```

And in `forward()`:
```python
if forward_batch.spec_input is not None and forward_batch.spec_input.is_medusa_verify():
    # Don't write final_state to layer_cache.temporal; write per-step states to intermediate_ssm
    layer_cache.intermediate_ssm[:request_number, step_idx] = final_state
else:
    self._store_final_state(layer_cache, mamba_indices, final_state)
```

### 3.3 `intermediate_ssm` allocation

Add a SimpleGLA-aware branch to `MambaPool` allocation. Buffer shape: `(num_simple_gla_layers, max_speculative_bs, max_draft_tokens, num_heads, head_dim_qk, head_dim_v)`. With SALA's `lightning_nkv=16`, head_dim=128 (qk and v same), max_bs=24, K=2: ~24·24·2·16·128·128 ≈ 600 MB BF16. **This is too much.**

Mitigation: at K=1 in R1, `max_draft_tokens=2` → 50 % savings → ~300 MB. Still high; budget allowable but worth measuring. R2 will quantize the intermediate buffer or fold it back into `ssm_states` with extra slots.

Action item: validate the buffer size against `mem-fraction-static=0.84` headroom before the spike.

## 4. Correctness gate (R1 exit criterion)

Server launched with `SOAR_SPEC_MEDUSA=1` and `accept_threshold=1.0`:

1. Send 50 prompts (10 from each task type: mcq, qa, niah, cwe, fwe).
2. For each prompt, capture `predictions.jsonl` from spec mode.
3. Compare token-by-token to `predictions.jsonl` from `SOAR_SPEC_MEDUSA=0` (same seed, same temp).
4. **Pass condition**: 100 % token sequences byte-identical for all 50 prompts.

If fail: GLA state-scatter has a bug. R1 stops; investigation begins.

## 5. Test commands

Local syntax / import-only:
```bash
cd /home/oldzhu/sglang/python
python -c "from sglang.srt.speculative.medusa_worker import MedusaWorker"
python -c "from sglang.srt.speculative.spec_info import SpeculativeAlgorithm; assert SpeculativeAlgorithm.from_string('MEDUSA').is_medusa()"
```

fcloud spike (requires fcloud back up):
```bash
cd /root/submission_sim
SOAR_SPEC_MEDUSA=1 source prepare_env.sh
python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" "${SGLANG_SERVER_ARGS[@]}" &
# wait healthy
curl -s -X POST http://localhost:30000/generate \
  -d '{"text": "Hello world", "sampling_params": {"temperature": 0, "max_new_tokens": 32}}'
# expect: same output as SOAR_SPEC_MEDUSA=0 baseline (byte-identity gate)
```

If single-prompt smoke passes:
```bash
python3 scripts/fcloud/fcloud_workflow.py accuracy \
  --env SOAR_SPEC_MEDUSA=1 \
  --speed-cap 5     # quick subset; full run is R2
```

## 6. Risks unique to R1

| Risk | Likelihood | Mitigation |
|---|---|---|
| `intermediate_ssm` buffer eats too much VRAM | Medium | Limit max_speculative_bs in R1; benchmark before scaling |
| `verify_tree_greedy_func` expects EAGLE's data layout differing from Medusa's chain-K=1 case | Low | Build a chain explicitly (parent_index = [-1, 0]) — sglang already supports this for NEXTN |
| Hidden state passed to heads is post-norm not pre-norm (or vice-versa) — heads see "wrong" h | Medium | At W1=0 it doesn't matter — both pre/post-norm collapse to lm_head(h). Will only matter at R3 head training. Lock the convention in R1 with a comment. |
| `--enable-torch-compile` produces different graphs for verify vs decode | Medium | R1 disables torch.compile when `SOAR_SPEC_MEDUSA=1`. Re-enable in R3 after R2 stability passes. |

## 7. Rollback

Default `SOAR_SPEC_MEDUSA=0` → none of the new code paths run. Worst-case rollback = `git revert <commit>`.

## 8. Decision points (please choose)

**Q1.** Should I proceed to write the R1 code now, or do you want to review this design doc first?
- (A) Proceed: I write the ~635 LOC across 10 files in the next 1–2 turns; we then wait for fcloud to come back to test.
- (B) Review first: I stop here and you read the design; we resume after.

**Q2.** R1 default `K` (number of Medusa heads):
- (A) `K=1` (chain, smallest blast radius — recommended)
- (B) `K=2` (true tree from day one — more code, more risk)

**Q3.** Should I also add a `SOAR_SPEC_NGRAM=1` opt-in to `prepare_env.sh` as a "free insurance" we can test in parallel once fcloud returns? Pure server-arg change; zero risk. Mentioned in [RESEARCH_speculative_decoding_survey_001](RESEARCH_speculative_decoding_survey_001.en.md) §5.2.
- (Y) yes
- (N) defer

## 9. References

- Champion article: https://mp.weixin.qq.com/s/fv-6qLagY1GLryrhx10E_Q
- Reference for state-scatter: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` L1373–1440 (`update_mamba_state_after_mtp_verify`)
- Reference for tree build/verify: `python/sglang/srt/speculative/eagle_utils.py` L41–199 (`build_tree_kernel_efficient`, `verify_tree_greedy_func`)
- Reference for spec worker pattern: `python/sglang/srt/speculative/eagle_worker.py` L79–328
