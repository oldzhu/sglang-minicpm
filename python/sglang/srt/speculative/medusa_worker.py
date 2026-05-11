"""Medusa speculative-decoding worker (SOAR 2026, MiniCPM-SALA).

Phase R1b Stage 3a — K=1 verify with GLA snapshot/clear.

Status
------
Stage 3a wires the real TARGET_VERIFY forward for Medusa K=1 (zero-init heads).
The draft token for each step is the previous step's argmax output, which for
zero-init heads is identical to what a normal decode would produce.  This means
the verify always accepts all K=1 draft tokens (accept_length always == 0, root
always in chain, no children to reject).

What this adds vs Stage 2
--------------------------
* Real TARGET_VERIFY forward instead of pure passthrough.
* K=1 NgramVerifyInput-compatible tree structure constructed manually.
* GLA state snapshot before verify + clear after verify (snapshot/restore
  protocol from hybrid_linear_attn_backend.py Stage 3 scaffolding).
* accept_length and num_accepted_tokens correctly populated (0 bonus tokens
  for K=1, same throughput as baseline — correctness proof, not speedup yet).

What is still NOT done (deferred to Stage 3b)
----------------------------------------------
* Real Medusa head forward (hidden-state capture + lm_head branch).
  Draft token is still derived from req.output_ids[-1], NOT from the Medusa
  head.  Zero-init heads produce the same token as req.output_ids[-1], so
  output is byte-identical to baseline for the Stage 3a validation.
* CUDA graph capture for TARGET_VERIFY (needs CHANGE_0153 §5 work).
* GLA partial-accept restore + correction forward (only needed for accept_length
  < K, which never happens with K=1 zero-init; correctness deferred to Stage 3b
  when K>1 trained heads are tested).

See ``docs/soar_2026_changes/PROPOSAL_medusa_stage3_verify_rewind.{en,zh}.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MedusaWorker:
    """Stage 3a Medusa worker — K=1 verify + GLA snapshot/clear.

    Mirrors ``NGRAMWorker``'s plain-class style (not an ABC subclass) so the
    scheduler's existing ``draft_worker.forward_batch_generation(batch)`` dispatch
    works without changes.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Stage 3a: K=1 draft (zero-init heads, always accept, no speedup yet).
        # Stage 3b will use the Medusa head forward for K>1.
        # Note: prepare_env.sh sets speculative_num_draft_tokens = num_heads + 1
        # (base position included), so draft_token_num must be derived from
        # num_heads (the number of speculative positions), not from
        # speculative_num_draft_tokens.
        self.num_heads: int = int(server_args.speculative_num_medusa_heads)
        self.draft_token_num: int = self.num_heads  # K speculative draft tokens
        assert (
            self.num_heads >= 1
        ), f"speculative_num_medusa_heads must be >= 1, got {self.num_heads}"
        assert self.num_heads == 1, (
            f"Stage 3a supports only num_heads=1 (K=1 linear chain), "
            f"got num_heads={self.num_heads}.  Stage 3b required for K>1."
        )

        # ----- Heads instantiation (for future Stage 3b actual head forward) -----
        from sglang.srt.models.minicpm_medusa_heads import MedusaHeads

        model = self.model_runner.model
        if not hasattr(model, "lm_head"):
            raise RuntimeError(
                "MedusaWorker requires the target model to expose ``lm_head``. "
                f"Got model={type(model).__name__} with no lm_head attribute. "
                "Medusa is only supported on MiniCPM-SALA in this branch."
            )

        hidden_size = self.model_runner.model_config.hidden_size
        dtype = self.model_runner.dtype
        self.medusa_heads = MedusaHeads(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            lm_head_module=model.lm_head,
            dtype=dtype,
        )
        self.medusa_heads = self.medusa_heads.to(self.device)
        self.medusa_heads.eval()

        logger.info(
            "MedusaWorker Stage 3a ready: K=%d, num_heads=%d, hidden=%d, "
            "dtype=%s, device=%s, approx_weight_MiB=%.1f",
            self.draft_token_num,
            self.num_heads,
            hidden_size,
            dtype,
            self.device,
            self.num_heads * hidden_size * hidden_size
            * torch.tensor([], dtype=dtype).element_size()
            / (1024 * 1024),
        )

    # ----- BaseSpecWorker-compatible duck-typed interface -----

    @property
    def draft_worker(self):  # type: ignore[override]
        return None

    def clear_cache_pool(self) -> None:  # type: ignore[override]
        pass

    # ----- GLA backend accessor -----

    def _get_gla_backend(self):
        """Return the SimpleGLAAttnBackend if the model uses one, else None.

        For MiniCPM-SALA: model_runner.attn_backend is HybridLinearAttnBackend
        which wraps SimpleGLAAttnBackend as linear_attn_backend.  The snapshot/
        restore helpers live on the inner backend.
        """
        attn_backend = getattr(self.model_runner, "attn_backend", None)
        if attn_backend is None:
            return None
        linear_backend = getattr(attn_backend, "linear_attn_backend", None)
        if linear_backend is None:
            return None
        if not hasattr(linear_backend, "snapshot_state_for_spec"):
            return None
        return linear_backend

    # ----- Forward dispatch -----

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Stage 3a: K=1 TARGET_VERIFY for decode, passthrough for extend.

        EXTEND path (same as Stage 2):
            For prefill / extend steps, spec decoding does not apply.  We flip
            spec_algorithm to NONE and delegate byte-identically to the target
            worker.  This preserves the extend output exactly.

        DECODE path (Stage 3a new):
            1. Derive the draft token from req.output_ids[-1] for each request.
               For zero-init Medusa heads, this equals what the head would
               predict (argmax of the previous step's lm_head output).
            2. Build a K=1 linear-chain NgramVerifyInput (trivial tree: root only,
               no children, positions = seq_lens, retrive_index = arange(bs)).
            3. Snapshot GLA recurrent state (SimpleGLAAttnBackend.snapshot_state_for_spec).
            4. Run TARGET_VERIFY forward.
            5. Run verify accept walk (K=1 zero-init → always accept root).
            6. Clear GLA snapshot (correct_state = live state after verify).
            7. Return GenerationBatchResult.

        GLA state correctness note (§14 PROPOSAL_medusa_stage3_verify_rewind):
            For K=1 with zero-init heads, accept_length == 0 (root accepted, no
            children).  batch.seq_lens advances by 1 per step — same as normal
            decode.  The GLA state after the single-token verify forward is
            therefore correct: the model saw exactly the one accepted token (root),
            and the state reflects that.  No restore + correction forward needed.
            When Stage 3b trains heads (K=1 still, but W1≠0 possible mismatch):
            restore_state_for_spec + correction_forward will be added here.
        """
        # ---- EXTEND path (passthrough, same as Stage 2) ----
        if batch.forward_mode.is_extend():
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            model_worker_batch = batch.get_model_worker_batch()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch
            )
            return GenerationBatchResult(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
                accept_lens=None,
            )

        # ---- DECODE path (Stage 3a verify) ----
        return self._forward_verify_k1(batch)

    def _forward_verify_k1(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run the K=1 Medusa verify step.

        Draft token = req.output_ids[-1] for each request (last accepted output).
        For zero-init heads this is identical to the Medusa head's prediction.
        """
        bs = batch.batch_size()

        # 1. Collect draft tokens (last accepted output per request).
        draft_tokens = torch.tensor(
            [req.output_ids[-1] for req in batch.reqs],
            dtype=torch.int64,
            device=self.device,
        )

        # 2. Build K=1 trivial tree structures.
        #    - positions: absolute sequence positions of the 1 draft token per req.
        #    - retrive_index: global draft-token index for each (req, slot).
        #    - retrive_next_token / retrive_next_sibling: -1 (no children).
        #    - tree_mask: full causal mask, 1 draft token attends to all prior tokens.
        #      For USE_FULL_MASK (consistent with ngram_worker.py USE_FULL_MASK=True),
        #      each request i contributes a ones-tensor of shape (seq_len_i,).
        positions = batch.seq_lens.clone()  # (bs,) — draft token absolute positions

        retrive_index = (
            torch.arange(bs, dtype=torch.int64, device=self.device).unsqueeze(1)
        )  # (bs, 1)
        retrive_next_token = torch.full(
            (bs, 1), -1, dtype=torch.int64, device=self.device
        )
        retrive_next_sibling = torch.full(
            (bs, 1), -1, dtype=torch.int64, device=self.device
        )

        # Full causal mask: each of the 1 draft tokens per request attends to
        # all previous tokens (seq_len_i tokens in KV cache) + itself (1 slot
        # allocated by prepare_for_verify).  Total: seq_len_i + 1 booleans, all True.
        tree_mask_pieces = []
        for req in batch.reqs:
            seq_len_i = len(req.origin_input_ids) + len(req.output_ids)
            tree_mask_pieces.append(
                torch.ones(seq_len_i, dtype=torch.bool, device=self.device)
            )
        tree_mask = torch.cat(tree_mask_pieces, dim=0)  # variable length, all True

        # 3. Build NgramVerifyInput for K=1 linear chain.
        spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            tree_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=1,
        )

        # 4. Prepare batch for TARGET_VERIFY:
        #    - sets batch.input_ids = draft_tokens
        #    - allocates K=1 KV slots per request (alloc_paged_token_slots_extend)
        #    - does NOT increment batch.seq_lens (verify() does that)
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM  # reuse NGRAM verify infra
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        spec_info.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()

        # 5. Snapshot GLA state before verify forward (Stage 3 §14).
        gla_backend = self._get_gla_backend()
        if gla_backend is not None:
            # Use req_to_token_pool.get_mamba_indices to get per-request GLA
            # pool indices (independent of forward_metadata which isn't set yet).
            mamba_indices = (
                gla_backend.req_to_token_pool.get_mamba_indices(
                    batch.req_pool_indices
                )
            )
            gla_backend.snapshot_state_for_spec(mamba_indices)

        # 6. Run TARGET_VERIFY forward.
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # 7. Accept walk (always accepts root for K=1 zero-init Medusa).
        logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(
            batch, logits_output, self.page_size
        )
        accept_lens = spec_info.accept_length  # (bs,) tensor, always 0 for K=1

        # 8. GLA state is correct after single-token verify (K=1, accept_len==0):
        #    The model processed exactly 1 token (the accepted root draft), so
        #    GLA state advanced by exactly 1 — same as a normal decode step.
        #    Just discard the snapshot.
        if gla_backend is not None:
            gla_backend.clear_state_snapshot_for_spec()

        # 9. Restore forward_mode / spec_algorithm for scheduler bookkeeping.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_algorithm = SpeculativeAlgorithm.MEDUSA

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens.tolist() if accept_lens is not None else None,
        )

