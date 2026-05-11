"""Medusa speculative-decoding worker (SOAR 2026, MiniCPM-SALA).

Phase R1b Stage 2 — heads-shadow smoke test.

Status
------
This worker is the **wiring smoke test** for the Medusa code path on
MiniCPM-SALA. Its contract is:

  * Server boots with ``--speculative-algorithm MEDUSA``.
  * ``MedusaHeads`` (K residual Linear modules + reference to the target
    model's ``lm_head``) is instantiated on GPU; weights are zero-init
    (R1 byte-identity invariant from CHANGE_0153 §2).
  * Every decode/extend step is delegated **byte-identically** to the
    target worker. No draft tokens are produced; no verify pass runs;
    no recurrent-state snapshot/restore happens; ``capture_hidden_mode``
    is **not** modified.
  * ``num_accepted_tokens`` is always 0 (no speculation yet).

What this proves
----------------
1. ``SpeculativeAlgorithm.MEDUSA.create_worker()`` dispatches here.
2. ``server_args`` post-init for MEDUSA succeeds end-to-end.
3. ``MedusaHeads`` allocates on GPU (~32 MiB BF16 at K=1, hidden=4096).
4. The host model's ``lm_head`` is reachable via ``target_worker``.
5. Accuracy/speed under ``SOAR_SPEC_MEDUSA=1`` match baseline.

What this does NOT yet do (deferred to Stage 3)
-----------------------------------------------
* Run heads forward on the last hidden state (needs LAST capture mode).
* Build a verify tree / call ``target_worker.forward_batch_generation
  (..., is_verify=True)``.
* Snapshot/restore SimpleGLA recurrent state across the verify pass
  (Stage 1 helpers in ``hybrid_linear_attn_backend.py`` remain unused).

See ``docs/soar_2026_changes/CHANGE_0155_medusa_phase_r1b_stage2.{en,zh}.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MedusaWorker:
    """Stage 2 Medusa worker — pure delegation + heads instantiation.

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

        # R1b Stage 2: K is fixed to whatever server_args reports (default 1).
        # Stage 3 will validate K vs ``speculative_num_draft_tokens`` once the
        # verify path runs. For now we only allocate the heads.
        self.num_heads: int = int(server_args.speculative_num_medusa_heads)
        assert (
            self.num_heads >= 1
        ), f"speculative_num_medusa_heads must be >= 1, got {self.num_heads}"

        # ----- Heads instantiation (smoke test for weight allocation) -----
        # Import locally so that import errors don't break ``--speculative-algorithm NONE``.
        from sglang.srt.models.minicpm_medusa_heads import MedusaHeads

        model = self.model_runner.model
        if not hasattr(model, "lm_head"):
            raise RuntimeError(
                "MedusaWorker requires the target model to expose ``lm_head``. "
                f"Got model={type(model).__name__} with no lm_head attribute. "
                "Medusa is only supported on MiniCPM-SALA in this branch."
            )

        hidden_size = self.model_runner.model_config.hidden_size
        # Match the host model's parameter dtype. ``model_runner.dtype`` is the
        # canonical source (bfloat16 for SALA in the SOAR submission config).
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
            "MedusaWorker Stage 2 ready: K=%d, hidden=%d, dtype=%s, device=%s, "
            "approx_weight_MiB=%.1f",
            self.num_heads,
            hidden_size,
            dtype,
            self.device,
            self.num_heads * hidden_size * hidden_size * torch.tensor([], dtype=dtype).element_size() / (1024 * 1024),
        )

    # ----- BaseSpecWorker-compatible duck-typed interface -----
    #
    # NGRAMWorker uses plain instance attributes (not @property) so the
    # scheduler can read ``draft_worker.target_worker`` directly. We mirror
    # that. ``draft_worker = None`` because Medusa has no separate draft
    # model — heads attach to the target's hidden state.

    @property
    def draft_worker(self):  # type: ignore[override]
        return None

    def clear_cache_pool(self) -> None:  # type: ignore[override]
        # No draft KV-cache pool in Stage 2 (no verify path yet).
        pass

    # ----- Forward dispatch -----

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Stage 2: pure passthrough to the target worker.

        We do NOT modify ``batch.forward_mode``, ``batch.spec_info`` or
        ``capture_hidden_mode``. The output is therefore byte-identical to
        running the server with ``--speculative-algorithm NONE``.
        """
        # Stage 2 debug: log the incoming batch state to diagnose
        # "forward_decode called on prefill" mismatch.
        logger.info(
            "[MedusaWorker.fbg] batch.forward_mode=%s batch_size=%s "
            "input_ids_len=%s seq_lens=%s extend_num_tokens=%s "
            "spec_algorithm=%s spec_info=%s",
            batch.forward_mode,
            batch.batch_size(),
            int(batch.input_ids.numel()) if batch.input_ids is not None else None,
            batch.seq_lens.tolist() if batch.seq_lens is not None else None,
            batch.extend_num_tokens,
            batch.spec_algorithm,
            type(batch.spec_info).__name__ if batch.spec_info is not None else None,
        )
        model_worker_batch = batch.get_model_worker_batch()
        logger.info(
            "[MedusaWorker.fbg] mwb.forward_mode=%s mwb.input_ids_len=%s "
            "mwb.spec_algorithm=%s",
            model_worker_batch.forward_mode,
            int(model_worker_batch.input_ids.numel()),
            model_worker_batch.spec_algorithm,
        )
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=None,
        )
