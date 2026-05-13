"""Medusa speculative-decoding worker (SOAR 2026, MiniCPM-SALA).

Phase R1b Stage 3b — K=1 verify with trained Medusa head forward.

Status
------
Stage 3b activates the Medusa head's W1 weights for actual draft generation.
When SOAR_MEDUSA_HEAD_PATH is set, the worker:
  1. Loads trained W1 weights from the checkpoint at startup.
  2. Uses ``medusa_heads.forward(h)`` to predict the next draft token from the
     hidden state captured at the end of each TARGET_VERIFY forward.
  3. Stores the predicted draft token on ``req._medusa_draft_token`` for use in
     the NEXT step's verify, replacing the Stage 3a fallback of using
     ``req.output_ids[-1]`` (last accepted token).
When SOAR_MEDUSA_HEAD_PATH is not set (default), the worker falls back to the
Stage 3a zero-init behaviour (``req.output_ids[-1]`` draft, always-accept).

What this adds vs Stage 3a
---------------------------
* Real Medusa head forward: ``draft = argmax(lm_head(SiLU(W1(h)) + h))``.
* Hidden-state capture via ``CaptureHiddenMode.LAST`` on the TARGET_VERIFY batch.
  TARGET_VERIFY is always eager (no CUDA graph), so this is safe.
* Per-request draft caching on ``req._medusa_draft_token``.
* ``load_trained_weights()`` call in ``__init__`` if env path is set.

What is still NOT done (deferred to Stage 4)
---------------------------------------------
* K>1 heads (requires tree structure changes beyond K=1 linear chain).
* GLA partial-accept restore + correction forward (only needed when accept_length
  < K for K>1 heads).
* CUDA graph capture for TARGET_VERIFY.

See ``docs/soar_2026_changes/CHANGE_0163_medusa_stage3b_trained_heads.{en,zh}.md``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MedusaWorker:
    """Stage 3b Medusa worker — K=1 verify with trained head forward.

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

        # K=1 draft (either zero-init Stage 3a fallback, or trained Stage 3b heads).
        # prepare_env.sh sets speculative_num_draft_tokens = num_heads + 1
        # (base position included), so draft_token_num must be derived from
        # num_heads (the number of speculative positions), not from
        # speculative_num_draft_tokens.
        self.num_heads: int = int(server_args.speculative_num_medusa_heads)
        self.draft_token_num: int = self.num_heads  # K speculative draft tokens
        assert (
            self.num_heads >= 1
        ), f"speculative_num_medusa_heads must be >= 1, got {self.num_heads}"
        assert self.num_heads == 1, (
            f"Stage 3b supports only num_heads=1 (K=1 linear chain), "
            f"got num_heads={self.num_heads}.  K>1 deferred to Stage 4."
        )

        # ----- Heads instantiation -----
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

        # Determine the correct lm_head: tie_word_embeddings uses embed_tokens.
        hf_config = self.model_runner.model_config.hf_config
        if getattr(hf_config, "tie_word_embeddings", False):
            lm_head_module = model.model.embed_tokens
        else:
            lm_head_module = model.lm_head

        self.medusa_heads = MedusaHeads(
            hidden_size=hidden_size,
            num_heads=self.num_heads,
            lm_head_module=lm_head_module,
            dtype=dtype,
        )
        self.medusa_heads = self.medusa_heads.to(self.device)
        self.medusa_heads.eval()

        # ----- Stage 3b: load trained weights if path is provided -----
        self._use_trained_heads: bool = False
        head_path = os.environ.get("SOAR_MEDUSA_HEAD_PATH", "").strip()
        if head_path:
            try:
                self.medusa_heads.load_trained_weights(head_path, device=self.device)
                self._use_trained_heads = True
                logger.info(
                    "MedusaWorker Stage 3b: loaded trained heads from %s", head_path
                )
            except Exception as exc:
                logger.warning(
                    "MedusaWorker: failed to load trained heads from %r: %s. "
                    "Falling back to Stage 3a zero-init behaviour.",
                    head_path,
                    exc,
                )
        else:
            logger.info(
                "MedusaWorker Stage 3a (zero-init fallback): "
                "SOAR_MEDUSA_HEAD_PATH not set."
            )

        logger.info(
            "MedusaWorker ready: K=%d, num_heads=%d, hidden=%d, "
            "dtype=%s, device=%s, trained=%s, approx_weight_MiB=%.1f",
            self.draft_token_num,
            self.num_heads,
            hidden_size,
            dtype,
            self.device,
            self._use_trained_heads,
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
        """Stage 3b: K=1 TARGET_VERIFY for decode, passthrough for extend.

        EXTEND path (same as Stage 2):
            For prefill / extend steps, spec decoding does not apply.  We flip
            spec_algorithm to NONE and delegate byte-identically to the target
            worker.  This preserves the extend output exactly.

        DECODE path (Stage 3b):
            1. If a cached draft token is available on req._medusa_draft_token
               (from the previous step's head forward), use it as the draft.
               Otherwise fall back to req.output_ids[-1] (Stage 3a behaviour).
            2. Build a K=1 linear-chain NgramVerifyInput.
            3. Set capture_hidden_mode=LAST on the spec_info so the TARGET_VERIFY
               forward returns hidden states (bs, hidden_size).
            4. Run TARGET_VERIFY forward (always eager — no CUDA graph).
            5. Capture hidden states BEFORE verify() (which may index them).
            6. Run verify accept walk.
            7. If trained heads are loaded, run medusa_heads.forward(h) on the
               captured hidden states and store the argmax as the next draft on
               req._medusa_draft_token for each live request.
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

        # ---- DECODE path (Stage 3b verify) ----
        return self._forward_verify_k1(batch)

    def _forward_verify_k1(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run the K=1 Medusa verify step (Stage 3b).

        Draft token per request:
          - Stage 3b (trained heads): req._medusa_draft_token (set by previous step).
          - Stage 3a fallback (zero-init / missing checkpoint): req.output_ids[-1].

        After TARGET_VERIFY, if trained heads are loaded, captures hidden states
        and stores the next predicted draft on req._medusa_draft_token.
        """
        bs = batch.batch_size()

        # 1. Collect draft tokens.
        #    Stage 3b: use cached draft from previous step's head forward.
        #    Stage 3a fallback: use last accepted output (zero-init invariant).
        draft_token_list = []
        for req in batch.reqs:
            cached = getattr(req, "_medusa_draft_token", None)
            if cached is not None and self._use_trained_heads:
                draft_token_list.append(cached)
            else:
                draft_token_list.append(req.output_ids[-1])
        draft_tokens = torch.tensor(
            draft_token_list, dtype=torch.int64, device=self.device
        )

        # 2. Build K=1 trivial tree structures.
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

        # Full causal mask (same as Stage 3a).
        tree_mask_pieces = []
        for req in batch.reqs:
            seq_len_i = len(req.origin_input_ids) + len(req.output_ids)
            tree_mask_pieces.append(
                torch.ones(seq_len_i, dtype=torch.bool, device=self.device)
            )
        tree_mask = torch.cat(tree_mask_pieces, dim=0)

        # 3. Build NgramVerifyInput for K=1 linear chain.
        #    Stage 3b: set capture_hidden_mode=LAST so the TARGET_VERIFY forward
        #    returns hidden states of shape (bs, hidden_size).  TARGET_VERIFY is
        #    always eager (no CUDA graph), so this is safe.
        spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            tree_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=1,
        )
        if self._use_trained_heads:
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

        # 4. Prepare batch for TARGET_VERIFY.
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        spec_info.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()

        # 5. Run TARGET_VERIFY forward (always eager; CUDA graph not used).
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # 6. Capture hidden states BEFORE verify() modifies them.
        #    verify() (via _fill_requests) re-indexes logits_output.hidden_states by
        #    accepted_indices.  We need the pre-index view (shape: bs × hidden_size)
        #    to map one hidden state per request for the next step's draft prediction.
        raw_hidden_states: Optional[torch.Tensor] = None
        if (
            self._use_trained_heads
            and logits_output is not None
            and logits_output.hidden_states is not None
        ):
            raw_hidden_states = logits_output.hidden_states  # (bs, hidden_size)

        # 7. Accept walk.
        logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(
            batch, logits_output, self.page_size
        )
        accept_lens = spec_info.accept_length  # (bs,) tensor

        # CHANGE_0160: zero bonus-token position in req_to_token after verify.
        accept_lens_cpu = spec_info.accept_length.cpu().tolist()
        for _i, _req in enumerate(batch.reqs):
            if accept_lens_cpu[_i] >= 1:
                _bonus_pos = int(batch.seq_lens[_i].item()) - 1
                batch.req_to_token_pool.req_to_token[_req.req_pool_idx, _bonus_pos] = 0

        # 8. Stage 3b: run trained head on captured hidden states, cache next draft.
        #    Each live (not finished) request gets a next draft token stored on
        #    req._medusa_draft_token.  The next call to _forward_verify_k1 will
        #    use these instead of req.output_ids[-1].
        if raw_hidden_states is not None and raw_hidden_states.shape[0] == bs:
            try:
                with torch.no_grad():
                    # (bs, 1, vocab_size) → (bs,)
                    draft_logits = self.medusa_heads(raw_hidden_states)
                    next_draft_ids = draft_logits[:, 0, :].argmax(dim=-1).tolist()
                for req_i, req in enumerate(batch.reqs):
                    if not req.finished():
                        req._medusa_draft_token = next_draft_ids[req_i]
                    else:
                        req._medusa_draft_token = None
            except Exception as exc:
                logger.warning(
                    "MedusaWorker: head forward failed (%s); "
                    "clearing cached drafts for this batch.",
                    exc,
                )
                for req in batch.reqs:
                    req._medusa_draft_token = None

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

