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

        # K=1 Medusa with canonical EAGLE/NGRAM-style verify layout:
        # ndt = num_heads + 1 = 2 positions per request in the verify input.
        #   position 0 = bonus = last committed token (T_N from output_ids[-1])
        #   position 1 = speculative draft (head's prediction of T_{N+1}, or
        #                fallback = output_ids[-1] when no trained head)
        # The sgl_kernel VerifyTreeGreedy treats position 0 as "root, always
        # accepted" and walks children (positions 1..ndt-1) to validate against
        # target_predict[root]. This is the layout sglang's NGRAM/EAGLE workers
        # use; using ndt=1 (which we did in CHANGE_0164) makes the verify tree
        # have zero children, so no draft can ever be accepted AND the kernel's
        # final assignment ``predicts[root] = target_predict[root]`` writes the
        # model's prediction conditioned on a duplicated last token, silently
        # biasing the committed output. See CHANGE_0164 continuation doc.
        self.num_heads: int = int(server_args.speculative_num_medusa_heads)
        # ndt = num_heads + 1: 1 bonus + num_heads speculative drafts.
        # Matches prepare_env.sh's NUM_DRAFT_TOKENS = SOAR_SPEC_MEDUSA_HEADS + 1.
        self.draft_token_num: int = self.num_heads + 1
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

        # ---- GPTQ hidden-state collection (CHANGE_0164) ----
        # Set SOAR_MEDUSA_DUMP_HIDDEN=<path> to collect (N, hidden_size) FP16
        # tensors from TARGET_VERIFY forwards.  Used to retrain the Medusa head
        # against GPTQ-quantized hidden states so that accept rate is non-zero.
        # Set SOAR_MEDUSA_DUMP_MAX_ROWS to cap collection (default 20000).
        self._dump_hidden_path: str = os.environ.get(
            "SOAR_MEDUSA_DUMP_HIDDEN", ""
        ).strip()
        self._dump_max_rows: int = int(
            os.environ.get("SOAR_MEDUSA_DUMP_MAX_ROWS", "20000")
        )
        self._dump_hidden_buffer: list = []
        self._dump_total_rows: int = 0
        self._dump_lm_head_saved: bool = False
        if self._dump_hidden_path:
            logger.info(
                "MedusaWorker: GPTQ hidden-state dump ENABLED → %s "
                "(max_rows=%d).  lm_head.weight will be saved to %s",
                self._dump_hidden_path,
                self._dump_max_rows,
                self._dump_hidden_path + ".lm_head_weight.pt",
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

    def _flush_hidden_dump(self) -> None:
        """Flush accumulated hidden states to disk (CHANGE_0164 dump mode)."""
        if not self._dump_hidden_buffer:
            return
        try:
            new_rows = torch.cat(self._dump_hidden_buffer, dim=0)
            if os.path.exists(self._dump_hidden_path):
                existing = torch.load(
                    self._dump_hidden_path, map_location="cpu", weights_only=True
                )
                combined = torch.cat([existing, new_rows], dim=0)
            else:
                combined = new_rows
            torch.save(combined, self._dump_hidden_path)
            logger.info(
                "MedusaWorker: flushed %d new rows \u2192 %s (total %d / %d)",
                new_rows.shape[0],
                self._dump_hidden_path,
                combined.shape[0],
                self._dump_max_rows,
            )
            self._dump_hidden_buffer.clear()
        except Exception as exc:
            logger.warning("MedusaWorker: failed to flush hidden dump: %s", exc)

    def _forward_verify_k1(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run the K=1 Medusa verify step (Stage 3b, ndt=2 canonical layout).

        Per-request verify input has 2 positions:
          position 0 (bonus) = req.output_ids[-1]   (last committed token T_N)
          position 1 (draft) = req._medusa_draft_token (head prediction of T_{N+1})
                               or req.output_ids[-1] (Stage 3a fallback)

        Verify tree: linear chain bonus(0) -> draft(1).
        After verify (greedy):
          - position 0 always accepted, predicts[0] = target_predict[0]
            (= model's true next-token, byte-equivalent to non-spec decode).
          - if candidates[1] == target_predict[0]: child accepted,
            predicts[1] = target_predict[1] (=  T_{N+2} given accepted draft).
          - Otherwise child rejected; only the bonus output is committed.

        Hidden state for the next draft is taken from position 0 (the bonus),
        which represents the model's state after consuming the prefix + T_N --
        the same context the trained Medusa head was fit on (last-position
        hidden of a TARGET_VERIFY ndt=1 forward).
        """
        bs = batch.batch_size()
        ndt = self.draft_token_num  # 2 for K=1

        # 1. Collect interleaved [bonus, draft] per request, flat shape (bs*ndt,).
        #    bonus = last committed token; draft = head prediction or fallback.
        interleaved: list[int] = []
        for req in batch.reqs:
            last_out = int(req.output_ids[-1])
            cached = getattr(req, "_medusa_draft_token", None)
            if cached is not None and self._use_trained_heads:
                draft_tok = int(cached)
            else:
                # Stage 3a fallback: no trained head, duplicate the last token
                # as the draft. Verify will reject (target_predict[0] != T_N in
                # general), and only the bonus's correct next-token is
                # committed, matching non-spec decode output.
                draft_tok = last_out
            interleaved.append(last_out)
            interleaved.append(draft_tok)
        draft_tokens = torch.tensor(
            interleaved, dtype=torch.int64, device=self.device
        )  # (bs*ndt,)

        # 2. Tree structures for 2-node linear chain.
        retrive_index = (
            torch.arange(bs * ndt, dtype=torch.int64, device=self.device)
            .view(bs, ndt)
        )  # [[0,1],[2,3],...]
        # bonus has child = draft (col 1); draft is leaf.
        retrive_next_token = torch.full(
            (bs, ndt), -1, dtype=torch.int64, device=self.device
        )
        retrive_next_token[:, 0] = 1
        retrive_next_sibling = torch.full(
            (bs, ndt), -1, dtype=torch.int64, device=self.device
        )

        # 3. Positions: per req [seq_lens, seq_lens+1], flat (bs*ndt,).
        seq_lens_dev = batch.seq_lens
        positions = (
            seq_lens_dev.unsqueeze(1)
            + torch.arange(ndt, dtype=seq_lens_dev.dtype, device=self.device)
        ).view(-1)

        # 4. tree_mask: per req (ndt, seq_len_i - 1 + ndt) full causal, then
        #    flatten and concat. Follows NGRAM USE_FULL_MASK convention
        #    (prefix width seq_len_i - 1; trailing ndt x ndt lower triangular).
        tree_mask_pieces = []
        tail = torch.tril(
            torch.ones((ndt, ndt), dtype=torch.bool, device=self.device)
        )
        for req in batch.reqs:
            seq_len_i = len(req.origin_input_ids) + len(req.output_ids)
            prefix = torch.ones(
                (ndt, seq_len_i - 1), dtype=torch.bool, device=self.device
            )
            req_mask = torch.cat([prefix, tail], dim=1).flatten()
            tree_mask_pieces.append(req_mask)
        tree_mask = torch.cat(tree_mask_pieces, dim=0)

        # 5. NgramVerifyInput. CaptureHiddenMode.FULL so we get hidden states
        #    for both verify positions (shape (bs*ndt, hidden)). Position 0
        #    (bonus) hidden is used both for the head forward and for dumping.
        spec_info = NgramVerifyInput(
            draft_token=draft_tokens,
            tree_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=ndt,
        )
        _should_capture = self._use_trained_heads or (
            bool(self._dump_hidden_path)
            and self._dump_total_rows < self._dump_max_rows
        )
        if _should_capture:
            spec_info.capture_hidden_mode = CaptureHiddenMode.FULL

        # 6. Prepare batch for TARGET_VERIFY.
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = spec_info
        spec_info.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()

        # 7. Run TARGET_VERIFY forward (always eager; CUDA graph not used).
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = batch_result.logits_output
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # 8. Capture position-0 (bonus) hidden BEFORE verify() reindexes
        #    logits_output.hidden_states by accepted_indices.
        bonus_hidden: Optional[torch.Tensor] = None
        if (
            _should_capture
            and logits_output is not None
            and logits_output.hidden_states is not None
        ):
            # FULL mode: hidden_states shape (bs*ndt, hidden_size).
            h_full = logits_output.hidden_states.view(bs, ndt, -1)
            bonus_hidden = h_full[:, 0, :].contiguous()  # (bs, hidden)

        # ---- Dump mode: accumulate bonus hidden states for retraining ----
        if (
            self._dump_hidden_path
            and self._dump_total_rows < self._dump_max_rows
            and bonus_hidden is not None
        ):
            if not self._dump_lm_head_saved:
                try:
                    lm_w = self.medusa_heads.lm_head.weight.detach().cpu()
                    torch.save(lm_w, self._dump_hidden_path + ".lm_head_weight.pt")
                    logger.info(
                        "MedusaWorker: saved lm_head.weight \u2192 %s (shape %s)",
                        self._dump_hidden_path + ".lm_head_weight.pt",
                        list(lm_w.shape),
                    )
                    self._dump_lm_head_saved = True
                except Exception as exc:
                    logger.warning(
                        "MedusaWorker: failed to save lm_head.weight: %s", exc
                    )
            h_cpu = bonus_hidden.detach().cpu().to(torch.float16)
            self._dump_hidden_buffer.append(h_cpu)
            self._dump_total_rows += h_cpu.shape[0]
            if (
                len(self._dump_hidden_buffer) >= 50
                or self._dump_total_rows >= self._dump_max_rows
            ):
                self._flush_hidden_dump()
            if self._dump_total_rows >= self._dump_max_rows:
                logger.info(
                    "MedusaWorker: dump complete (%d rows).  "
                    "Run train_medusa_head.py --hidden-dump-path %s to retrain.",
                    self._dump_total_rows,
                    self._dump_hidden_path,
                )

        # 9. Accept walk.
        logits_output, next_token_ids, num_accepted_tokens = spec_info.verify(
            batch, logits_output, self.page_size
        )
        accept_lens = spec_info.accept_length  # (bs,) tensor

        # 10. Stage 3b: run trained head on bonus hidden, cache next draft.
        if bonus_hidden is not None and bonus_hidden.shape[0] == bs:
            try:
                with torch.no_grad():
                    # (bs, 1, vocab_size) -> argmax -> (bs,)
                    draft_logits = self.medusa_heads(bonus_hidden)
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

        # 11. Restore forward_mode / spec_algorithm for scheduler bookkeeping.
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_algorithm = SpeculativeAlgorithm.MEDUSA

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=num_accepted_tokens,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens.tolist() if accept_lens is not None else None,
        )

