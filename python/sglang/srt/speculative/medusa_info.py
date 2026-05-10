"""Medusa speculative decoding datastructures (SOAR 2026, MiniCPM-SALA).

Phase R1 — see docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.{en,zh}.md.

Medusa attaches K lightweight MLP heads to the main model's last hidden state.
Each head predicts position t+k. Top-s candidates per head are arranged into a
tree (a chain at K=1) and verified in one main-model forward pass; the longest
accepted prefix wins.

R1 only provides the data containers and (later in R1b) the verify input. The
worker's forward_batch_generation lands in R1b alongside SimpleGLA state-scatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.speculative.spec_info import SpecInput, SpecInputType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch


@dataclass
class MedusaInput(SpecInput):
    """Tree-verify input produced by Medusa heads.

    At K=1 (R1 default), the tree degenerates to a chain of length 2: the base
    model's argmax plus one Medusa-head draft token.

    Fields are intentionally aligned with EagleVerifyInput so the existing
    eagle_utils tree-build / verify_tree_greedy_func kernels can be reused
    without modification.
    """

    # Flattened draft tokens for the tree, per request. Shape: (bs, num_draft_tokens).
    # Position 0 is the parent (base model's argmax). Positions 1..K are head outputs.
    draft_token_ids: torch.Tensor = None  # type: ignore[assignment]

    # Per-token parent index in the tree (-1 for root). Shape: (bs, num_draft_tokens).
    parent_index: Optional[torch.Tensor] = None

    # Flatten-order retrieval index used by verify_tree_greedy_func.
    # Shape: (bs, num_draft_tokens).
    retrieve_index: Optional[torch.Tensor] = None

    # Attention mask over the K draft tokens. Shape: (bs, num_draft_tokens, num_draft_tokens).
    # At K=1 chain this collapses to a causal 2x2 mask per request.
    tree_mask: Optional[torch.Tensor] = None

    # Absolute positions of each draft token in the sequence.
    # Shape: (bs, num_draft_tokens).
    positions: Optional[torch.Tensor] = None

    # Greedy-verify threshold. R1 = 1.0 (byte-identity gate): a draft token is
    # accepted iff it matches the base model's argmax. With heads zero-init
    # (W1=0 in MedusaHeads), every head output equals lm_head(h), so every
    # draft trivially matches argmax — R1 has accept_rate = 1.0 by construction.
    accept_threshold: float = 1.0

    # Number of heads K. Stored explicitly so the worker can dispatch without
    # peeking into draft_token_ids.shape.
    num_heads: int = 1

    def __init__(
        self,
        draft_token_ids: torch.Tensor,
        parent_index: Optional[torch.Tensor] = None,
        retrieve_index: Optional[torch.Tensor] = None,
        tree_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        accept_threshold: float = 1.0,
        num_heads: int = 1,
    ) -> None:
        super().__init__(SpecInputType.MEDUSA_VERIFY)
        self.draft_token_ids = draft_token_ids
        self.parent_index = parent_index
        self.retrieve_index = retrieve_index
        self.tree_mask = tree_mask
        self.positions = positions
        self.accept_threshold = accept_threshold
        self.num_heads = num_heads

    def is_medusa_verify(self) -> bool:
        return True

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        """How many extra tokens verify-forward expands per request.

        For Medusa: every request runs K+1 tokens through the main model
        (1 base argmax + K head predictions). The logprob coefficient is 1
        because we only need logits at the accepted-prefix tail.
        """
        num_draft_tokens = (
            int(self.draft_token_ids.shape[1])
            if self.draft_token_ids is not None and self.draft_token_ids.dim() >= 2
            else (self.num_heads + 1)
        )
        return (num_draft_tokens, 1)


@dataclass
class MedusaVerifyOutput:
    """Result of one Medusa generation step (post-verify).

    R1b will populate these fields from verify_tree_greedy_func; in R1a the
    class is defined so downstream code can import it.
    """

    # Accepted token ids per request, concatenated. Shape: (sum(accept_lens),).
    verified_id: torch.Tensor

    # Per-request accepted lengths (>= 1; verify_tree_greedy guarantees the
    # base argmax always accepts). Shape: (bs,) on CPU for scheduler ergonomics.
    accept_length_per_req_cpu: List[int]

    # Last hidden state at the new "frontier" token, used as input to Medusa
    # heads on the next step. Shape: (bs, hidden_size).
    last_hidden_state: Optional[torch.Tensor] = None
