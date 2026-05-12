"""Medusa heads for MiniCPM-SALA (SOAR 2026, CHANGE_0153 Phase R1).

Each Medusa head is a residual MLP applied to the main model's last hidden
state. The final classification uses the SHARED main-model lm_head, so the
total weight overhead is K * hidden^2 (R1 K=1 → ~32 MB BF16 at hidden=4096).

Critical R1 invariant: W1 is zero-initialized (Medusa paper §3.2). With W1=0,

    SiLU(0 @ h) + h = SiLU(0) + h = 0 + h = h
    p^(k) = softmax(lm_head(h)) == base model's next-token distribution

so every Medusa-head draft token equals the base model's argmax. At
accept_threshold=1.0 the verifier accepts every draft → output is byte-identical
to no-spec baseline. This is the correctness gate R1 rests on.

R1b will wire forward() into the worker. R3 will train W1 on eval-distribution
data so accept_rate stays high while heads become useful predictors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


class MedusaHead(nn.Module):
    """A single residual MLP head.

    Forward:  p = SiLU(W1 @ h) + h   (no classifier here — caller applies lm_head)
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        # nn.Linear(bias=False) gives shape (out, in); zero-init satisfies the
        # R1 byte-identity invariant.
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.W1.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden) → returns (B, hidden)
        return F.silu(self.W1(h)) + h


class MedusaHeads(nn.Module):
    """Bundle of K Medusa heads sharing a single lm_head classifier.

    The lm_head is passed in by reference from the host model (MiniCPMSALAForCausalLM)
    so weight sharing is automatic and submission size stays small.

    Args:
        hidden_size: model hidden size (4096 for MiniCPM-SALA).
        num_heads: K (R1 default = 1).
        lm_head_module: the main model's lm_head (e.g. ParallelLMHead). Called
            on the residual MLP output to produce logits of shape (B, vocab).
        dtype: parameter dtype; defaults to bfloat16 to match the main model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        lm_head_module: nn.Module,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # ModuleList so each head's W1 is independently trainable in R3.
        self.heads = nn.ModuleList(
            [MedusaHead(hidden_size, dtype=dtype) for _ in range(num_heads)]
        )
        # Stored as attribute (not submodule) to avoid double-counting parameters
        # when MedusaHeads is part of MiniCPMSALAForCausalLM. The host model owns
        # lm_head; we just hold a non-owning reference.
        self._lm_head = [lm_head_module]

    @property
    def lm_head(self) -> nn.Module:
        return self._lm_head[0]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute per-head logits.

        Args:
            h: hidden state of shape (B, hidden_size).

        Returns:
            logits stacked along a new dim K, shape (B, num_heads, vocab_size).
        """
        outs: List[torch.Tensor] = []
        for head in self.heads:
            outs.append(self.lm_head(head(h)))
        # NOTE: lm_head may return a tuple in sglang (logits, others) depending
        # on the LogitsProcessor wiring. R1b will adapt as needed; R1a only
        # defines the structure.
        return torch.stack(outs, dim=1) if not isinstance(outs[0], tuple) else outs  # type: ignore[return-value]

    def load_trained_weights(self, path: str, device: str = "cuda") -> None:
        """Load W1 weights from a checkpoint saved by ``train_medusa_head.py``.

        The checkpoint is a plain ``torch.save`` dict with keys
        ``"head_{k}.W1.weight"`` for k in 0..num_heads-1.  Only the W1
        parameters are saved/loaded; lm_head weights live in the main model.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            device: Device to load weights onto (default ``"cuda"``).
        """
        import os

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"MedusaHeads: checkpoint not found: {path!r}. "
                "Run train_medusa_head.py first to generate it."
            )
        state = torch.load(path, map_location=device, weights_only=True)
        # Build expected keys, check coverage and load.
        expected_keys = {
            f"heads.{k}.W1.weight" for k in range(self.num_heads)
        }
        missing = expected_keys - set(state.keys())
        unexpected = set(state.keys()) - expected_keys
        if missing:
            raise KeyError(
                f"MedusaHeads: checkpoint missing keys: {sorted(missing)}"
            )
        if unexpected:
            import logging
            logging.getLogger(__name__).warning(
                "MedusaHeads: unexpected keys in checkpoint (ignored): %s",
                sorted(unexpected),
            )
        missing_actual, unexpected_actual = self.load_state_dict(
            state, strict=False
        )
        # strict=False is intentional: lm_head is not in the checkpoint.
        # Report any genuinely missing head keys.
        non_lm_missing = [k for k in missing_actual if "lm_head" not in k]
        if non_lm_missing:
            raise KeyError(
                f"MedusaHeads: could not load some keys: {non_lm_missing}"
            )
        import logging
        logging.getLogger(__name__).info(
            "MedusaHeads: loaded %d head(s) from %s", self.num_heads, path
        )
