"""Medusa speculative-decoding worker (SOAR 2026, MiniCPM-SALA, CHANGE_0153).

Phase R1a — scaffolding only. The real forward_batch_generation lands in R1b
alongside SimpleGLAAttnBackend.update_simple_gla_state_after_verify.

This file exists so that:
  * SpeculativeAlgorithm.MEDUSA.create_worker() resolves
  * server startup with --speculative-algorithm MEDUSA fails LOUDLY with a
    clear message until R1b is implemented (rather than crashing later in
    obscure ways)

R1a contract: importing this module must not crash. Instantiating MedusaWorker
raises NotImplementedError citing the R1b doc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_R1B_BLOCKER_MESSAGE = (
    "MedusaWorker R1a is a scaffold only — forward_batch_generation lands in "
    "Phase R1b (port update_mamba_state_after_mtp_verify to SimpleGLAAttnBackend "
    "and wire forward_with_hidden through MiniCPMSALAForCausalLM). See "
    "docs/soar_2026_changes/CHANGE_0153_medusa_phase_r1_design.en.md sections 2 and 3."
)


class MedusaWorker(BaseSpecWorker):
    """Skeleton Medusa worker; raises until R1b lands."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        logger.error(
            "MedusaWorker instantiated but R1b is not implemented yet. "
            "Set --speculative-algorithm to a different value or wait for R1b."
        )
        raise NotImplementedError(_R1B_BLOCKER_MESSAGE)

    # The abstract properties below are declared so the class is concrete enough
    # for static analyzers, but they will never be called because __init__ aborts.

    @property
    def target_worker(self):  # type: ignore[override]
        raise NotImplementedError(_R1B_BLOCKER_MESSAGE)

    @property
    def draft_worker(self):  # type: ignore[override]
        raise NotImplementedError(_R1B_BLOCKER_MESSAGE)

    def clear_cache_pool(self):  # type: ignore[override]
        raise NotImplementedError(_R1B_BLOCKER_MESSAGE)
