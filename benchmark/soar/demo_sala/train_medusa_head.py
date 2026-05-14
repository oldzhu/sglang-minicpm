#!/usr/bin/env python3
"""Train a single Medusa head (K=1) for MiniCPM-SALA.

Usage (on fcloud):
  python3 /root/sglang-minicpm/benchmark/soar/demo_sala/train_medusa_head.py \
    --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
    --data-path  /root/data/perf_public_set.jsonl \
    --output     /root/medusa_head_k1.pt \
    --epochs 5 \
    --lr 1e-4 \
    --max-len 512

Output: a checkpoint file with key ``heads.0.W1.weight`` (BF16, shape
[hidden_size, hidden_size]) loadable by ``MedusaHeads.load_trained_weights()``.

Then set in prepare_env.sh before server launch:
  export SOAR_MEDUSA_HEAD_PATH=/root/medusa_head_k1.pt

Design
------
1. Load the non-quantized MiniCPM-SALA model via transformers (trust_remote_code).
2. Register a forward hook on the final RMSNorm (model.model.norm) to capture
   hidden states.  Because the transformers model applies scale_emb / scale_width
   internally, the captured states match what the server's lm_head sees.
3. Tokenize each prompt (from perf_public_set.jsonl, chat-template formatted).
4. For each sample, collect (h_t, token_{t+1}) training pairs from all positions.
5. Train a single Linear W1 (bias=False, dtype=bfloat16) to minimise:
     CrossEntropy( lm_head( SiLU(W1(h_t)) + h_t ), token_{t+1} )
6. Save state_dict with keys ``heads.0.W1.weight``.

Note: the main model weights (lm_head, transformer layers) are fully frozen.
Gradient flows only through W1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prompts(data_path: str, max_samples: int = 200) -> List[dict]:
    """Load prompts from perf_public_set.jsonl (or any JSONL with 'prompt' key)."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append(obj)
            if len(samples) >= max_samples:
                break
    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


def format_prompt(sample: dict, tokenizer) -> Optional[str]:
    """Apply the chat template if available; fall back to raw 'prompt' field."""
    prompt_text = sample.get("prompt", sample.get("question", ""))
    if not prompt_text:
        return None
    # Try chat template (MiniCPM-SALA uses a thinking chat template).
    try:
        messages = [{"role": "user", "content": prompt_text}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text
    except Exception:
        return prompt_text


# ---------------------------------------------------------------------------
# Hook-based hidden state collector
# ---------------------------------------------------------------------------

class HiddenStateCollector:
    """Registers a forward hook on ``module`` and stores the last output."""

    def __init__(self, module: nn.Module) -> None:
        self._last: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output is the tensor from model.model.norm
        if isinstance(output, (tuple, list)):
            self._last = output[0].detach()
        else:
            self._last = output.detach()

    @property
    def last(self) -> Optional[torch.Tensor]:
        return self._last

    def remove(self):
        self._handle.remove()


# ---------------------------------------------------------------------------
# Collection pass: gather (hidden_state, next_token) pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_training_data(
    model,
    tokenizer,
    samples: List[dict],
    max_len: int,
    device: str,
    scale_width: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass over all samples and collect training pairs.

    Returns:
        h_tensor: (N, hidden_size) — hidden states at positions t
        label_tensor: (N,) — token ids at positions t+1 (next-token labels)
    """
    # Find the final norm module (model.model.norm in sglang style).
    # transformers layout: model.model.norm (for CausalLM)
    norm_module = None
    for attr in ["model", "transformer"]:
        inner = getattr(model, attr, None)
        if inner is not None:
            norm_module = getattr(inner, "norm", None)
            if norm_module is not None:
                break
    if norm_module is None:
        raise RuntimeError(
            "Cannot find model.model.norm — check the model architecture."
        )
    logger.info("Hooking on: %s", type(norm_module).__name__)
    collector = HiddenStateCollector(norm_module)

    all_h: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    model.eval()
    for idx, sample in enumerate(samples):
        text = format_prompt(sample, tokenizer)
        if text is None:
            continue
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
        ).input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        try:
            _ = model(input_ids, output_hidden_states=False, use_cache=False)
        except TypeError:
            # Some models don't support use_cache keyword
            _ = model(input_ids)

        hidden = collector.last  # (1, seq_len, hidden_size) or (seq_len, hidden_size)
        if hidden is None:
            continue

        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)  # (seq_len, hidden_size)

        # Apply scale_width correction (MiniCPM-SALA divides by scale_width in
        # the sglang forward; the transformers model may or may not do this).
        # We detect it by checking if the model config has dim_model_base and
        # hidden_size attributes and the caller passed scale_width.
        if scale_width is not None and scale_width != 1.0:
            hidden = hidden / scale_width

        # hidden[:-1] = states at positions 0..T-2
        # input_ids[0, 1:] = next tokens at positions 1..T-1
        h = hidden[:-1].float()          # (T-1, hidden_size) — use fp32 for grads
        labels = input_ids[0, 1:].long() # (T-1,)

        all_h.append(h)
        all_labels.append(labels)

        if (idx + 1) % 20 == 0:
            n = sum(t.shape[0] for t in all_h)
            logger.info("  sample %d/%d  total pairs so far: %d", idx + 1, len(samples), n)

    collector.remove()

    if not all_h:
        raise RuntimeError("No training pairs collected — check data path and model.")

    h_tensor = torch.cat(all_h, dim=0)       # (N, hidden_size)
    label_tensor = torch.cat(all_labels, dim=0) # (N,)
    logger.info(
        "Collected %d (h, label) pairs, hidden_size=%d", h_tensor.shape[0], h_tensor.shape[1]
    )
    return h_tensor, label_tensor


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    h: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    hidden_size: int,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> nn.Linear:
    """Train a single MedusaHead W1 and return it."""
    W1 = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.bfloat16).to(device)
    nn.init.zeros_(W1.weight)

    # Move data to device (fp32 → bfloat16 for W1 compatibility)
    h = h.to(device, dtype=torch.bfloat16)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(W1.parameters(), lr=lr)
    N = h.shape[0]
    steps_per_epoch = math.ceil(N / batch_size)

    # Freeze lm_head
    for p in lm_head.parameters():
        p.requires_grad_(False)
    lm_head.to(device)
    lm_head.eval()

    # Random permutation index (shuffled once per epoch inside loop)
    logger.info("Training W1: N=%d, epochs=%d, lr=%g, batch=%d", N, epochs, lr, batch_size)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        total_correct = 0
        t0 = time.time()

        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            h_batch = h[idx]        # (B, hidden)
            y_batch = labels[idx]   # (B,)

            optimizer.zero_grad()

            # MedusaHead forward: SiLU(W1(h)) + h
            draft_h = F.silu(W1(h_batch)) + h_batch  # (B, hidden), bfloat16

            # lm_head forward (frozen)
            with torch.no_grad():
                logits_ref = lm_head(draft_h)  # may return (B, vocab) or tuple

            if isinstance(logits_ref, (tuple, list)):
                logits_ref = logits_ref[0]

            # We need gradients through W1, so recompute lm_head WITH grad
            # (but only for the W1 grad, not lm_head grad)
            logits = lm_head(draft_h)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            loss = F.cross_entropy(logits.float(), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.shape[0]
            total_correct += (logits.argmax(dim=-1) == y_batch).sum().item()

        avg_loss = total_loss / N
        acc = total_correct / N
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d: loss=%.4f  top1_acc=%.3f%%  time=%.1fs",
            epoch + 1, epochs, avg_loss, acc * 100, elapsed,
        )

    return W1


def _train_from_gptq_dump(args) -> None:
    """Train Medusa head using pre-collected GPTQ hidden states (CHANGE_0164).

    Skips model loading entirely.  Instead:
      1. Loads hidden states (N, hidden_size) FP16 from ``args.hidden_dump_path``.
      2. Loads lm_head.weight (vocab_size, hidden_size) from
         ``args.hidden_dump_path + '.lm_head_weight.pt'``.
      3. Computes targets y = argmax(F.linear(h, lm_head_weight)) to replicate
         what the GPTQ model actually predicts, not the ground-truth next token.
      4. Trains the W1 layer using the existing ``train()`` function.
    """
    device = args.device

    # 1. Load hidden states
    logger.info("Loading GPTQ hidden states from %s …", args.hidden_dump_path)
    h_tensor = torch.load(
        args.hidden_dump_path, map_location="cpu", weights_only=True
    )
    h_tensor = h_tensor.float()
    logger.info("Hidden states: shape=%s", list(h_tensor.shape))

    # 2. Load lm_head.weight
    lm_weight_path = args.hidden_dump_path + ".lm_head_weight.pt"
    if not os.path.exists(lm_weight_path):
        logger.error("lm_head weight file not found: %s", lm_weight_path)
        sys.exit(1)
    lm_head_weight = torch.load(
        lm_weight_path, map_location="cpu", weights_only=True
    )
    logger.info("lm_head.weight: shape=%s, dtype=%s", list(lm_head_weight.shape), lm_head_weight.dtype)
    vocab_size, hidden_size = lm_head_weight.shape

    # 3. Compute labels: y[i] = argmax(F.linear(h[i+1], lm_head_weight)).
    # Medusa head predicts the NEXT token from current hidden state, so pair
    # (h[i], y[i+1]). We drop the last hidden state since it has no follower.
    # Cross-request boundary noise is negligible (~0.3% with ~350 tok/req).
    logger.info("Computing GPTQ training labels (shifted by 1 for next-token prediction) …")
    lm_w_gpu = lm_head_weight.to(device, dtype=torch.bfloat16)
    h_bf16 = h_tensor.to(device, dtype=torch.bfloat16)
    chunk_size = 1024
    label_parts = []
    with torch.no_grad():
        for i in range(0, h_bf16.shape[0], chunk_size):
            chunk = h_bf16[i : i + chunk_size]
            logits = F.linear(chunk, lm_w_gpu)
            label_parts.append(logits.argmax(dim=-1).cpu())
    all_tokens = torch.cat(label_parts, dim=0)  # token at each position
    # Shift: input h[i] -> target token at position i+1
    h_tensor = h_tensor[:-1].contiguous()
    label_tensor = all_tokens[1:].contiguous()
    del h_bf16, lm_w_gpu, label_parts, all_tokens
    logger.info("Computed %d (h, next-token) pairs", len(label_tensor))

    # 4. Build a frozen nn.Linear wrapper for the train() function
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)
    lm_head.weight = nn.Parameter(
        lm_head_weight.to(torch.bfloat16), requires_grad=False
    )

    # 5. Train
    W1 = train(
        h=h_tensor,
        labels=label_tensor,
        lm_head=lm_head,
        hidden_size=hidden_size,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # 6. Save checkpoint
    checkpoint = {"heads.0.W1.weight": W1.weight.cpu()}
    torch.save(checkpoint, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    logger.info(
        "Saved GPTQ-aligned checkpoint → %s (%.1f MiB, shape=%s)",
        args.output,
        size_mb,
        list(W1.weight.shape),
    )
    logger.info("To enable: export SOAR_MEDUSA_HEAD_PATH=%s", args.output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Medusa head K=1 for MiniCPM-SALA")
    parser.add_argument(
        "--model-path",
        default="/root/models/openbmb/MiniCPM-SALA-Copy",
        help="Path to non-quantized MiniCPM-SALA model (default: %(default)s)",
    )
    parser.add_argument(
        "--data-path",
        default="/root/data/perf_public_set.jsonl",
        help="Path to JSONL dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="/root/medusa_head_k1.pt",
        help="Output checkpoint path (default: %(default)s)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=512,
                        help="Max token length per sample during collection")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max samples to load from JSONL")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--hidden-dump-path",
        default=None,
        help="Path to pre-collected GPTQ hidden states (.pt from SOAR_MEDUSA_DUMP_HIDDEN). "
             "When set, skips model loading and trains directly on GPTQ server's hidden states. "
             "Requires <hidden-dump-path>.lm_head_weight.pt in the same location.",
    )
    args = parser.parse_args()

    if args.hidden_dump_path:
        # Fast path: train from GPTQ dump, no model loading required.
        _train_from_gptq_dump(args)
        return

    device = args.device
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 1. Load model and tokenizer
    logger.info("Loading model from %s (dtype=%s) ...", args.model_path, args.dtype)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError:
        logger.error("transformers not installed; run: pip install transformers")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Determine hidden size and scale_width.
    hf_config = model.config
    hidden_size = hf_config.hidden_size
    scale_width = None
    if hasattr(hf_config, "dim_model_base") and hf_config.dim_model_base > 0:
        scale_width = hidden_size / hf_config.dim_model_base
    logger.info(
        "Model: hidden_size=%d, vocab_size=%d, scale_width=%s",
        hidden_size,
        hf_config.vocab_size,
        scale_width,
    )

    # Get lm_head: try model.lm_head, then model.model.embed_tokens for tied models.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None and getattr(hf_config, "tie_word_embeddings", False):
        lm_head = model.model.embed_tokens
    if lm_head is None:
        raise RuntimeError("Cannot find lm_head on the loaded model.")
    logger.info("lm_head: %s", type(lm_head).__name__)

    # 2. Load training data
    samples = load_prompts(args.data_path, max_samples=args.max_samples)

    # 3. Collect (h, label) pairs
    logger.info("Collecting hidden states (max_len=%d)...", args.max_len)
    h_tensor, label_tensor = collect_training_data(
        model, tokenizer, samples,
        max_len=args.max_len,
        device=device,
        scale_width=scale_width,
    )

    # 4. Train W1
    W1 = train(
        h=h_tensor,
        labels=label_tensor,
        lm_head=lm_head,
        hidden_size=hidden_size,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # 5. Save checkpoint with keys matching MedusaHeads.load_trained_weights()
    #    Keys: "heads.{k}.W1.weight" for k in 0..K-1 (K=1 here).
    checkpoint = {"heads.0.W1.weight": W1.weight.cpu()}
    torch.save(checkpoint, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    logger.info(
        "Saved checkpoint to %s (%.1f MiB, key='heads.0.W1.weight', shape=%s)",
        args.output, size_mb, list(W1.weight.shape),
    )
    logger.info(
        "To enable trained heads: export SOAR_MEDUSA_HEAD_PATH=%s", args.output
    )


if __name__ == "__main__":
    main()
