"""Evaluation: held-out loss + perplexity for a (LoRA) causal LM.

Perplexity = exp(mean cross-entropy). It's the standard intrinsic metric for a
language model — lower means the model is less surprised by held-out text — so
it's the natural signal for whether a fine-tune is actually learning.
"""
from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator


@torch.no_grad()
def compute_eval_loss(model, batches) -> float:
    """Mean causal-LM loss over an iterable of batches.

    Each batch is a dict with at least ``input_ids`` (and usually ``labels`` and
    ``attention_mask``) as tensors.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in batches:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels", batch["input_ids"]),
        )
        total += float(out.loss)
        n += 1
    return total / max(n, 1)


def perplexity(loss: float) -> float:
    """Convert a mean cross-entropy loss to perplexity."""
    return math.exp(loss)


def evaluate_dataset(model, dataset, batch_size: int = 2) -> dict:
    """Evaluate a tokenized HF dataset; returns {'eval_loss', 'perplexity'}."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    loss = compute_eval_loss(model, loader)
    return {"eval_loss": loss, "perplexity": perplexity(loss)}
