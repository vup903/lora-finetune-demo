"""Dataset prep: format instruction/response pairs and tokenize for causal LM SFT."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def format_example(example: dict) -> str:
    """Render one {instruction, response} pair into the training prompt."""
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"].strip(),
        response=example["response"].strip(),
    )


def train_eval_split(rows: list[dict], eval_frac: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Shuffle and split rows into (train, eval). eval_frac<=0 returns (rows, [])."""
    if eval_frac <= 0 or len(rows) < 2:
        return list(rows), []
    import random

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    k = max(1, int(len(shuffled) * eval_frac))
    return shuffled[k:], shuffled[:k]


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_dataset(tokenizer, rows: list[dict], max_len: int = 128) -> Dataset:
    """Tokenize formatted examples into input_ids / attention_mask / labels.

    Pad positions are masked to -100 so they don't contribute to the loss.
    """
    texts = [format_example(r) + tokenizer.eos_token for r in rows]

    def _tokenize(batch):
        enc = tokenizer(
            batch["text"], truncation=True, max_length=max_len, padding="max_length"
        )
        enc["labels"] = [
            [tok if mask == 1 else -100 for tok, mask in zip(ids, attn)]
            for ids, attn in zip(enc["input_ids"], enc["attention_mask"])
        ]
        return enc

    ds = Dataset.from_dict({"text": texts})
    return ds.map(_tokenize, batched=True, remove_columns=["text"])
