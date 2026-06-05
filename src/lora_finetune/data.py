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
