"""Merge a trained LoRA adapter into the base model and save a standalone model.

    python merge.py --base distilgpt2 --adapter out --out merged

The output in `merged/` loads as a normal Transformers model — no PEFT needed
at inference, which is convenient for deployment.
"""
from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_finetune.model import merge_adapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="distilgpt2")
    p.add_argument("--adapter", default="out")
    p.add_argument("--out", default="merged")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)

    merged = merge_adapter(model)
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved merged standalone model + tokenizer to '{args.out}/'")


if __name__ == "__main__":
    main()
