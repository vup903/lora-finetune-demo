"""Load the base model + trained LoRA adapter and generate a response.

    python generate.py --prompt "What is LoRA in one sentence?"
"""
from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_finetune.data import PROMPT_TEMPLATE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="distilgpt2")
    p.add_argument("--adapter", default="out")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    text = PROMPT_TEMPLATE.format(instruction=args.prompt, response="")
    inputs = tokenizer(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                         pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
