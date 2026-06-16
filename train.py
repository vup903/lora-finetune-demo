"""Fine-tune an open causal LM with LoRA (or 4-bit QLoRA) on a small SFT set.

Examples:
    python train.py                       # LoRA on distilgpt2 (CPU-friendly)
    python train.py --model Qwen/Qwen2.5-0.5B --epochs 3
    python train.py --qlora               # 4-bit QLoRA (needs CUDA + bitsandbytes)
"""
from __future__ import annotations

import argparse

from transformers import Trainer, TrainingArguments, default_data_collator

from lora_finetune.data import build_dataset, load_jsonl, train_eval_split
from lora_finetune.evaluate import evaluate_dataset
from lora_finetune.model import build_lora_model, count_trainable, load_base_model


def parse_args():
    p = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning demo")
    p.add_argument("--model", default="distilgpt2", help="base model id")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--out", default="out")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--r", type=int, default=8, help="LoRA rank")
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-frac", type=float, default=0.0,
                   help="Fraction of examples held out to report eval loss + perplexity after training")
    p.add_argument("--qlora", action="store_true", help="4-bit QLoRA (CUDA + bitsandbytes)")
    p.add_argument(
        "--target-modules", nargs="+", default=None,
        help="LoRA target modules (default: GPT-2 'c_attn'; e.g. for Qwen/Llama pass "
             "q_proj k_proj v_proj o_proj)",
    )
    p.add_argument(
        "--linear", action="store_true",
        help="Targets are nn.Linear (Llama/Qwen) — sets fan_in_fan_out=False. "
             "Omit for GPT-2 family (Conv1D).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_base_model(args.model, quantize_4bit=args.qlora)
    model = build_lora_model(
        model, r=args.r, target_modules=args.target_modules,
        fan_in_fan_out=not args.linear,
    )

    trainable, total = count_trainable(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    rows = load_jsonl(args.data)
    train_rows, eval_rows = train_eval_split(rows, eval_frac=args.eval_frac)
    dataset = build_dataset(tokenizer, train_rows, max_len=args.max_len)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()

    if eval_rows:
        eval_dataset = build_dataset(tokenizer, eval_rows, max_len=args.max_len)
        metrics = evaluate_dataset(model, eval_dataset, batch_size=args.batch_size)
        print(f"Held-out eval: loss={metrics['eval_loss']:.4f}  perplexity={metrics['perplexity']:.2f}  (n={len(eval_rows)})")

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter + tokenizer to '{args.out}/'")


if __name__ == "__main__":
    main()
