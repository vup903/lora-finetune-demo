# lora-finetune-demo

Parameter-efficient fine-tuning of an open causal LM with **LoRA** (and optional
4-bit **QLoRA**) using **PEFT + Transformers**. Freezes the base model and trains
only small low-rank adapters — a tiny fraction of parameters.

## What it demonstrates
- **LoRA setup** ([`model.py`](src/lora_finetune/model.py)): `LoraConfig` injected into the attention projection; `count_trainable` shows only ~a fraction of params train.
- **QLoRA path**: 4-bit base load via `BitsAndBytesConfig` + `prepare_model_for_kbit_training` (CUDA only) — same adapter code, quantized base.
- **SFT data pipeline** ([`data.py`](src/lora_finetune/data.py)): instruction/response formatting, tokenization, and pad-masked labels (`-100`).
- **Training** ([`train.py`](train.py)): `transformers.Trainer` over a bundled instruction set; saves a portable LoRA adapter.
- **Inference** ([`generate.py`](generate.py)): load base + adapter and generate.
- **Hermetic tests + CI**: the suite builds a *tiny model from config* (no download, no GPU, no network), applies LoRA, runs real training steps, and checks the adapter saves — so [CI](.github/workflows/ci.yml) is fast and offline.

## Run the tests (no GPU, no network)
```bash
pip install -e ".[dev]"
pytest -q
```

## Fine-tune for real
```bash
pip install -e .
python train.py                 # LoRA on distilgpt2 (CPU-friendly)
python generate.py --prompt "What is LoRA in one sentence?"
```

4-bit **QLoRA** (needs a CUDA GPU + bitsandbytes):
```bash
pip install -e ".[qlora]"
python train.py --model Qwen/Qwen2.5-0.5B --qlora --linear \
  --target-modules q_proj k_proj v_proj o_proj --epochs 3
```
(`--linear` + `--target-modules` because Qwen/Llama use `nn.Linear` projections;
the GPT-2 default targets `c_attn`, a `Conv1D`.)

## Layout
```
src/lora_finetune/
  model.py   # load_base_model (LoRA/QLoRA) + build_lora_model + count_trainable
  data.py    # instruction formatting + tokenization (pad-masked labels)
train.py     # LoRA/QLoRA fine-tuning with transformers.Trainer
generate.py  # load base + adapter and generate
data/train.jsonl   # small bundled instruction dataset
tests/             # hermetic (tiny model from config — no download/GPU)
```
