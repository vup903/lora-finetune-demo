"""Model setup: load a base causal LM and attach LoRA / QLoRA adapters.

The LoRA logic here is shared by `train.py` and the unit tests, so the exact
adapter wiring that ships is the wiring that's tested.
"""
from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPT-2 family fuses Q/K/V into a single Conv1D named ``c_attn`` — the standard
# LoRA target for these models. Override via build_lora_model(target_modules=...).
DEFAULT_TARGET_MODULES = ["c_attn"]


def load_base_model(model_name: str, quantize_4bit: bool = False):
    """Load a base model + tokenizer.

    If ``quantize_4bit`` and CUDA is available, load in 4-bit (QLoRA) via
    bitsandbytes; otherwise load normally (LoRA on CPU/GPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def build_lora_model(base_model, r: int = 8, alpha: int = 16, dropout: float = 0.05,
                     target_modules: list[str] | None = None,
                     fan_in_fan_out: bool = True):
    """Wrap a base model with a LoRA adapter and return the PEFT model.

    ``fan_in_fan_out`` defaults to True because the default target (GPT-2's
    ``c_attn``) is a ``Conv1D`` layer, whose weights are stored transposed.
    Set it False for models built from ``nn.Linear`` projections (e.g. Llama /
    Qwen ``q_proj``/``k_proj``/...).
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        fan_in_fan_out=fan_in_fan_out,
        bias="none",
    )
    return get_peft_model(base_model, config)


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to cut activation memory (trades extra compute).

    Also enables input grads (needed when the base is frozen, as with LoRA, so
    gradients can flow back through checkpointed segments) and disables the
    kv-cache (incompatible with checkpointing during training).
    """
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def merge_adapter(peft_model):
    """Fold LoRA weights into the base model and return a standalone model.

    After merging there are no PEFT/LoRA layers left, so the result can be saved
    and loaded as a normal Transformers model (no adapter, no PEFT dependency at
    inference) — useful for deployment.
    """
    return peft_model.merge_and_unload()


def count_trainable(model) -> tuple[int, int]:
    """Return (trainable_param_count, total_param_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
