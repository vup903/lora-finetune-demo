import torch

from lora_finetune.model import build_lora_model, merge_adapter


def test_merge_removes_lora_layers(tiny_base_model):
    lora = build_lora_model(tiny_base_model, r=4)
    assert any("lora_" in n for n, _ in lora.named_modules())  # adapters present before

    merged = merge_adapter(lora)
    assert not any("lora_" in n for n, _ in merged.named_modules())  # ...gone after merge


def test_merged_model_still_runs(tiny_base_model):
    merged = merge_adapter(build_lora_model(tiny_base_model, r=4))
    out = merged(input_ids=torch.randint(0, 256, (1, 8)))
    assert out.logits.shape[-1] == 256  # vocab preserved; forward works
