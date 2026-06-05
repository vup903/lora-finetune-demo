from lora_finetune.model import build_lora_model, count_trainable


def test_lora_trains_small_fraction(tiny_base_model):
    base_total = sum(p.numel() for p in tiny_base_model.parameters())
    lora = build_lora_model(tiny_base_model, r=4)
    trainable, total = count_trainable(lora)
    # The whole point of LoRA: only a tiny fraction of params is trainable.
    assert 0 < trainable < 0.1 * base_total
    assert total >= base_total


def test_lora_injects_adapter_modules(tiny_base_model):
    lora = build_lora_model(tiny_base_model, r=4)
    names = [n for n, _ in lora.named_modules()]
    assert any("lora_A" in n for n in names)
    assert any("lora_B" in n for n in names)
