from lora_finetune.model import build_lora_model


def test_adapter_save_roundtrip(tiny_base_model, tmp_path):
    lora = build_lora_model(tiny_base_model, r=4)
    lora.save_pretrained(tmp_path)
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "adapter_model.safetensors").exists() or (
        tmp_path / "adapter_model.bin"
    ).exists()
