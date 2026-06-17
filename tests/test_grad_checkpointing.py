import torch

from lora_finetune.model import build_lora_model, enable_gradient_checkpointing


def test_training_step_works_with_grad_checkpointing(tiny_base_model):
    model = build_lora_model(tiny_base_model, r=4)
    enable_gradient_checkpointing(model)
    model.train()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    ids = torch.randint(0, 256, (2, 16))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    optimizer.step()

    assert torch.isfinite(out.loss)
    # at least one LoRA param actually received a gradient through the checkpointed graph
    assert any(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)
