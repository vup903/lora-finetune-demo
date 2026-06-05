import torch

from lora_finetune.model import build_lora_model


def test_training_steps_produce_finite_loss(tiny_base_model):
    model = build_lora_model(tiny_base_model, r=4)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)

    batch_ids = torch.randint(0, 256, (2, 16))
    losses = []
    for _ in range(3):
        out = model(input_ids=batch_ids, labels=batch_ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())

    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_only_lora_params_require_grad(tiny_base_model):
    model = build_lora_model(tiny_base_model, r=4)
    grad_named = [n for n, p in model.named_parameters() if p.requires_grad]
    assert grad_named  # something trains
    assert all("lora" in n.lower() for n in grad_named)  # ...and only the adapters
