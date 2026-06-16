import math

import torch

from lora_finetune.evaluate import compute_eval_loss, perplexity
from lora_finetune.model import build_lora_model


def test_compute_eval_loss_is_finite(tiny_base_model):
    model = build_lora_model(tiny_base_model, r=4)
    ids = torch.randint(0, 256, (2, 16))
    batches = [{"input_ids": ids, "labels": ids}]
    loss = compute_eval_loss(model, batches)
    assert loss > 0 and math.isfinite(loss)


def test_perplexity_matches_exp_loss():
    assert math.isclose(perplexity(0.0), 1.0)
    assert perplexity(2.0) > perplexity(1.0)  # higher loss -> higher perplexity


def test_train_eval_split():
    from lora_finetune.data import train_eval_split

    rows = [{"instruction": str(i), "response": str(i)} for i in range(10)]
    train, ev = train_eval_split(rows, eval_frac=0.2, seed=0)
    assert len(train) == 8 and len(ev) == 2
    # no overlap
    assert not ({r["instruction"] for r in train} & {r["instruction"] for r in ev})
    # eval_frac=0 -> all train
    train2, ev2 = train_eval_split(rows, eval_frac=0.0)
    assert len(train2) == 10 and ev2 == []
