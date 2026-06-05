import pytest
from transformers import GPT2Config, GPT2LMHeadModel


@pytest.fixture
def tiny_config():
    # Small enough to build instantly on CPU with no download.
    return GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=64)


@pytest.fixture
def tiny_base_model(tiny_config):
    return GPT2LMHeadModel(tiny_config)
