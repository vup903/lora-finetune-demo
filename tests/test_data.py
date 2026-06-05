from pathlib import Path

from lora_finetune.data import format_example, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_format_example():
    s = format_example({"instruction": "Say hi", "response": "Hi!"})
    assert "### Instruction:" in s
    assert "### Response:" in s
    assert "Say hi" in s and "Hi!" in s


def test_load_bundled_dataset():
    rows = load_jsonl(ROOT / "data" / "train.jsonl")
    assert len(rows) >= 10
    assert all("instruction" in r and "response" in r for r in rows)
