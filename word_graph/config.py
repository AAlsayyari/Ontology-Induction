"""Project paths, model defaults, and shared term normalization."""
from __future__ import annotations

import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_CACHE = ROOT / ".cache" / "model"


def default_model_source() -> str:
    # An interrupted first download may leave config.json before the weights.
    # Returning the model ID lets snapshot_download resume those missing files.
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if not all((MODEL_CACHE / name).is_file() for name in required):
        return MODEL_ID
    try:
        index = MODEL_CACHE / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                return MODEL_ID
            shards = set(weight_map.values())
        else:
            shards = {"model.safetensors"}
        if shards and all((MODEL_CACHE / shard).is_file() for shard in shards):
            return str(MODEL_CACHE)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return MODEL_ID


def normalize(term: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", term).split()).casefold()


def make_messages(terms: list[str], *, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Terms:\n" + json.dumps(terms, ensure_ascii=False)},
    ]
