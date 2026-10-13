"""Verify first-run and interrupted-download model source selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from word_graph import config


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "model"
        source = patch.object(config, "MODEL_CACHE", self.cache)
        source.start()
        self.addCleanup(source.stop)

    def write_tokenizer_and_config(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.cache / name).write_text("{}", encoding="utf-8")

    def write_shard_index(self):
        (self.cache / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "layer_a": "model-00001-of-00002.safetensors",
            "layer_b": "model-00002-of-00002.safetensors",
        }}), encoding="utf-8")

    def test_absent_config_only_and_partial_downloads_resume_from_model_id(self):
        self.assertEqual(config.default_model_source(), config.MODEL_ID)
        self.cache.mkdir()
        (self.cache / "config.json").write_text("{}", encoding="utf-8")
        self.assertEqual(config.default_model_source(), config.MODEL_ID)
        self.write_tokenizer_and_config()
        self.write_shard_index()
        (self.cache / "model-00001-of-00002.safetensors").touch()
        self.assertEqual(config.default_model_source(), config.MODEL_ID)

    def test_complete_two_shard_download_uses_local_cache(self):
        self.write_tokenizer_and_config()
        self.write_shard_index()
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            (self.cache / name).touch()
        self.assertEqual(config.default_model_source(), str(self.cache))

    def test_corrupt_or_malformed_shard_indexes_resume_download(self):
        self.write_tokenizer_and_config()
        for content in ("{invalid JSON", "{}", '{"weight_map": {}}', '{"weight_map": []}', "[]"):
            with self.subTest(index=content):
                (self.cache / "model.safetensors.index.json").write_text(content, encoding="utf-8")
                self.assertEqual(config.default_model_source(), config.MODEL_ID)
