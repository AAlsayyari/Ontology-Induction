"""Check the runtime loader contract with mocked GPU/model dependencies."""
import sys
import unittest
from unittest.mock import MagicMock, patch

from word_graph.inference import GraphGenerator
from word_graph.config import MODEL_CACHE, MODEL_ID


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.is_bf16_supported.return_value = True
        self.transformers = MagicMock()
        self.model = MagicMock()
        self.model.eval.return_value = self.model
        self.transformers.AutoModelForCausalLM.from_pretrained.return_value = self.model
        self.transformers.AutoTokenizer.from_pretrained.return_value.pad_token = None
        self.transformers.AutoTokenizer.from_pretrained.return_value.eos_token = "<eos>"
        self.integration = MagicMock()
        self.dependencies = patch.dict(sys.modules, {
            "torch": self.torch,
            "transformers": self.transformers,
            "lmformatenforcer.integrations.transformers": self.integration,
        })
        self.dependencies.start()
        self.addCleanup(self.dependencies.stop)

    def test_runtime_loads_base_model_and_its_tokenizer_without_an_adapter(self):
        generator = GraphGenerator("local-model")
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "local-model", trust_remote_code=False)
        args, kwargs = self.transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(args, ("local-model",))
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertNotIn("quantization_config", kwargs)
        self.assertIs(generator.model, self.model)
        self.assertTrue(generator.model.config.use_cache)
        self.assertEqual(generator.tokenizer.pad_token, "<eos>")

    def test_four_bit_loading_is_an_explicit_option(self):
        GraphGenerator("local-model", quantization="4bit")
        self.transformers.BitsAndBytesConfig.assert_called_once()
        self.assertTrue(self.transformers.BitsAndBytesConfig.call_args.kwargs["load_in_4bit"])
        options = self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertIn("quantization_config", options)

    def test_first_run_downloads_then_loads_the_returned_local_folder(self):
        hub = MagicMock()
        hub.snapshot_download.return_value = "downloaded-model-folder"
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            GraphGenerator(MODEL_ID)
        args, options = hub.snapshot_download.call_args
        self.assertEqual(args, (MODEL_ID,))
        self.assertEqual(options["local_dir"], str(MODEL_CACHE))
        self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            "downloaded-model-folder", trust_remote_code=False)
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.args,
                         ("downloaded-model-folder",))

    def test_missing_bf16_support_fails_before_loading_weights(self):
        self.torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            GraphGenerator("local-model")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_missing_cuda_fails_before_loading_weights(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            GraphGenerator("local-model")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_invalid_quantization_fails_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError, "quantization"):
            GraphGenerator("local-model", quantization="unsupported")
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
