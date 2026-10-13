"""Load one local instruct model and constrain generation to JSON schemas."""
from __future__ import annotations

import os

from .config import MODEL_CACHE, MODEL_ID
from .topology import GraphStructureFirstMixin


class GraphGenerator(GraphStructureFirstMixin):
    """Shared inference engine for topology and individual relation descriptions."""

    def __init__(self, model_source: str, quantization: str = "bf16", seed: int = 42):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required; run setup_environment.ps1.")
        if quantization not in {"bf16", "4bit"}:
            raise ValueError("quantization must be bf16 or 4bit")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This model configuration requires a GPU with BF16 support.")
        if model_source == MODEL_ID:
            from huggingface_hub import snapshot_download
            model_source = snapshot_download(
                MODEL_ID, local_dir=str(MODEL_CACHE),
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"],
            )
        set_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        options = dict(torch_dtype=torch.bfloat16, device_map={"": 0},
                       attn_implementation="sdpa", trust_remote_code=False)
        if quantization == "4bit":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_source, **options).eval()
        self.model.config.use_cache = True
        self.constraint_tokenizer = build_token_enforcer_tokenizer_data(self.tokenizer)

    def _decode(self, messages: list[dict], schema: dict, budget: int,
                temperature: float, sample: bool) -> str:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if inputs.input_ids.shape[1] + budget > self.model.config.max_position_embeddings:
            raise ValueError("Input plus output exceeds the model context; reduce the term list or term length.")
        parser = JsonSchemaParser(schema, CharacterLevelParserConfig(
            max_consecutive_whitespaces=2, force_json_field_order=True,
        ))
        prefix = build_transformers_prefix_allowed_tokens_fn(self.constraint_tokenizer, parser)
        options = dict(max_new_tokens=budget, do_sample=sample, prefix_allowed_tokens_fn=prefix,
                       pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        if sample:
            options.update(temperature=temperature, top_p=0.9)
        else:
            options.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
