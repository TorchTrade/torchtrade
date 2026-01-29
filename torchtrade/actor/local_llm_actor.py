"""Local LLM Actor using vllm or transformers backends."""
import logging
from typing import Optional

import torch

from torchtrade.actor.base_llm_actor import BaseLLMActor

logger = logging.getLogger(__name__)


class LocalLLMActor(BaseLLMActor):
    """
    LLM trading actor that runs models locally via vllm or transformers.

    Args:
        model: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
        backend: Inference backend ("vllm" or "transformers").
        device: Device for inference ("cuda", "cpu", "mps").
        quantization: Quantization mode (None, "4bit", "8bit").
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0).
        All other args are inherited from BaseLLMActor.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        backend: str = "vllm",
        device: str = "cuda",
        quantization: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model
        self.backend = backend
        self.device = device
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = None
        self.tokenizer = None
        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "vllm":
            self._initialize_vllm()
        elif self.backend == "transformers":
            self._initialize_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'vllm' or 'transformers'")

    def _initialize_vllm(self):
        try:
            from vllm import LLM, SamplingParams
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=["</answer>"],
            )
            kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            }
            if self.quantization == "4bit":
                kwargs["quantization"] = "bitsandbytes"
                kwargs["load_format"] = "bitsandbytes"
            elif self.quantization == "8bit":
                kwargs["quantization"] = "bitsandbytes_8bit"

            self.llm = LLM(**kwargs)
        except ImportError:
            logger.warning("vllm not available, falling back to transformers")
            self.backend = "transformers"
            self._initialize_transformers()

    def _initialize_transformers(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            model_kwargs = {"trust_remote_code": True}
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            elif self.quantization == "8bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "auto" if (self.device == "cuda" and torch.cuda.is_available()) else self.device

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.llm = pipeline(
                "text-generation", model=model, tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens, temperature=self.temperature,
                do_sample=self.temperature > 0,
            )
        except ImportError as e:
            raise ImportError(
                "Neither vllm nor transformers available. "
                "Install with: pip install 'torchtrade[llm]'"
            ) from e

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = None
        if self.backend == "vllm" and hasattr(self.llm, "get_tokenizer"):
            tokenizer = self.llm.get_tokenizer()
        elif self.backend == "transformers" and self.tokenizer:
            tokenizer = self.tokenizer

        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return f"{system_prompt}\n\n{user_prompt}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        prompt = self._format_chat_prompt(system_prompt, user_prompt)
        if self.backend == "vllm":
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            outputs = self.llm(
                prompt, max_new_tokens=self.max_tokens,
                temperature=self.temperature, do_sample=self.temperature > 0,
                return_full_text=False,
            )
            return outputs[0]["generated_text"]
