"""Frontier (API-based) LLM Actor using OpenAI-compatible APIs."""
from typing import Optional

from torchtrade.actor.base_llm_actor import BaseLLMActor


class FrontierLLMActor(BaseLLMActor):
    """
    LLM trading actor that calls an OpenAI-compatible API.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "gpt-5-nano").
        api_key: OpenAI API key. If None, reads OPENAI_API_KEY from .env.
        All other args are inherited from BaseLLMActor.
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model

        from openai import OpenAI
        if api_key is None:
            from dotenv import dotenv_values
            api_key = dotenv_values(".env").get("OPENAI_API_KEY")
        self.llm = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text
