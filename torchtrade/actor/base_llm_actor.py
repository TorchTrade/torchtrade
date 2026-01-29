"""Base LLM Actor with environment-driven prompt construction and action extraction."""
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class BaseLLMActor(ABC):
    """
    Base class for LLM-based trading actors.

    All configuration is derived from the environment — no hardcoded action
    mappings or account state labels.

    Args:
        market_data_keys: From env.market_data_keys (e.g. ["market_data_1Hour_48"]).
        account_state_labels: From env.account_state (e.g. ["exposure_pct", ...]).
        action_levels: From env.action_levels (e.g. [-1, 0, 1] or [0, 0.5, 1]).
        symbol: Trading symbol (e.g. "BTC/USD").
        execute_on: Execution timeframe (e.g. "1Hour").
        feature_keys: Column names in market data tensors.
        debug: Enable debug output.
    """

    def __init__(
        self,
        market_data_keys: List[str],
        account_state_labels: List[str],
        action_levels: List[float],
        symbol: str = "BTC/USD",
        execute_on: Union[str, "TimeFrame"] = "1Hour",
        feature_keys: Optional[List[str]] = None,
        debug: bool = False,
    ):
        self.market_data_keys = market_data_keys
        self.account_state_labels = account_state_labels
        self.action_levels = action_levels
        self.symbol = symbol
        # Accept TimeFrame objects — format as e.g. "1Hour"
        if hasattr(execute_on, 'value') and hasattr(execute_on, 'unit'):
            self.execute_on = f"{execute_on.value}{execute_on.unit.name}"
        else:
            self.execute_on = str(execute_on)
        self.feature_keys = feature_keys or ["open", "high", "low", "close", "volume"]
        self.debug = debug

        # Build action descriptions from action_levels
        self._action_descriptions = self._build_action_descriptions()

        # Pre-compile regex
        self._answer_pattern = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE | re.DOTALL)

    def _build_action_descriptions(self) -> List[str]:
        """Build human-readable descriptions for each action index."""
        descriptions = []
        for i, level in enumerate(self.action_levels):
            pct = level * 100
            if level == 0:
                descriptions.append(f"Action {i} → target 0% (flat/no position)")
            elif level > 0:
                descriptions.append(f"Action {i} → target +{pct:.0f}% (long)")
            else:
                descriptions.append(f"Action {i} → target {pct:.0f}% (short)")
        return descriptions

    def _build_system_prompt(self) -> str:
        """Build system prompt dynamically from env configuration."""
        action_list = "\n".join(f"  {d}" for d in self._action_descriptions)
        return (
            f"You are a trading agent for {self.symbol} on the {self.execute_on} timeframe.\n"
            f"At each step you receive account state and market data.\n\n"
            f"Available actions (target exposure levels):\n{action_list}\n\n"
            f"- Think step-by-step inside <think></think>.\n"
            f"- Output your chosen action number in exact format: <answer>N</answer>\n"
            f"  where N is the action number (0 to {len(self.action_levels) - 1})."
        )

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts. Subclasses implement this."""

    def __call__(self, tensordict):
        return self.forward(tensordict)

    def forward(self, tensordict):
        """Main forward pass: construct prompts, generate, extract action, save to tensordict."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._construct_user_prompt(tensordict)

        if self.debug:
            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("\nUSER PROMPT:")
            print(user_prompt)
            print("=" * 80)

        response = self.generate(system_prompt, user_prompt)

        if self.debug:
            print("RESPONSE:")
            print(response)
            print("=" * 80)

        action_idx = self._extract_action(response)

        tensordict.set("action", torch.tensor(action_idx, dtype=torch.long))
        tensordict.set("thinking", response)
        tensordict.set("system_prompt", system_prompt)
        tensordict.set("user_prompt", user_prompt)

        return tensordict

    # --- Prompt construction ---

    def _construct_user_prompt(self, tensordict) -> str:
        return self._construct_account_state(tensordict) + self._construct_market_data(tensordict)

    def _construct_account_state(self, tensordict) -> str:
        account_state = tensordict.get("account_state")
        if account_state.dim() == 2:
            account_state = account_state.squeeze(0)

        out = "Current account state:\n"
        for idx, label in enumerate(self.account_state_labels):
            out += f"  {label}: {round(account_state[idx].item(), 4)}\n"
        out += "\n---\n"
        return out

    def _construct_market_data(self, tensordict) -> str:
        out = "Current market data:\n\n"
        for key in self.market_data_keys:
            if key not in tensordict:
                continue

            data = tensordict[key].cpu().numpy()
            if data.ndim == 3:
                data = data.squeeze(0)
            if data.ndim != 2 or data.shape[1] != len(self.feature_keys):
                if self.debug:
                    print(f"[Warning] Unexpected market data shape for {key}: {data.shape}")
                continue

            out += f"{key}:\n\n"
            header = " | ".join(f"{k:>8}" for k in self.feature_keys)
            out += header + "\n\n"
            for t in range(data.shape[0]):
                row = " | ".join(f"{v:8.1f}" for v in data[t])
                out += row + "\n"
            out += "\n"

        return out

    # --- Action extraction ---

    def _extract_action(self, response: str) -> int:
        """Extract action index from <answer>N</answer> tag."""
        match = self._answer_pattern.search(response)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(self.action_levels):
                return idx
            if self.debug:
                print(f"[Warning] Action {idx} out of range, defaulting to 0")
            return 0

        if self.debug:
            logger.warning("No <answer> tag found in response, defaulting to action 0")
        return 0
