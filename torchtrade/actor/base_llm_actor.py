"""Base LLM Actor with shared prompt construction, action extraction, and forward loop."""
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class BaseLLMActor(ABC):
    """
    Base class for LLM-based trading actors.

    Handles prompt construction from TensorDicts, action extraction from
    <answer> tags, and saving prompts/responses for SFT reproducibility.
    Subclasses only need to implement `generate()`.

    Args:
        debug: Enable debug output.
        action_space_type: Type of action space ("standard", "sltp", "futures_sltp").
        action_map: Action map for SLTP environments (dict[int, tuple]).
        symbol: Trading symbol (e.g., "BTC/USD").
        execute_on: Execution timeframe (e.g., "5Minute").
        feature_keys: Column names in market data tensors.
    """

    def __init__(
        self,
        debug: bool = False,
        action_space_type: str = "standard",
        action_map: Optional[Dict[int, Tuple[Optional[str], Optional[float], Optional[float]]]] = None,
        symbol: str = "BTC/USD",
        execute_on: str = "5Minute",
        feature_keys: Optional[list] = None,
    ):
        self.debug = debug
        self.action_space_type = action_space_type
        self.action_map = action_map
        self.symbol = symbol
        self.execute_on = execute_on
        self.feature_keys = feature_keys or ["close", "open", "high", "low", "volume"]

        # Market data keys (auto-detected from first tensordict)
        self.market_data_keys = []

        # System prompts
        self.system_prompt_standard = """
You are a disciplined trading agent for {symbol} on the {execute_on} timeframe.
At each step, you receive the latest account state and market data.
You must choose exactly one action: buy, sell, or hold.

- Base your decision on the provided data.
- Think step-by-step inside <think></think>.
- Output your final action in exact format: <answer>buy</answer>, <answer>sell</answer>, or <answer>hold</answer>.
        """

        self.system_prompt_futures = """
You are a futures trading agent for {symbol} on the {execute_on} timeframe with {leverage}x leverage.
At each step, you receive account state (including margin and liquidation info) and market data.
{action_instructions}

- Consider liquidation risk when position_size != 0
- Leverage amplifies both gains and losses
- Think step-by-step inside <think></think>.
- Output your final action: <answer>{action_format}</answer>
        """

        # Account state definitions
        self.account_state_standard = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnlpct", "holding_time"
        ]
        self.account_state_futures = [
            "cash", "position_size", "position_value", "entry_price",
            "current_price", "unrealized_pnl_pct", "leverage",
            "margin_ratio", "liquidation_price", "holding_time"
        ]

        # Action mappings
        if action_space_type == "standard":
            self.action_dict = {"buy": 2, "sell": 0, "hold": 1}
        elif action_space_type in ["sltp", "futures_sltp"]:
            if action_map is None:
                raise ValueError("action_map required for SLTP action spaces")
            self.action_dict = None

        # Pre-compile regex
        self._answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response given system and user prompts. Subclasses implement this."""

    def __call__(self, tensordict):
        return self.forward(tensordict)

    def forward(self, tensordict):
        """Main forward pass: construct prompts, generate, extract action, save to tensordict."""
        system_prompt = self._format_system_prompt(tensordict)
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

        action_str = self._extract_action(response)
        action_idx = self._map_action_to_index(action_str)

        tensordict.set("action", torch.tensor(action_idx, dtype=torch.long))
        tensordict.set("thinking", response)
        tensordict.set("system_prompt", system_prompt)
        tensordict.set("user_prompt", user_prompt)

        return tensordict

    # --- Prompt construction ---

    def _construct_user_prompt(self, tensordict) -> str:
        """Construct full user prompt from tensordict."""
        return self._construct_account_state(tensordict) + self._construct_market_data(tensordict)

    def _construct_account_state(self, tensordict) -> str:
        account_state = tensordict.get("account_state")
        state_size = account_state.shape[-1]
        state_labels = self._get_account_state_labels(state_size)

        if account_state.dim() == 2:
            account_state = account_state.squeeze(0)

        out = "Current account state:\n"
        for idx, label in enumerate(state_labels):
            out += f"{label}: {round(account_state[idx].item(), 4)}\n"
        out += "\n---\n"
        return out

    def _construct_market_data(self, tensordict) -> str:
        # Auto-detect market data keys on first call
        if not self.market_data_keys:
            self.market_data_keys = [k for k in tensordict.keys() if k.startswith("market_data_")]

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
            header = " | ".join([f"{k:>8}" for k in self.feature_keys])
            out += header + "\n\n"
            for t in range(data.shape[0]):
                row = " | ".join([f"{v:8.1f}" for v in data[t]])
                out += row + "\n"
            out += "\n"

        return out

    def _format_system_prompt(self, tensordict) -> str:
        account_state = tensordict.get("account_state")
        state_size = account_state.shape[-1]
        action_instructions, action_format = self._build_action_space_description()

        if state_size == 7 and self.action_space_type == "standard":
            return self.system_prompt_standard.format(
                symbol=self.symbol, execute_on=self.execute_on,
            )

        leverage = self._extract_leverage(account_state, state_size)
        return self.system_prompt_futures.format(
            symbol=self.symbol, execute_on=self.execute_on,
            leverage=leverage, action_instructions=action_instructions,
            action_format=action_format,
        )

    def _build_action_space_description(self) -> Tuple[str, str]:
        if self.action_space_type == "standard":
            return "You must choose exactly one action: buy, sell, or hold.", "buy/sell/hold"

        instructions = "Available actions:\n"
        for idx, (side, sl, tp) in self.action_map.items():
            if side is None:
                instructions += f"  {idx}: Hold position\n"
            elif side == "close":
                instructions += f"  {idx}: Close current position\n"
            else:
                instructions += f"  {idx}: {side.capitalize()} SL={sl*100:.1f}% TP={tp*100:.1f}%\n"
        instructions += "\nYou must choose exactly one action by number (e.g., 0, 1, 2, ...)."
        return instructions, "action_number"

    def _get_account_state_labels(self, state_size: int) -> list:
        if state_size == 7:
            return self.account_state_standard
        elif state_size == 10:
            return self.account_state_futures
        raise ValueError(f"Unknown account state size: {state_size}. Expected 7 or 10.")

    def _extract_leverage(self, account_state: torch.Tensor, state_size: int) -> int:
        if state_size != 10:
            return 1
        val = account_state[0, 6].item() if account_state.dim() == 2 else account_state[6].item()
        return int(val) if val > 0 else 1

    # --- Action extraction ---

    def _extract_action(self, response: str) -> str:
        match = self._answer_pattern.search(response)
        if match:
            return match.group(1).strip()
        if self.debug:
            logger.warning("No <answer> tag found in response, using default action")
        return "hold" if self.action_space_type == "standard" else "0"

    def _map_action_to_index(self, action_str: str) -> int:
        if self.action_space_type == "standard":
            action_lower = action_str.lower()
            if action_lower in self.action_dict:
                return self.action_dict[action_lower]
            if self.debug:
                print(f"[Warning] Unknown action '{action_str}', defaulting to hold")
            return 1  # hold

        # SLTP: numeric action
        try:
            action_idx = int(action_str)
            if action_idx in self.action_map:
                return action_idx
        except ValueError:
            pass
        if self.debug:
            print(f"[Warning] Invalid SLTP action '{action_str}', defaulting to 0")
        return 0
