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

    # Abbreviation lookup for compact prompt format
    _FEATURE_ABBREVIATIONS = {
        "open": "O",
        "high": "H",
        "low": "L",
        "close": "C",
        "volume": "V",
    }

    def __init__(
        self,
        market_data_keys: List[str],
        account_state_labels: List[str],
        action_levels: List[float],
        symbol: str = "BTC/USD",
        execute_on: Union[str, "TimeFrame"] = "1Hour",
        feature_keys: Optional[List[str]] = None,
        prompt_format: str = "verbose",
        price_columns: Optional[List[str]] = None,
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
        self.prompt_format = prompt_format
        self.price_columns = price_columns or ["open", "high", "low", "close"]
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
        """Dispatch to verbose or compact formatter based on self.prompt_format."""
        if self.prompt_format == "compact":
            return self._format_market_data_compact(tensordict)
        return self._format_market_data_verbose(tensordict)

    def _format_market_data_verbose(self, tensordict) -> str:
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

    # --- Compact prompt format helpers ---

    @staticmethod
    def _parse_timeframe_header(key: str) -> str:
        """Convert market data key to compact shorthand.

        Examples:
            market_data_1Hour_24   -> 1H(24)
            market_data_5Minute_12 -> 5M(12)
            market_data_1Day_30    -> 1D(30)
        """
        # Expected format: market_data_{value}{Unit}_{window}
        parts = key.split("_")  # ["market", "data", "1Hour", "24"]
        timeframe_part = parts[2]  # e.g. "1Hour", "5Minute"
        window = parts[3]

        # Extract numeric prefix and unit name
        i = 0
        while i < len(timeframe_part) and (timeframe_part[i].isdigit()):
            i += 1
        value = timeframe_part[:i]
        unit = timeframe_part[i:]  # e.g. "Hour", "Minute", "Day"

        unit_abbrev = unit[0].upper()  # H, M, D
        return f"{value}{unit_abbrev}({window})"

    @classmethod
    def _abbreviate_feature(cls, name: str) -> str:
        """Return abbreviated feature name for compact format."""
        return cls._FEATURE_ABBREVIATIONS.get(name, name.upper())

    def _format_market_data_compact(self, tensordict) -> str:
        """Format market data in compact delta-encoded format.

        Output example:
            1H(24) ref:93600
            O,H,L,C,V
            +8,+505,-5,+499,277
        """
        out = ""
        # Determine close column index for reference price
        try:
            close_idx = self.feature_keys.index("close")
        except ValueError:
            close_idx = 0

        # Determine which column indices are price columns
        price_col_indices = set()
        for i, name in enumerate(self.feature_keys):
            if name in self.price_columns:
                price_col_indices.add(i)

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

            # Compute reference price: first candle's close, rounded to nearest 100
            ref = int(round(float(data[0, close_idx]), -2))

            # Timeframe header
            shorthand = self._parse_timeframe_header(key)
            out += f"{shorthand} ref:{ref}\n"

            # Column header
            abbrevs = [self._abbreviate_feature(name) for name in self.feature_keys]
            out += ",".join(abbrevs) + "\n"

            # Data rows
            for t in range(data.shape[0]):
                fields = []
                for col_idx, val in enumerate(data[t]):
                    if col_idx in price_col_indices:
                        delta = int(round(float(val))) - ref
                        fields.append(f"{delta:+d}")
                    else:
                        fields.append(str(int(round(float(val)))))
                out += ",".join(fields) + "\n"

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
