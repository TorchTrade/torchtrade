"""Base LLM Actor with environment-driven prompt construction and action extraction."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from torchtrade.actor.parsers import extract_action, parse_tool_calls
from torchtrade.actor.tools import Tool

if TYPE_CHECKING:
    from tensordict import TensorDict

SystemPrompt = Union[str, Callable[["BaseLLMActor"], str]]
UserPromptFn = Callable[["BaseLLMActor", "TensorDict"], str]


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
        system_prompt: Optional override for the system prompt. Pass a string
            for a static replacement, or a callable `f(actor) -> str` for
            dynamic construction (e.g. to prepend extra context to the
            default: `lambda a: extra + a._build_system_prompt()`).
            If None, the default prompt built from `symbol`, `execute_on`,
            and `action_levels` is used.
        user_prompt_fn: Optional callable `f(actor, tensordict) -> str` that
            replaces the default user prompt construction. Useful for custom
            data layouts or formats. If None, the default prompt (account
            state + market data tables) is used. The callable receives the
            live tensordict for the current step — read freely, but do NOT
            mutate it (writes will leak into observation/action keys).
        tools: Optional list of `Tool` instances the LLM may call mid-reasoning
            (e.g. `GoogleNewsTool`). When non-empty, the system prompt gains a
            tool-calling protocol block. If None/empty, prompt behavior is
            unchanged (no-tools path).
        max_tool_iters: Maximum number of tool-call round-trips per step
            (advertised in the prompt and enforced by the tool loop).
    """

    def __init__(
        self,
        market_data_keys: List[str],
        account_state_labels: List[str],
        action_levels: List[float],
        symbol: str = "BTC/USD",
        execute_on: object = "1Hour",
        feature_keys: Optional[List[str]] = None,
        action_descriptions: Optional[List[str]] = None,
        debug: bool = False,
        system_prompt: Optional[SystemPrompt] = None,
        user_prompt_fn: Optional[UserPromptFn] = None,
        tools: Optional[List[Tool]] = None,
        max_tool_iters: int = 3,
    ):
        self.market_data_keys = market_data_keys
        self.account_state_labels = account_state_labels
        self.action_levels = action_levels
        self.symbol = symbol
        # Accept TimeFrame objects from env configs — they normalize execute_on
        # in __post_init__, so callers passing config.execute_on get a TimeFrame.
        # obs_key_freq() renders "1Hour"-style strings; str() would leak repr.
        self.execute_on = (
            execute_on.obs_key_freq() if hasattr(execute_on, "obs_key_freq") else str(execute_on)
        )
        self.feature_keys = feature_keys or ["open", "high", "low", "close", "volume"]
        self.debug = debug
        self._system_prompt_override = system_prompt
        self._user_prompt_fn = user_prompt_fn
        self.tools = list(tools) if tools else []
        self._tools_by_name = {t.name: t for t in self.tools}
        self.max_tool_iters = max_tool_iters

        # Action descriptions: explicit override (e.g. for binary up/down envs)
        # falls back to the auto-generated "target exposure" language.
        self._action_descriptions = (
            list(action_descriptions)
            if action_descriptions is not None
            else self._build_action_descriptions()
        )

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
    def generate_batch(self, system_prompt: str, user_prompts: list) -> list:
        """Generate one response per user prompt.

        Given a shared system prompt and a list of N user prompts, return a
        list of N response strings (same order). Subclasses implement this;
        N=1 is the single-observation case.
        """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Single-prompt convenience wrapping generate_batch (kept for callers)."""
        return self.generate_batch(system_prompt, [user_prompt])[0]

    def __call__(self, tensordict):
        return self.forward(tensordict)

    def _build_user_prompt(self, tensordict) -> str:
        return (
            self._user_prompt_fn(self, tensordict)
            if self._user_prompt_fn is not None
            else self._construct_user_prompt(tensordict)
        )

    def forward(self, tensordict):
        """Construct N prompts, generate, extract N actions, write to tensordict.

        Handles an unbatched tensordict (batch_size=[]) as N=1 with scalar
        outputs identical to the pre-batching behavior, and a 1-D batched
        tensordict (batch_size=[N], e.g. from ParallelEnv) as N independent
        decisions generated in one generate_batch call.
        """
        batched = tensordict.batch_dims > 0
        if batched:
            if tensordict.batch_dims != 1:
                raise ValueError(
                    f"LLM actor supports batch_dims 0 or 1, got batch_size="
                    f"{tuple(tensordict.batch_size)}. Flatten to a single batch dim first."
                )
            n = int(tensordict.batch_size[0])
            sub_tds = [tensordict[i] for i in range(n)]
        else:
            sub_tds = [tensordict]

        system_prompt = self._resolve_system_prompt()
        user_prompts = [self._build_user_prompt(st) for st in sub_tds]
        self._debug("SYSTEM PROMPT", system_prompt)
        self._debug("USER PROMPTS", "\n---\n".join(user_prompts))

        responses = self.generate_batch(system_prompt, user_prompts)
        self._debug("RESPONSES", "\n---\n".join(responses))

        responses = self._resolve_tools(system_prompt, user_prompts, responses)

        actions = [extract_action(r, num_actions=len(self.action_levels)) for r in responses]

        if batched:
            tensordict.set("action", torch.tensor(actions, dtype=torch.long))
            tensordict.set("thinking", list(responses))
            tensordict.set("system_prompt", [system_prompt] * n)
            tensordict.set("user_prompt", list(user_prompts))
        else:
            tensordict.set("action", torch.tensor(actions[0], dtype=torch.long))
            tensordict.set("thinking", responses[0])
            tensordict.set("system_prompt", system_prompt)
            tensordict.set("user_prompt", user_prompts[0])

        return tensordict

    def _resolve_tools(
        self, system_prompt: str, user_prompts: List[str], responses: List[str]
    ) -> List[str]:
        """Run the multi-turn tool loop when tools are configured.

        Default (no tools): returns responses unchanged. Otherwise, each round
        finds responses containing a <tool> call, executes the tools, injects a
        <tool_results> block, and RE-GENERATES only those conversations in one
        batched call — preserving the single-batched-call throughput for the
        conversations that answered directly.
        """
        if not self.tools:
            return responses
        convo = list(user_prompts)
        for _ in range(self.max_tool_iters):
            pending = []
            for i, resp in enumerate(responses):
                _, calls = parse_tool_calls(resp)
                if not calls:
                    continue
                results = self._run_tool_calls(calls)
                convo[i] = self._linearize(convo[i], resp, results)
                pending.append(i)
            if not pending:
                break
            regen = self.generate_batch(system_prompt, [convo[i] for i in pending])
            for j, i in enumerate(pending):
                responses[i] = regen[j]
        return responses

    def _build_tools_prompt(self) -> str:
        tool_list = "\n".join(f"  - {t.description}" for t in self.tools)
        return (
            "You may call tools before deciding. Available tools:\n"
            f"{tool_list}\n\n"
            'To call a tool, output exactly: <tool name="<tool>">{"arg": "value"}</tool> '
            "and stop. You will receive a <tool_results>...</tool_results> block, then continue.\n"
            f"You may call tools up to {self.max_tool_iters} times. Finish with <answer>N</answer>."
        )

    def _run_tool_calls(self, calls: List[dict]) -> str:
        lines = ["<tool_results>"]
        for idx, call in enumerate(calls, 1):
            name = call["name"]
            tool = self._tools_by_name.get(name)
            if tool is None:
                lines.append(f"Tool {name} (call {idx}) failed:")
                lines.append(f"  Error: unknown tool '{name}'")
                continue
            try:
                result = tool.run(**call["args"])
                lines.append(f"Tool {name} (call {idx}) succeeded:")
                lines.append(f"  Result: {result}")
            except Exception as exc:  # per-tool guard; never crash a live step
                lines.append(f"Tool {name} (call {idx}) failed:")
                lines.append(f"  Error: {exc}")
        lines.append("</tool_results>")
        return "\n".join(lines)

    def _linearize(self, base_prompt: str, response: str, results: str) -> str:
        return (
            f"{base_prompt}\n\n{response}\n{results}\n\n"
            "Continue your analysis. When ready, respond with <answer>N</answer>."
        )

    def _debug(self, label: str, content: str) -> None:
        if self.debug:
            print(f"{'=' * 80}\n{label}:\n{content}")

    def _resolve_system_prompt(self) -> str:
        override = self._system_prompt_override
        if override is None:
            base = self._build_system_prompt()
        elif callable(override):
            base = override(self)
        else:
            base = override
        if self.tools:
            base = base + "\n\n" + self._build_tools_prompt()
        return base

    # --- Prompt construction ---

    def _construct_user_prompt(self, tensordict) -> str:
        return self._construct_account_state(tensordict) + self._construct_market_data(tensordict)

    def _construct_account_state(self, tensordict) -> str:
        # Envs that don't expose account_state (e.g. PolymarketBetEnv) just
        # omit the key — skip this block.
        if "account_state" not in tensordict:
            return ""

        account_state = tensordict.get("account_state")
        if account_state.dim() == 2 and account_state.shape[0] == 1:
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
            if data.ndim == 3 and data.shape[0] == 1:
                data = data.squeeze(0)

            # Flat 1D market state (e.g. PolymarketBetEnv's
            # [yes_price, spread, vol_24h, liquidity]) — render as one labeled row.
            if data.ndim == 1:
                if len(self.feature_keys) != data.shape[0]:
                    raise ValueError(
                        f"Unexpected market data shape for {key}: {data.shape} "
                        f"(expected 1D with {len(self.feature_keys)} feature values)"
                    )
                out += f"{key}:\n"
                for label, value in zip(self.feature_keys, data, strict=True):
                    out += f"  {label}: {value:.4f}\n"
                out += "\n"
                continue

            if data.ndim != 2 or data.shape[1] != len(self.feature_keys):
                raise ValueError(
                    f"Unexpected market data shape for {key}: {data.shape} "
                    f"(expected 2D with {len(self.feature_keys)} feature columns)"
                )

            out += f"{key}:\n\n"
            header = " | ".join(f"{k:>8}" for k in self.feature_keys)
            out += header + "\n\n"
            for t in range(data.shape[0]):
                row = " | ".join(f"{v:8.1f}" for v in data[t])
                out += row + "\n"
            out += "\n"

        return out
