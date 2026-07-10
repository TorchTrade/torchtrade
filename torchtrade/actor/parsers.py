"""Standalone parsers for LLM trading-actor responses."""
import json
import logging
import re

logger = logging.getLogger(__name__)

_ANSWER_PATTERN = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_action(response: str, num_actions: int) -> int:
    """Parse the chosen action index from an LLM response's <answer>N</answer> tag.

    Returns N when a tag is found and 0 <= N < num_actions. Falls back to 0
    (logging a warning) when no tag is present or N is out of range — a trading
    agent must always emit a valid action.
    """
    match = _ANSWER_PATTERN.search(response)
    if match:
        idx = int(match.group(1))
        if 0 <= idx < num_actions:
            return idx
        logger.warning("Action %d out of range [0, %d); defaulting to 0", idx, num_actions)
        return 0

    logger.warning("No <answer> tag found in response; defaulting to action 0")
    return 0


_TOOL_PATTERN = re.compile(
    r'<tool\s+name="(?P<name>[^"]+)"(?:\s+tag="(?P<tag>[^"]+)")?\s*>\s*(?P<body>.*?)\s*</tool>',
    re.DOTALL,
)


def parse_tool_calls(response: str) -> tuple[str, list[dict]]:
    """Parse torchrl-style <tool name="..."[ tag="..."]>{json}</tool> blocks.

    Mirrors torchrl's XMLBlockParser: name/tag are attributes, the body is the
    args JSON (empty body -> {}, JSON error -> {"raw": body}). Returns the text
    with tool blocks removed, and the list of parsed calls (possibly empty).
    """
    calls: list[dict] = []

    def repl(m: re.Match) -> str:
        body = m.group("body")
        try:
            args = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            args = {"raw": body}
        calls.append({"name": m.group("name"), "args": args, "tag": m.group("tag")})
        return ""

    cleaned = _TOOL_PATTERN.sub(repl, response).strip()
    return cleaned, calls
