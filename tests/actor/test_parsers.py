"""Tests for the standalone LLM action-extraction parser."""
import pytest

from torchtrade.actor.parsers import extract_action


@pytest.mark.parametrize("response,expected", [
    ("<think>go long</think><answer>2</answer>", 2),   # normal
    ("<answer>0</answer>", 0),                          # boundary low
    ("<ANSWER> 1 </ANSWER>", 1),                        # case-insensitive + whitespace
    ("<answer>99</answer>", 0),                         # out of range -> 0
    ("no tags here", 0),                                # missing tag -> 0
    ("<answer>-1</answer>", 0),                         # negative not matched -> 0 (no accidental short)
    ("<answer>1</answer> <answer>2</answer>", 1),       # first match wins
], ids=["normal", "low", "case-ws", "out-of-range", "no-tag", "negative", "first-wins"])
def test_extract_action(response, expected):
    assert extract_action(response, num_actions=3) == expected


@pytest.mark.parametrize("response,fragment", [
    ("<answer>99</answer>", "out of range"),
    ("no tag", "No <answer> tag"),
], ids=["out-of-range", "no-tag"])
def test_extract_action_warns(caplog, response, fragment):
    """The silent fallback-to-0 must be observable via a WARNING."""
    with caplog.at_level("WARNING", logger="torchtrade.actor.parsers"):
        extract_action(response, num_actions=3)
    assert any(fragment in r.message for r in caplog.records)
