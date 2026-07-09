"""Tests for the standalone LLM action-extraction parser."""
import pytest

from torchtrade.actor.parsers import extract_action, parse_tool_calls


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


@pytest.mark.parametrize("response,expected_calls,expected_text", [
    # single call, JSON body
    ('<tool name="google_news">{"query": "Bitcoin"}</tool>\nSome text.',
     [{"name": "google_news", "args": {"query": "Bitcoin"}, "tag": None}], "Some text."),
    # tag attribute
    ('<tool name="search" tag="A">{"q": 1}</tool>',
     [{"name": "search", "args": {"q": 1}, "tag": "A"}], ""),
    # empty body -> {}
    ('<tool name="google_news"></tool>',
     [{"name": "google_news", "args": {}, "tag": None}], ""),
    # malformed JSON -> {"raw": body}
    ('<tool name="x">not json</tool>',
     [{"name": "x", "args": {"raw": "not json"}, "tag": None}], ""),
    # multiple calls
    ('<tool name="a">{}</tool> mid <tool name="b">{}</tool>',
     [{"name": "a", "args": {}, "tag": None}, {"name": "b", "args": {}, "tag": None}], "mid"),
    # no tag
    ("just an answer <answer>1</answer>", [], "just an answer <answer>1</answer>"),
], ids=["single", "tag", "empty-body", "malformed", "multiple", "none"])
def test_parse_tool_calls(response, expected_calls, expected_text):
    text, calls = parse_tool_calls(response)
    assert calls == expected_calls
    assert text == expected_text
