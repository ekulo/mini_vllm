"""Unit tests for the HTTP layer's helpers (no server needed).

Importing ``mini_vllm.api_server`` pulls in FastAPI; skip cleanly if it is not
installed rather than failing the whole suite.
"""

from __future__ import annotations

import pytest

from mini_vllm.tokenizer import Tokenizer
from tests.test_engine import _engine  # noqa: F401  (reused fixture helper)

api = pytest.importorskip("mini_vllm.api_server", reason="FastAPI not installed")
TextAccumulator = api.TextAccumulator
_render_chat = api._render_chat
ChatMessage = api.ChatMessage
_usage = api._usage


@pytest.fixture(scope="module")
def tok():
    return _engine().tokenizer


def test_accumulator_counts_every_token_not_just_visible_words(tok):
    """usage.completion_tokens must include special tokens, which render as ''."""
    acc = TextAccumulator(tok, None)
    acc.push(tok.word2id["word_5"])
    acc.push(tok.unk_id)          # renders as "" -- the bug this guards
    acc.push(tok.word2id["word_9"])
    acc.push(tok.eos_id)          # renders as ""

    assert acc.num_tokens == 4
    assert acc.words == ["word_5", "word_9"]
    assert acc.text == "word_5 word_9"


def test_accumulator_joins_with_single_spaces(tok):
    acc = TextAccumulator(tok, None)
    acc.push(tok.word2id["word_1"])
    acc.push(tok.word2id["word_2"])
    acc.push(tok.word2id["word_3"])
    assert acc.text == "word_1 word_2 word_3"


def _push_text(acc, delta: str) -> str:
    """Mimic ``TextAccumulator.push`` for a delta of our choosing.

    The vocab has no word that contains "STOP", so stop-string behaviour cannot
    be exercised through ``push(token_id)`` directly. This reproduces exactly
    what push does -- append to ``words``/``text``, then let the caller trim.
    """
    acc.words.append(delta)
    acc.text += delta
    return acc.trim_delta_for_stop(delta)


def test_stop_string_inside_a_delta_trims_only_the_tail(tok):
    acc = TextAccumulator(tok, ["STOP"])
    acc.text = "alpha "
    acc.words = ["alpha"]

    emitted = _push_text(acc, "beta STOP gamma")

    assert acc.stop_index() == len("alpha beta ")
    assert emitted == "beta "                       # "gamma" never reaches the client
    assert acc.text[: acc.stop_index()] == "alpha beta "


def test_stop_string_at_the_start_of_a_delta_emits_nothing(tok):
    acc = TextAccumulator(tok, ["STOP"])
    acc.text = "alpha beta "
    acc.words = ["alpha", "beta"]

    assert _push_text(acc, "STOP gamma") == ""


def test_no_stop_string_leaves_the_delta_alone(tok):
    acc = TextAccumulator(tok, ["NOPE"])
    acc.text = "alpha "
    assert acc.stop_index() is None
    assert _push_text(acc, "beta gamma") == "beta gamma"


def test_empty_stop_list_never_triggers(tok):
    acc = TextAccumulator(tok, [])
    acc.text = ""
    assert _push_text(acc, "STOP") == "STOP"
    assert acc.stop_index() is None


def test_usage_totals_add_up():
    assert _usage(3, 5) == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_chat_template_ends_with_assistant_turn():
    prompt = _render_chat([ChatMessage(role="system", content="s"),
                           ChatMessage(role="user", content="u")])
    assert prompt == "system: s\nuser: u\nassistant:"
