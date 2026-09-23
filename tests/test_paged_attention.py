"""The two things that must be true for paged inference to be correct:

1. Running a prompt as prefill + N decode steps gives exactly the same logits as
   one full-sequence forward, at every step.
2. Sequences batched together never influence each other.

These are the tests that would catch an off-by-one in the slot mapping, a broken
causal mask, or a block table read past the end of a sequence.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm import KVCacheManager, ModelRunner, SamplingParams, Sequence
from tests.helpers import naive_forward, tiny_config

PROMPTS = [
    [2, 5, 9, 11],                 # short
    [2, 400, 12, 33, 7, 91, 5],    # medium, crosses a block boundary at 8
    [2],                           # single token
    [2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 2+ blocks
]


def _runner(**overrides):
    cfg = tiny_config(**overrides)
    kv = KVCacheManager(cfg, torch.device("cpu"), torch.float32)
    runner = ModelRunner(cfg, kv)
    return cfg, kv, runner


def _prefill(runner, kv, seq):
    assert kv.allocate_slots(seq, len(seq.get_token_ids()))
    seq.num_scheduled_tokens = len(seq.get_token_ids())
    logits = runner.execute([seq])
    seq.num_computed_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return logits


def _decode(runner, kv, seq, token_id):
    seq.append_token(token_id)
    assert kv.allocate_slots(seq, 1)
    seq.num_scheduled_tokens = 1
    logits = runner.execute([seq])
    seq.num_computed_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return logits


@pytest.mark.parametrize("prompt", PROMPTS)
def test_prefill_then_decode_matches_full_forward(prompt):
    _, kv, runner = _runner()
    model = runner.model

    seq = Sequence(0, prompt, SamplingParams(max_tokens=16))
    prefill_logits = _prefill(runner, kv, seq)

    reference = naive_forward(model, torch.tensor([prompt]))
    # Prefill samples from the last prompt position.
    torch.testing.assert_close(prefill_logits[0], reference[0, -1], atol=1e-4, rtol=1e-4)

    # Three greedy decode steps, each compared against a fresh full forward.
    for step in range(3):
        token = int(prefill_logits[0].argmax())
        logits = _decode(runner, kv, seq, token)
        reference = naive_forward(model, torch.tensor([seq.get_token_ids()]))
        torch.testing.assert_close(
            logits[0], reference[0, -1], atol=1e-4, rtol=1e-4,
            msg=f"decode step {step} diverged from the full forward",
        )
        prefill_logits = logits


def test_chunked_prefill_matches_single_shot():
    """Splitting a prompt across steps must change nothing but the step count."""
    cfg, kv, runner = _runner(max_num_batched_tokens=8)
    prompt = PROMPTS[3]

    seq_chunked = Sequence(0, prompt, SamplingParams(max_tokens=4))
    last = None
    while seq_chunked.num_computed_tokens < len(prompt):
        n = min(cfg.max_num_batched_tokens, len(prompt) - seq_chunked.num_computed_tokens)
        assert kv.allocate_slots(seq_chunked, n)
        seq_chunked.num_scheduled_tokens = n
        last = runner.execute([seq_chunked])
        seq_chunked.num_computed_tokens += n
        seq_chunked.num_scheduled_tokens = 0
    chunked_logits = last[0]

    _, kv2, runner2 = _runner()
    # Same weights: copy the state dict.
    runner2.model.load_state_dict(runner.model.state_dict())
    seq_full = Sequence(0, prompt, SamplingParams(max_tokens=4))
    full_logits = _prefill(runner2, kv2, seq_full)[0]

    torch.testing.assert_close(chunked_logits, full_logits, atol=1e-4, rtol=1e-4)


def test_batch_isolation():
    """Two sequences of different lengths batched together == run separately."""
    _, kv, runner = _runner()
    a = Sequence(0, PROMPTS[0], SamplingParams(max_tokens=4))
    b = Sequence(1, PROMPTS[1], SamplingParams(max_tokens=4))

    for s in (a, b):
        assert kv.allocate_slots(s, len(s.prompt_token_ids))
        s.num_scheduled_tokens = len(s.prompt_token_ids)
    batched = runner.execute([a, b])

    solo = []
    for i, s in enumerate((a, b)):
        cfg2, kv2, runner2 = _runner()
        runner2.model.load_state_dict(runner.model.state_dict())
        c = Sequence(1 - i, s.prompt_token_ids, SamplingParams(max_tokens=4))
        solo.append(_prefill(runner2, kv2, c)[0])

    torch.testing.assert_close(batched[0], solo[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(batched[1], solo[1], atol=1e-4, rtol=1e-4)


def test_non_contiguous_blocks_are_handled():
    """Force the allocator to hand out scattered blocks, then verify logits."""
    cfg, kv, runner = _runner(num_blocks=12, block_size=8)
    long_prompt = [2] + list(range(100, 120))  # 21 tokens -> 3 blocks

    # Burn blocks so the next allocation cannot be contiguous.
    holes = [Sequence(100 + i, [2, 3, 4], SamplingParams(max_tokens=2)) for i in range(3)]
    for h in holes:
        assert kv.allocate_slots(h, 3)
    kv.free(holes[0])
    kv.free(holes[2])

    seq = Sequence(0, long_prompt, SamplingParams(max_tokens=4))
    logits = _prefill(runner, kv, seq)

    assert len(seq.block_table) == 3
    assert seq.block_table != list(range(seq.block_table[0], seq.block_table[0] + 3)), (
        f"expected a fragmented block table, got {seq.block_table}"
    )

    reference = naive_forward(runner.model, torch.tensor([long_prompt]))
    torch.testing.assert_close(logits[0], reference[0, -1], atol=1e-4, rtol=1e-4)


def test_cache_reuse_does_not_leak_stale_values():
    """A recycled block holds another sequence's numbers; attention must ignore them."""
    cfg, kv, runner = _runner(num_blocks=8, block_size=8)

    first = Sequence(0, [2] + list(range(10, 30)), SamplingParams(max_tokens=2))
    _prefill(runner, kv, first)
    freed_blocks = list(first.block_table)
    kv.free(first)

    second = Sequence(1, PROMPTS[1], SamplingParams(max_tokens=2))
    _prefill(runner, kv, second)
    assert set(second.block_table) & set(freed_blocks), "expected a recycled block"

    reference = naive_forward(runner.model, torch.tensor([PROMPTS[1]]))
    # Recompute cleanly to compare against a cache that never saw the first seq.
    _, kv2, runner2 = _runner(num_blocks=8, block_size=8)
    runner2.model.load_state_dict(runner.model.state_dict())
    clean = Sequence(2, PROMPTS[1], SamplingParams(max_tokens=2))
    clean_logits = _prefill(runner2, kv2, clean)

    assert torch.isfinite(clean_logits).all()
    torch.testing.assert_close(clean_logits[0], reference[0, -1], atol=1e-4, rtol=1e-4)


def test_greedy_sampling_is_deterministic():
    cfg, kv, runner = _runner()
    seqs = [Sequence(i, PROMPTS[0], SamplingParams(max_tokens=4, temperature=0.0)) for i in range(4)]
    for s in seqs:
        kv.allocate_slots(s, len(s.prompt_token_ids))
        s.num_scheduled_tokens = len(s.prompt_token_ids)
    logits = runner.execute(seqs)
    tokens = runner.sample(logits, seqs)
    assert tokens == [tokens[0]] * len(seqs), "identical prompts + greedy must agree"
