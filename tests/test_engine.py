"""End-to-end engine tests: greedy determinism, stopping, batching, streaming."""

from __future__ import annotations

import asyncio

import pytest

from mini_vllm import EngineConfig, LLMEngine, SamplingParams
from mini_vllm.llm_engine import AsyncEngine


def _engine(**over):
    defaults = dict(
        device="cpu",
        dtype="float32",
        num_blocks=128,
        block_size=8,
        max_num_seqs=8,
        max_model_len=128,
        time_steps=False,
    )
    defaults.update(over)
    return LLMEngine(EngineConfig(**defaults))


def test_generate_is_deterministic_for_identical_prompts():
    eng = _engine()
    sp = SamplingParams(max_tokens=6, temperature=0.0)
    first = [eng.generate(p, sp) for p in ["word_1 word_2", "word_1 word_2", "word_3"]]
    second = [eng.generate(p, sp) for p in ["word_1 word_2", "word_1 word_2", "word_3"]]
    assert [r.token_ids for r in first] == [r.token_ids for r in second]
    assert first[0].token_ids == first[1].token_ids
    assert first[0].token_ids != first[2].token_ids


def test_max_tokens_is_respected():
    eng = _engine()
    r = eng.generate("word_5", SamplingParams(max_tokens=3, ignore_eos=True))
    assert len(r.token_ids) == 3
    assert r.finish_reason == "length"


def test_prompt_is_not_truncated_or_padded():
    eng = _engine()
    ids = eng.encode("word_1 word_2 word_3")
    # <BOS> + three words, and nothing else.
    assert ids[0] == eng.tokenizer.bos_id
    assert len(ids) == 4


def test_empty_prompt_still_runs():
    eng = _engine()
    r = eng.generate("", SamplingParams(max_tokens=2))
    assert len(r.token_ids) == 2


def test_batch_generation_matches_individual_generation():
    """Continuous batching must not change greedy results."""
    eng = _engine()
    prompts = ["word_10 word_20", "word_30", "word_40 word_50 word_60"]
    sp = SamplingParams(max_tokens=5, temperature=0.0)

    batched = eng.generate_batch(prompts, sp)
    solo = [eng.generate(p, sp) for p in prompts]
    assert [r.token_ids for r in batched] == [r.token_ids for r in solo]


def test_batch_generation_more_sequences_than_capacity_still_completes():
    eng = _engine(max_num_seqs=2)
    prompts = [f"word_{i}" for i in range(6)]
    results = eng.generate_batch(prompts, SamplingParams(max_tokens=4, temperature=0.0))
    assert len(results) == 6
    assert all(len(r.token_ids) == 4 for r in results)


def test_token_budget_forces_chunked_prefill_and_matches():
    """A tiny budget must give the same tokens as an unlimited one."""
    long_prompt = " ".join(f"word_{i}" for i in range(30))
    sp = SamplingParams(max_tokens=4, temperature=0.0)

    big = _engine(max_num_batched_tokens=256, chunked_prefill=True)
    small = _engine(max_num_batched_tokens=8, chunked_prefill=True)

    # Same weights: copy the state dict across engines.
    small.core.model_runner.model.load_state_dict(big.core.model_runner.model.state_dict())

    a = big.generate(long_prompt, sp)
    b = small.generate(long_prompt, sp)
    assert a.token_ids == b.token_ids


def test_kv_blocks_are_all_returned_after_generation():
    eng = _engine()
    eng.generate_batch([f"word_{i}" for i in range(5)], SamplingParams(max_tokens=4))
    stats = eng.core.kv_manager.stats()
    assert stats["blocks_used"] == 0
    assert stats["blocks_free"] == stats["blocks_total"]


def test_request_too_long_is_rejected():
    eng = _engine(max_model_len=16)
    with pytest.raises(ValueError):
        eng.generate(" ".join(f"word_{i}" for i in range(40)), SamplingParams(max_tokens=8))


def test_stats_separate_prefill_from_decode():
    eng = _engine()
    eng.generate("word_1 word_2 word_3 word_4", SamplingParams(max_tokens=5, ignore_eos=True))
    s = eng.stats()
    assert s["prefill_steps"] == 1
    assert s["decode_steps"] == 4
    assert s["prefill_tokens"] == 5      # <BOS> + 4 words
    assert s["decode_tokens"] == 4
    assert s["prefill_tokens_per_s"] > 0
    assert s["decode_tokens_per_s"] > 0


# --------------------------------------------------------------------------- #
#  async serving path
# --------------------------------------------------------------------------- #
def test_async_engine_streams_every_token():
    eng = _engine()
    ae = AsyncEngine(eng)

    async def main():
        ae.start(asyncio.get_running_loop())
        try:
            sp = SamplingParams(max_tokens=5, ignore_eos=True)
            outputs = [
                out async for out in ae.stream(eng.encode("word_7 word_8"), sp)
            ]
            return outputs
        finally:
            ae.stop()

    outputs = asyncio.run(main())
    assert len(outputs) == 5
    assert outputs[-1].finished
    assert [len(o.output_token_ids) for o in outputs] == [1, 2, 3, 4, 5]
    assert eng.core.kv_manager.stats()["blocks_used"] == 0


def test_async_engine_interleaves_two_requests():
    eng = _engine()
    ae = AsyncEngine(eng)

    async def consume(prompt, n):
        sp = SamplingParams(max_tokens=n, ignore_eos=True)
        return [out.token_id async for out in ae.stream(eng.encode(prompt), sp)]

    async def main():
        ae.start(asyncio.get_running_loop())
        try:
            return await asyncio.gather(consume("word_1 word_2", 4), consume("word_9", 4))
        finally:
            ae.stop()

    a, b = asyncio.run(main())
    assert len(a) == 4 and len(b) == 4
    # Greedy + identical weights: both streams are internally consistent.
    solo = eng.generate("word_1 word_2", SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.0))
    assert a == solo.token_ids


def test_async_engine_aborts_on_client_disconnect():
    eng = _engine()
    ae = AsyncEngine(eng)

    async def main():
        ae.start(asyncio.get_running_loop())
        try:
            sp = SamplingParams(max_tokens=100, ignore_eos=True)
            stream = ae.stream(eng.encode("word_1"), sp)
            first = await stream.__anext__()
            assert first.token_id is not None
            await stream.aclose()
            # Give the abort a moment to be picked up by the engine thread.
            await asyncio.sleep(0.3)
            return eng.core.stats()
        finally:
            ae.stop()

    stats = asyncio.run(main())
    assert stats["scheduler"]["running"] == 0
    assert stats["scheduler"]["waiting"] == 0
