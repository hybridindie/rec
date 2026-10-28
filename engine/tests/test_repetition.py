import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from rec_mlx.engine import Engine, Sampling
from rec_mlx.prefix import PromptCache
from rec_mlx.repetition import RepetitionGuard


def first_trigger(guard, tokens):
    for index, token in enumerate(tokens):
        if guard.add(token):
            return index
    return None


class RepetitionGuardTests(unittest.TestCase):
    def test_short_loop_stops_after_the_minimum_run(self):
        loop = list(range(8)) * 200
        # Position n matches n-8 from n=8; the 512th match is token 519.
        self.assertEqual(first_trigger(RepetitionGuard(), loop), 519)

    def test_long_loop_must_repeat_twice_before_stopping(self):
        rng = random.Random(1)
        body = [rng.randrange(50_000) for _ in range(600)]
        guard = RepetitionGuard()
        self.assertEqual(first_trigger(guard, body * 4), 600 + 2 * 600 - 1)
        self.assertEqual(guard.period, 600)

    def test_varied_output_and_periods_past_the_limit_never_stop(self):
        rng = random.Random(7)
        self.assertIsNone(
            first_trigger(RepetitionGuard(), [rng.randrange(400) for _ in range(20_000)])
        )
        body = [rng.randrange(50_000) for _ in range(1100)]
        self.assertIsNone(first_trigger(RepetitionGuard(max_period=1024), body * 5))

    def test_a_break_in_the_loop_resets_the_run(self):
        tokens = list(range(8)) * 40 + [99] + list(range(8)) * 40
        self.assertIsNone(first_trigger(RepetitionGuard(), tokens))


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class DecodeLoopTests(unittest.TestCase):
    def setUp(self):
        text = SimpleNamespace(add_token=lambda token: "x", finish=lambda tokens: "")
        patcher = patch("rec_mlx.engine.StreamingText", lambda tokenizer: text)
        patcher.start()
        self.addCleanup(patcher.stop)

    def engine(self, *, repetition_guard):
        engine = Engine.__new__(Engine)
        engine.mx = mx
        engine.tokenizer = None
        engine.context_limit = 4096
        engine.eos = {0}
        engine.repetition_guard = repetition_guard
        engine._prompt_cache = None
        engine._prefix_preparation = None
        cache = PromptCache([], [], None, None)
        kv = SimpleNamespace(nbytes=0)

        class Session:
            cached_tokens = common_prefix_tokens = 0
            prompt_cache = cache
            _state = {"kv": kv, "capacity": 0, "growth_count": 0, "growth_seconds": 0.0}

            def prefill(self, ids, *, prefix, should_stop):
                return mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 9.0]])

        loop = list(range(1, 9))

        def block(state, previous, count, seed, **kwargs):
            start = loop.index(previous) + 1 if previous in loop else 0
            return [loop[(start + i) % len(loop)] for i in range(count)]

        engine.runtime = SimpleNamespace(
            make_session=lambda **kw: Session(),
            metadata=[],
            prefill_chunk_size=10**9,
            device_profile=SimpleNamespace(chip="M4 Pro"),
        )
        engine.resident_sampler = SimpleNamespace(block=block)
        return engine

    def run_to_end(self, engine):
        sampling = Sampling(temperature=0, max_tokens=2000)
        events = list(engine._generate_tokens([1, 2, 3], sampling, should_stop=lambda: False))
        return [e for e in events if "token" in e], events[-1]

    def test_guard_stops_a_looping_generation_as_length(self):
        tokens, final = self.run_to_end(self.engine(repetition_guard=True))
        self.assertLess(len(tokens), 2000)
        self.assertEqual(tokens[-1]["finish_reason"], "length")
        self.assertEqual(final["finish_reason"], "length")
        self.assertEqual(final["metrics"]["repetition_stopped"], {"period": 8})

    def test_guard_off_runs_to_max_tokens(self):
        tokens, final = self.run_to_end(self.engine(repetition_guard=False))
        self.assertEqual(len(tokens), 2000)
        self.assertNotIn("repetition_stopped", final["metrics"])


if __name__ == "__main__":
    unittest.main()
