import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from rec_mlx.cache import FixedKVCache, FixedSlidingKVCache, KVStorage
from rec_mlx.engine import Engine
from rec_mlx.m4_max import KVStorage as M4MaxKVStorage
from rec_mlx.prefix import Checkpoint, PromptCache, clone_prefix, usable_prefix


def cache(tokens, checkpoints=()):
    return PromptCache(
        list(tokens),
        [],
        mx.array([len(tokens)]),
        object(),
        [Checkpoint(p, [], mx.array([p])) for p in checkpoints],
    )


class UsablePrefixTests(unittest.TestCase):
    def test_full_match_reuses_everything(self):
        self.assertEqual(usable_prefix(cache(range(10)), list(range(14))), 10)

    def test_divergence_rolls_back_to_the_last_checkpoint_inside_the_shared_head(self):
        prompt = list(range(5)) + [99] * 9
        self.assertEqual(usable_prefix(cache(range(10), [2, 4, 8]), prompt), 4)

    def test_no_shared_checkpoint_reuses_nothing(self):
        self.assertEqual(usable_prefix(cache(range(10), [8]), [0, 1, 99]), 0)
        self.assertEqual(usable_prefix(None, [0]), 0)


class SlotPolicyTests(unittest.TestCase):
    def engine(self, slots, current=None, idle=()):
        engine = Engine.__new__(Engine)
        engine.prompt_cache_slots = slots
        engine._prompt_cache = current
        engine._prefix_preparation = None
        engine._idle_caches = list(idle)
        return engine

    def clones(self):
        made = []

        def clone(prefix, position):
            made.append((prefix, position))
            return cache(prefix.tokens[:position])

        return made, patch("rec_mlx.prefix.clone_prefix", side_effect=clone)

    def test_one_slot_keeps_todays_behaviour(self):
        a = cache(range(10))
        engine = self.engine(1, a)
        self.assertIs(engine._take_prefix([50, 51]), a)
        self.assertEqual(engine._idle_caches, [])

    def test_a_new_conversation_parks_the_old_one(self):
        a = cache(range(10))
        engine = self.engine(2, a)
        self.assertIsNone(engine._take_prefix([50, 51]))
        self.assertEqual(engine._idle_caches, [a])
        self.assertEqual(engine.prompt_cache_source, "none")

    def test_returning_conversation_comes_back_from_idle(self):
        a, b = cache(range(10)), cache(range(50, 60))
        engine = self.engine(2, b, [a])
        self.assertIs(engine._take_prefix(list(range(12))), a)
        self.assertEqual(engine._idle_caches, [b])
        self.assertEqual(engine.prompt_cache_source, "idle")

    def test_oldest_conversation_is_evicted_beyond_the_slot_count(self):
        a, b, c = cache(range(10)), cache(range(20, 30)), cache(range(40, 50))
        engine = self.engine(3, c, [a, b])
        self.assertIsNone(engine._take_prefix([90]))
        self.assertEqual(engine._idle_caches, [b, c])

    def test_most_recent_wins_a_tie(self):
        a, b = cache(range(10)), cache(range(10))
        engine = self.engine(3, b, [a])
        self.assertIs(engine._take_prefix(list(range(11))), b)

    def test_short_rollback_of_the_same_conversation_happens_in_place(self):
        long = cache(range(20_000), [16_384])
        engine = self.engine(2, long)
        made, patcher = self.clones()
        with patcher:
            prompt = list(range(19_000)) + [-1]
            self.assertIs(engine._take_prefix(prompt), long)
        self.assertEqual(made, [])
        self.assertEqual(engine.prompt_cache_source, "current")

    def test_sharing_only_a_head_clones_it_and_keeps_the_conversation(self):
        long = cache(range(40_000), [4096, 8192])
        engine = self.engine(2, long)
        made, patcher = self.clones()
        with patcher:
            selected = engine._take_prefix(list(range(9000)) + [-1])
        self.assertEqual(made, [(long, 8192)])
        self.assertEqual(selected.tokens, list(range(8192)))
        self.assertEqual(engine._idle_caches, [long])
        self.assertEqual(long.tokens, list(range(40_000)))
        self.assertEqual(engine.prompt_cache_source, "clone")


def metadata(global_rows):
    rows = []
    for index in range(49):
        row = [0] * 63
        row[1] = int(index in global_rows)
        rows.append(row)
    return rows


@unittest.skipUnless(mx.metal.is_available(), "requires Apple Metal")
class ClonePrefixTests(unittest.TestCase):
    def test_clone_copies_the_head_into_independent_storage(self):
        self.check_clone(KVStorage(metadata({0}), 16))

    def test_clone_keeps_m4_max_storage_and_its_block_size(self):
        storage = M4MaxKVStorage(metadata({0}), 16, block_tokens=16)
        clone = self.check_clone(storage)
        self.assertIsInstance(clone.storage, M4MaxKVStorage)
        self.assertEqual(clone.storage.block_tokens, 16)

    def check_clone(self, storage):
        layers = [
            FixedKVCache(storage, 0),
            FixedSlidingKVCache(
                max_sequence_tokens=16, num_kv_heads=4, head_dim=128, window_size=8
            ),
        ]
        keys = mx.random.normal((1, 4, 10, 128)).astype(mx.bfloat16)
        values = mx.random.normal((1, 4, 10, 128)).astype(mx.bfloat16)
        layers[0].update_and_fetch(keys[..., :6, :], values[..., :6, :])
        layers[1].update_and_fetch(keys[..., :6, :], values[..., :6, :])
        prefix = PromptCache(list(range(6)), layers, mx.array([6]), storage)
        prefix.checkpoint()
        layers[0].update_and_fetch(keys[..., 6:, :], values[..., 6:, :])
        layers[1].update_and_fetch(keys[..., 6:, :], values[..., 6:, :])
        prefix.advance(list(range(6, 10)), mx.array([10]))

        clone = clone_prefix(prefix, 6)

        self.assertIsNot(clone.storage, storage)
        self.assertEqual(clone.tokens, list(range(6)))
        self.assertEqual(clone.checkpoints, [])
        self.assertEqual(clone.layers[0].offset, 6)
        head = clone.storage.layer(0, 6)
        self.assertTrue(mx.array_equal(head[0:1], keys[..., :6, :]).item())
        self.assertTrue(mx.array_equal(head[1:2], values[..., :6, :]).item())
        sliding = clone.layers[1]
        self.assertEqual((sliding.offset, sliding._valid), (6, 6))
        self.assertTrue(mx.array_equal(sliding._prior()[0], keys[..., :6, :]).item())
        # Writing to the original never reaches the clone.
        storage.write(0, 0, mx.zeros_like(keys[..., :6, :]), mx.zeros_like(values[..., :6, :]))
        self.assertTrue(mx.array_equal(clone.storage.layer(0, 6)[0:1], keys[..., :6, :]).item())
        self.assertEqual(prefix.tokens, list(range(10)))
        return clone


class SlotsMetricsTests(unittest.TestCase):
    def test_idle_bytes_count_storage_and_checkpoints(self):
        engine = Engine.__new__(Engine)
        a = cache(range(4))
        a.storage = SimpleNamespace(nbytes=100)
        engine._idle_caches = [a]
        self.assertEqual(engine.idle_prompt_cache_bytes, 100 + a.checkpoint_bytes)


if __name__ == "__main__":
    unittest.main()
