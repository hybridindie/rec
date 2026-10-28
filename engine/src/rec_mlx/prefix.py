import copy
import logging
from dataclasses import dataclass, field

import mlx.core as mx

from .cache import FixedKVCache, FixedSlidingKVCache, KVStorage, copy_array


class PrefillCancelled(Exception):
    pass


@dataclass
class Checkpoint:
    position: int
    sliding: list
    logits: mx.array


@dataclass
class PromptCache:
    tokens: list[int]
    layers: list
    logits: mx.array | None
    storage: object
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def fork(self):
        layers = [copy.copy(layer) for layer in self.layers]
        for layer in layers:
            if isinstance(layer, FixedSlidingKVCache):
                keys = copy_array(layer.key_buffers[layer._slot])
                values = copy_array(layer.value_buffers[layer._slot])
                layer.key_buffers = (keys, mx.zeros_like(keys))
                layer.value_buffers = (values, mx.zeros_like(values))
                layer._slot = 0
        return PromptCache(self.tokens[:], layers, self.logits, self.storage, self.checkpoints[:])

    def advance(self, tokens, logits):
        previous = len(self.tokens)
        self.tokens.extend(tokens)
        self.logits = logits
        if len(self.tokens) // 8192 != previous // 8192:
            logging.info("Prefill progress: %d tokens processed", len(self.tokens))

    def checkpoint(self):
        if not self.tokens:
            return
        self.checkpoints = self.checkpoints[-1:]
        sliding = []
        for layer in self.layers:
            if isinstance(layer, FixedSlidingKVCache):
                sliding.append(
                    (
                        copy.deepcopy(layer.key_buffers[layer._slot]),
                        copy.deepcopy(layer.value_buffers[layer._slot]),
                        layer._valid,
                        layer._ring_index,
                    )
                )
        mx.eval(sliding)
        self.checkpoints.append(Checkpoint(len(self.tokens), sliding, self.logits))

    @property
    def checkpoint_bytes(self):
        total = 0
        for checkpoint in self.checkpoints:
            total += sum(keys.nbytes + values.nbytes for keys, values, _, _ in checkpoint.sliding)
            total += checkpoint.logits.nbytes
        return total


# Never stage decode KV: its arithmetic differs from batched prefill and changes probabilities.
@dataclass
class PrefixPreparation:
    origin: PromptCache
    tokens: list[int]
    cache: PromptCache | None = None


def select_prefix(prefix, tokens):
    common = 0
    if prefix is not None:
        for previous, current in zip(prefix.tokens, tokens):
            if previous != current:
                break
            common += 1
        if common != len(prefix.tokens):
            checkpoint = next(
                (item for item in reversed(prefix.checkpoints) if item.position <= common), None
            )
            if checkpoint is None:
                prefix = None
            else:
                sliding = iter(checkpoint.sliding)
                for layer in prefix.layers:
                    layer.offset = checkpoint.position
                    if isinstance(layer, FixedSlidingKVCache):
                        keys, values, layer._valid, layer._ring_index = next(sliding)
                        layer.capacity = keys.shape[-2]
                        layer.key_buffers = (keys, mx.zeros_like(keys))
                        layer.value_buffers = (values, mx.zeros_like(values))
                        layer._slot = 0
                prefix.tokens = prefix.tokens[: checkpoint.position]
                prefix.logits = checkpoint.logits
                prefix.checkpoints = [
                    item for item in prefix.checkpoints if item.position < checkpoint.position
                ]
    cached = 0 if prefix is None else len(prefix.tokens)
    logging.info(
        "Prefill: prompt=%d common_prefix=%d reused=%d new=%d",
        len(tokens),
        common,
        cached,
        len(tokens) - cached,
    )
    return prefix, common


def usable_prefix(prefix, tokens):
    """How many cached tokens select_prefix would reuse for tokens, without changing prefix."""
    if prefix is None:
        return 0
    common = 0
    for previous, current in zip(prefix.tokens, tokens):
        if previous != current:
            break
        common += 1
    if common == len(prefix.tokens):
        return common
    return next(
        (item.position for item in reversed(prefix.checkpoints) if item.position <= common), 0
    )


def empty_storage_like(storage, capacity):
    """New storage of the same kind as storage. M4 Max storage also carries its block size,
    which its prefill reads; a plain KVStorage there fails on the first restored slot."""
    block_tokens = getattr(storage, "block_tokens", None)
    if block_tokens is None:
        return KVStorage(storage.rows, capacity)
    return type(storage)(storage.rows, capacity, block_tokens=block_tokens)


def clone_prefix(prefix, position):
    """The first position tokens of prefix in new storage, leaving prefix untouched.

    position must be len(prefix.tokens) or one of its checkpoints. The clone has no
    checkpoints of its own: sharing checkpoint arrays would let one cache's prefill write
    into the other's sliding buffers.
    """
    if position == len(prefix.tokens):
        sliding = [
            (
                layer.key_buffers[layer._slot],
                layer.value_buffers[layer._slot],
                layer._valid,
                layer._ring_index,
            )
            for layer in prefix.layers
            if isinstance(layer, FixedSlidingKVCache)
        ]
        logits = prefix.logits
    else:
        checkpoint = next(item for item in prefix.checkpoints if item.position == position)
        sliding, logits = checkpoint.sliding, checkpoint.logits
    storage = empty_storage_like(prefix.storage, position)
    rings = iter(sliding)
    layers = []
    for layer in prefix.layers:
        if isinstance(layer, FixedKVCache):
            data = prefix.storage.layer(layer.index, position)
            storage.write(layer.index, 0, data[0:1], data[1:2])
            clone = FixedKVCache(storage, layer.index)
        else:
            keys, values, valid, ring = next(rings)
            clone = copy.copy(layer)
            keys, values = copy_array(keys), copy_array(values)
            clone.key_buffers = (keys, mx.zeros_like(keys))
            clone.value_buffers = (values, mx.zeros_like(values))
            clone.capacity = keys.shape[-2]
            clone._slot, clone._valid, clone._ring_index = 0, valid, ring
        clone.offset = position
        layers.append(clone)
    mx.eval([layer.state for layer in layers])
    return PromptCache(prefix.tokens[:position], layers, logits, storage)
