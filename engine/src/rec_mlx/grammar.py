import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import xgrammar as xgr

from . import _grammar_native as native
from .grammar_builder import build_flint_glm_grammar

VOCAB = 150144
WORDS = VOCAB // 32


@dataclass(frozen=True)
class Constraint:
    compiled: object
    stops: tuple[int, ...]
    # Reasoning delimiters; a reasoning budget needs both.
    think: int | None = None
    think_end: int | None = None

    def session(self, reasoning_budget=None):
        return ConstraintSession(self, reasoning_budget)


class ConstraintCompiler:
    def __init__(self, tokenizer, *, enforce_tool_schemas=False):
        self.tokenizer = tokenizer
        self.enforce_tool_schemas = enforce_tool_schemas
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=VOCAB, stop_token_ids=[tokenizer.eos_token_id]
        )
        self.compiler = xgr.GrammarCompiler(info, cache_limit_bytes=128 * 1024**2)

    def tools(self, tools=None):
        tag = build_flint_glm_grammar(
            self.tokenizer, tools, enforce_tool_schemas=self.enforce_tool_schemas
        )
        compiled = self.compiler.compile_structural_tag(tag)
        stops = tuple(
            self.tokenizer.convert_tokens_to_ids(x)
            for x in ("<|end_of_text|>", "<|user|>", "<|observation|>")
        )
        think, think_end = self.tokenizer.convert_tokens_to_ids(["<think>", "</think>"])
        return Constraint(compiled, stops, think, think_end)


class ConstraintSession:
    def __init__(self, constraint, reasoning_budget=None):
        self.constraint = constraint
        self.stops = constraint.stops
        # Reasoning budget: once the open <think> block holds this many tokens, the
        # only allowed token is </think>, so the model moves on to its answer. A
        # budget of 0 yields <think></think>, the GLM no-thinking form.
        if reasoning_budget is not None and (
            constraint.think is None or constraint.think_end is None
        ):
            raise ValueError("A reasoning budget needs a grammar with reasoning delimiters")
        self.reasoning_budget = reasoning_budget
        self.reasoning = None
        self.reasoning_tokens = 0
        if reasoning_budget is not None:
            self.think_end_mask = np.zeros(WORDS, np.uint32)
            self.think_end_mask[constraint.think_end >> 5] = 1 << (constraint.think_end & 31)
        self.matcher = xgr.GrammarMatcher(constraint.compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, VOCAB)
        self.numpy_mask = self.bitmask.numpy().view(np.uint32).reshape(-1)
        self.full_mask = np.full(WORDS, 0xFFFFFFFF, np.uint32)
        self.stopped = False
        self.mask_seconds = []
        self.accepted = 0
        first = self._fill()
        self.initial_mask = mx.array(np.concatenate([first, np.array([1], np.uint32)]))
        self.identifier = native.create(self._advance)
        self.handle = mx.array(self.identifier, mx.uint64)
        mx.eval(self.initial_mask, self.handle)

    def _fill(self):
        start = time.perf_counter()
        self.matcher.fill_next_token_bitmask(self.bitmask)
        self.mask_seconds.append(time.perf_counter() - start)
        return self.numpy_mask

    def _advance(self, token, active):
        if not active or self.stopped:
            return self.full_mask
        if not self.matcher.accept_token(token):
            raise RuntimeError(f"Grammar rejected sampled token {token}")
        self.accepted += 1
        if token in self.stops:
            self.stopped = True
            return self.full_mask
        if token == self.constraint.think:
            self.reasoning = 0
        elif token == self.constraint.think_end:
            self.reasoning = None
        elif self.reasoning is not None:
            self.reasoning += 1
            self.reasoning_tokens += 1
        mask = self._fill()
        if self.reasoning is not None and self.reasoning_budget is not None:
            if self.reasoning >= self.reasoning_budget:
                return self.think_end_mask
        return mask

    def first_logits(self, logits):
        words = mx.arange(VOCAB, dtype=mx.uint32)
        bits = self.initial_mask
        allowed = ((bits[words >> 5] >> (words & 31)) & 1) != 0
        return mx.where(allowed, logits, -float("inf"))

    def accept_first(self, token, state):
        mask = self._advance(token, True)
        state["grammar_mask"] = mx.array(np.concatenate([mask, np.array([1], np.uint32)]))

    def drain(self):
        native.drain(self.identifier)

    def metrics(self):
        return {
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_budget": self.reasoning_budget,
            "grammar_accepted_tokens": self.accepted,
            "grammar_mask_seconds": sum(self.mask_seconds),
            "grammar_max_mask_seconds": max(self.mask_seconds),
        }

    def __enter__(self):
        return self

    def __exit__(self, *error):
        native.close(self.identifier)
