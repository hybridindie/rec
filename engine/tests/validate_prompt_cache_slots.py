"""Compare prompt-cache slot reuse with cold prefill on the Flint checkpoint."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from rec_mlx.engine import CLONE_MIN_DISCARD, Engine, Sampling


def equal(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


def snapshot(cache):
    sliding = [
        tuple(mx.array(x) for x in layer._prior())
        for layer in cache.layers
        if hasattr(layer, "_prior")
    ]
    return mx.array(cache.logits), sliding


def same_state(cache, expected):
    logits, sliding = expected
    equal(cache.logits, logits)
    actual = [layer._prior() for layer in cache.layers if hasattr(layer, "_prior")]
    for layer, reference in zip(actual, sliding, strict=True):
        for a, b in zip(layer, reference, strict=True):
            equal(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="checkpoint manifest with the expected file hashes",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the results")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=24000,
        help="length of the first conversation (default: %(default)s)",
    )
    parser.add_argument(
        "--shared-tokens",
        type=int,
        default=9000,
        help="tokens the second conversation shares with the first (default: %(default)s)",
    )
    args = parser.parse_args()
    assert args.prompt_tokens - args.shared_tokens >= CLONE_MIN_DISCARD + 2048
    engine = Engine(args.model, args.manifest, context_limit=262144, prompt_cache_slots=2)
    notes = "The warehouse has seven blue crates. " * (args.prompt_tokens // 6)
    first = engine.prompt([{"role": "user", "content": "Remember these notes. " + notes}])[
        : args.prompt_tokens
    ]
    assert len(first) == args.prompt_tokens
    detour = engine.tokenizer.encode(
        " Ignore that. Write a haiku about lighthouses." * 40, add_special_tokens=False
    )
    second = first[: args.shared_tokens] + detour
    settings = Sampling(temperature=0, top_p=1, top_k=0, max_tokens=16, seed=42)

    def generate(tokens):
        return list(engine.generate_tokens(tokens, settings))[-1]["metrics"]

    def reset(slots):
        engine._prompt_cache = None
        engine._idle_caches = []
        engine.prompt_cache_slots = slots

    # 1. First conversation, then a second one sharing only its head: the head is copied.
    generate(first)
    original = engine._prompt_cache
    original_tokens = list(original.tokens)
    branched = generate(second)
    assert branched["prompt_cache_source"] == "clone", branched
    boundary = branched["cached_prompt_tokens"]
    assert 0 < boundary <= args.shared_tokens
    assert engine._idle_caches == [original] and original.tokens == original_tokens
    assert engine._prompt_cache.storage is not original.storage
    branched_state = snapshot(engine._prompt_cache)

    # 2. Back to the first conversation: it resumes from the parked cache untouched.
    follow_up = original_tokens + engine.tokenizer.encode(
        " And how many crates are there?", add_special_tokens=False
    )
    resumed = generate(follow_up)
    assert resumed["prompt_cache_source"] == "idle", resumed
    assert resumed["cached_prompt_tokens"] == len(original_tokens)
    resumed_state = snapshot(engine._prompt_cache)
    del original

    # References: cold prefill with the same chunk boundaries, one slot, nothing shared.
    reset(1)
    generate(second[:boundary])
    branch_reference = generate(second)
    same_state(engine._prompt_cache, branched_state)
    assert branched["token_ids"] == branch_reference["token_ids"]

    reset(1)
    generate(first)
    assert engine._prompt_cache.tokens == original_tokens
    resume_reference = generate(follow_up)
    same_state(engine._prompt_cache, resumed_state)
    assert resumed["token_ids"] == resume_reference["token_ids"]

    report = {
        "first_prompt_tokens": len(first),
        "second_prompt_tokens": len(second),
        "clone_boundary_tokens": boundary,
        "clone_prefilled_tokens": branched["prefilled_tokens"],
        "clone_prefill_seconds": branched["prefill_seconds"],
        "cold_prefill_seconds": branch_reference["prefill_seconds"],
        "resumed_cached_tokens": resumed["cached_prompt_tokens"],
        "resumed_prefilled_tokens": resumed["prefilled_tokens"],
        "resumed_prefill_seconds": resumed["prefill_seconds"],
        "idle_prompt_cache_bytes": resumed["idle_prompt_cache_bytes"],
        "clone_bitwise_logits_and_sliding": True,
        "clone_greedy_tokens_equal": True,
        "resume_bitwise_logits_and_sliding": True,
        "resume_greedy_tokens_equal": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
