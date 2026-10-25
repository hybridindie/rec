# hybridindie/rec — fork notes

A fork of [recursive-oss/rec](https://github.com/recursive-oss/rec) that tracks the fixes
and improvements we run in production for local Godot agent evals (godot-agents /
godot-mcp). The goal is to upstream what's generally useful; this file records what's in
the fork and where each change stands.

## Branches

| Branch | Contents |
|---|---|
| `main` | Mirror of `upstream/main`. Never commit here; sync with `git pull upstream main`. |
| `godot` | Our build: `main` + the upstream PRs and our patches listed below. |
| `feat/*` | One of our patches each, on `main`, ready for an upstream PR. |
| `upstream-pr/<n>` | An open upstream PR we depend on, as submitted. |

## What's in `godot`

| Change | Branch | Upstream status | Why we need it |
|---|---|---|---|
| Apple M4 Max support (76/64/64 worker profile) | `upstream-pr/17` | open: recursive-oss/rec#17 | Our machine is an M4 Max; stock rec refuses it |
| Keep 64 prefix checkpoints instead of 2 | `upstream-pr/24` | open: recursive-oss/rec#24 | Agent harnesses send long shared preambles; 2 checkpoints reuse none of them |
| `reasoning_effort` / `reasoning_budget` | `feat/reasoning-budget` | not yet proposed | Flint otherwise decides its own reasoning length (9–19k tokens per call); the grammar session forces `</think>` at the budget |
| `REC_ENGINE_ARGS` passthrough | `feat/engine-args` | not yet proposed | `rec serve` hard-codes the engine's flags; we need `--metrics` (and others) |
| `--repetition-guard` | `feat/repetition-guard` | not yet proposed | Flint with reasoning off can loop to the 32k token cap (~6 min); the guard ends exact repetition loops early (#6) |
| `--metrics-prompt-ids` | `feat/metrics-prompt-ids` | not yet proposed | Logs prompt token ids (as deltas) so we can measure prompt-lookup speculative decoding before building it (#10) |

## Build

```bash
git checkout godot
python_root="$(uv python dir)/cpython-3.12.12-macos-aarch64-none"
uv python install 3.12.12
"$python_root/bin/python3.12" packaging/build-mlx-runtime.py --python-root "$python_root" \
  --output .build/bundle-mlx-$(git rev-parse --short HEAD)   # must be a new directory
python3 packaging/package.py --runtime-dir .build/bundle-mlx-$(git rev-parse --short HEAD) \
  --manifest packaging/flint-int4-mlx.json --version 0.1.0-godot.$(git rev-parse --short HEAD) \
  --output dist/rec-godot
dist/rec-godot doctor
```

`build-mlx-runtime.py` refuses an existing output directory. If you rerun `package.py`
against an old bundle, the binary ships the **old** engine.

## Notes

- `tests/test_serving.py::test_queue_capacity_rejects_excess_and_preserves_admitted_requests`
  is timing-sensitive and can fail when the machine is loaded (e.g. during an eval). Re-run.
- rec is single-session by design: one shared KV store, batch-1 megakernel decode, and a
  serialized scheduler. Concurrency is tracked as an investigation, not a patch.
