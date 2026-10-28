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

Based on upstream v0.1.1. Our M4 Max profile (`upstream-pr/17`) was dropped: upstream closed
the PR and shipped its own M4 Max kernels in v0.1.1.

| Change | Branch | Upstream status | Why we need it |
|---|---|---|---|
| Keep 64 prefix checkpoints instead of 2 | `upstream-pr/24` | open: recursive-oss/rec#24 | Agent harnesses send long shared preambles; 2 checkpoints reuse none of them |
| `reasoning_effort` / `reasoning_budget` | `feat/reasoning-budget` | not yet proposed | Flint otherwise decides its own reasoning length (9–19k tokens per call); the grammar session forces `</think>` at the budget |
| `REC_ENGINE_ARGS` passthrough | `feat/engine-args` | not yet proposed | `rec serve` hard-codes the engine's flags; we need `--metrics` (and others) |
| `--repetition-guard` | `feat/repetition-guard` | not yet proposed | Flint with reasoning off can loop to the 32k token cap (~6 min); the guard ends exact repetition loops early (#6) |
| `--metrics-prompt-ids` | `feat/metrics-prompt-ids` | not yet proposed | Logs prompt token ids (as deltas) so we can measure prompt-lookup speculative decoding before building it (#10) |
| `--prompt-cache-slots` | `feat/prompt-cache-slots` | not yet proposed | Keeps several conversations cached (subagents, harness roles) so switching back doesn't reread the whole prompt; validated bitwise against cold prefill on Flint (#8) |
| Accept identical `reasoning` + `reasoning_content` | `feat/reasoning-field-alias` | not yet proposed | Hermes Agent replays reasoning under both fields to local servers; stock rec rejects every turn after the first |

## Build

```bash
git checkout godot
python_root="$(uv python dir)/cpython-3.12.12-macos-aarch64-none"
uv python install 3.12.12
"$python_root/bin/python3.12" packaging/build-mlx-runtime.py --python-root "$python_root" \
  --output .build/bundle-mlx-$(git rev-parse --short HEAD)   # must be a new directory
python3 packaging/package.py --runtime-dir .build/bundle-mlx-$(git rev-parse --short HEAD) \
  --manifest packaging/flint-int4-mlx.json --version 0.1.1-godot.$(git rev-parse --short HEAD) \
  --output dist/rec
dist/rec doctor
```

`build-mlx-runtime.py` refuses an existing output directory. If you rerun `package.py`
against an old bundle, the binary ships the **old** engine.

## Install

Install a build the way `install.sh` does, so `rec uninstall` recognises it: an immutable
release directory with a `.managed` hash, and `~/.local/bin/rec` linking to it.

```bash
v=$(dist/rec --version | awk '{print $2}'); r=~/.local/share/rec/releases/$v
mkdir "$r" && cp dist/rec "$r/rec" && shasum -a 256 "$r/rec" | awk '{print $1}' > "$r/.managed"
ln -sfn "$r/rec" ~/.local/bin/rec
launchctl kickstart -k gui/$(id -u)/com.hybridindie.rec-flint   # restart the Flint service
```

Never run `rec update`: it runs the upstream installer and relinks `rec` to a stock release,
dropping every patch above.

## Notes

- rec is single-session by design: one shared KV store, batch-1 megakernel decode, and a
  serialized scheduler. Concurrency is tracked as an investigation, not a patch.
