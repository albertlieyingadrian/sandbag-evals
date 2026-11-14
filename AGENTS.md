# Project instructions

sandbag-evals holds [Inspect](https://inspect.aisi.org.uk/) evals for AI sandbagging (a model scoring lower than it can, on purpose). Each eval is a `@task` in its own package under `src/sandbag_evals/<eval>/`, with its data next to it, and runs on any Inspect model. The evals are listed in the Inspect Evals Register at a pinned commit, so the README is the eval card: keep its results table in step with the code. Explain in plain language, with analogies and small worked examples: readers include people new to ML.

## Workflow

- Work on a feature branch from up-to-date `main`. Use worktrees only when asked, and put them in `.claude/worktrees/`.
- Changes land through pull requests, rebase-merged once CI is green. Never push to `main`.
- When recursive deletes are blocked, use `trash`.
- Commit messages are public: describe the change, nothing else.
- A change to a task's prompt, data, scorer or metrics bumps its `version` in the `Task(...)` call, adds a line to the eval's changelog in the README, and reruns the results table.

## Checks

- `uv run pytest` (offline, uses Inspect's mock model), `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`. CI runs all four.
- `uvx inspect-evals-lint --preset single-eval --all` runs the linter the Register uses.

## Public repo hygiene

- Everything committed here is public. Never add personal notes, private plans, or files from outside this repo.
- Never commit model weights. Assets are pinned: Hugging Face files by `revision=` (a commit SHA), GitHub raw files by commit.
- No absolute paths to a personal machine in code, docs or committed logs: use paths relative to the repo root. Check `.eval` logs before committing them (`uv run inspect log dump <file>`).

## Secrets

- `.env` holds API keys. Never print, echo, or commit it.
- Never print `~/.cache/huggingface/token`.

## Running models

- Use `uv run` (or `.venv/bin/python`). Local models need `uv sync --extra hf`.
- Inspect's `hf/` provider samples unless given `-M do_sample=false`; pass it for greedy runs.
- Inspect reads `-T` and `-M` values as YAML: quote strings that start with `|`, such as `-T "password='|TEST|'"`.
