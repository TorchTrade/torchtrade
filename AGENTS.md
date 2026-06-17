# AGENTS.md — TorchTrade

Operating guidance for AI coding agents (Claude Code, Codex, Cursor, the `@claude`
GitHub Action, and any other assistant) working in this repository. This file is
self-contained and vendor-neutral: it is the source of truth that every agent reads,
regardless of tool. Human contributors should also skim it — see `README.md` for the
full project documentation and `Contributing` section.

> **TorchTrade is a research and engineering library — not a trading product, a
> strategy, or financial advice.** Everything below exists to keep it that way.

---

## 1. What this repository is

- **Project:** `torchtrade` — reinforcement-learning **environments** for trading
  applications (offline backtesting + live-exchange env wrappers). It is a library,
  published to PyPI, with docs at `torchtrade.github.io/torchtrade`.
- **Stack:** Python `>=3.11`, `hatchling` build backend, `uv` for dependency/venv
  management, `pytest` for tests, `mkdocs-material` for docs.
- **Heavy dependencies (relevant to risk):** `torchrl`, live broker/exchange clients
  (`alpaca-py`, `python-binance`, `pybit`, `ccxt`), `wandb`, HuggingFace `datasets`,
  and optional extras `llm` (`openai`, `vllm`, `transformers`, `accelerate`,
  `bitsandbytes`) and `chronos` (`chronos-forecasting`).
- **Layout:**
  - `torchtrade/` — the package: `envs/` (`core/`, `offline/`, `live/`, `transforms/`),
    `actor/`, `losses/`, `metrics/`, `models/`.
  - `examples/` — runnable scripts (`offline_rl/`, `online_rl/`, `rule_based/`,
    `losses/`, `transforms/`, `llm/`); some place **live trades** (e.g.
    `examples/rule_based/live.py`).
  - `tests/` — `pytest` suite with `mocks/` and `conftest.py`.
  - `docs/` + `mkdocs.yml` — documentation site.

## 2. Ownership & branch model — read before you branch

- **This is a shared, multi-maintainer open-source upstream**, not a personal fork.
  `origin` is the canonical repo `https://github.com/TorchTrade/torchtrade.git`
  (org: TorchTrade; author: Sebastian Dittert). There is **no separate `upstream`
  remote** — `origin` *is* upstream.
- **Base branch: `main`.** Releases and the docs site deploy from `main`.
- **Contribution flow (from `README.md`):** fork → feature branch
  (`feature/<name>` / `fix/<name>` / `docs/<name>`) → `pytest tests/ -v` → PR into
  `main`. Never push directly to `main`; never force-push shared branches.
- Because changes here reach a public OSS project and other maintainers, **prefer the
  smallest correct change** and preserve existing project conventions, attribution,
  and the MIT `LICENSE`.

## 3. Golden rules — financial & execution safety (non-negotiable)

This is a trading-adjacent codebase. Agents must treat money, markets, and credentials
as live wires.

- **No live trading.** Do not place, modify, or cancel orders; do not call broker or
  exchange APIs that mutate an account (Alpaca, Binance, Bitget, Bybit, or any `ccxt`
  venue). Read-only/code-level work only.
- **No real-money execution paths.** Do not change defaults so that an example or env
  trades with real funds. Where examples expose `paper=True` / sandbox / testnet flags,
  keep them safe-by-default; never flip a paper/testnet default to live.
- **No portfolio or account mutations.** No position changes, transfers, withdrawals,
  API-key provisioning, or balance operations.
- **No financial advice, performance claims, or return guarantees.** Do not add
  language promising profit, "guaranteed" returns, backtested-edge claims, or
  recommendations to trade. Keep docs descriptive and neutral; preserve existing
  risk/disclaimer framing.
- **No credentials, ever.** Do not read, print, echo, log, commit, or transmit the
  contents of `.env`, broker/exchange keys, `WANDB_API_KEY`, HuggingFace tokens,
  `CODECOV_TOKEN`, `GH_TOKEN`, or any secret. `.env.example` (placeholders only) is the
  only credential-shaped file you may reference.

When a requested task would cross any of these lines, **stop and ask a human** instead
of finding a workaround.

## 4. Protected surfaces — do not read, print, mutate, or commit

**Secrets & credentials (deny):**
- `.env`, `.envrc`, anything matching `*-private*`, `.pypirc`.
- Broker/exchange keys (`API_KEY`, `SECRET_KEY`, `BINANCE_*`, `BITGET_*`, `BYBIT_*`),
  `WANDB_API_KEY`, `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`, `CODECOV_TOKEN`, `GH_TOKEN`,
  `CLAUDE_CODE_OAUTH_TOKEN`.

**Generated / data / experiment artifacts (already in `.gitignore` — never add, never
inspect private contents):**
- `torchtrade/data/`, `real_live_data/`, `outputs/`, `replay_buffer_random.pt`,
  `*.pth` (model checkpoints), `data_preprocessing.ipynb`, `.ipynb_checkpoints/`.
- Build/test/coverage output: `build/`, `dist/`, `*.egg-info/`, `htmlcov/`,
  `coverage.xml`, `.pytest_cache/`, `/site` (built docs), `wandb/` run dirs.

If a task seems to require one of these, surface it rather than touching it.

## 5. Commands — what's safe vs. what needs explicit human approval

**Safe (read-only, no side effects) — fine to run anytime:**
- `git status`, `git diff`, `git log`, `git branch -vv`, `git remote -v`,
  `git check-ignore -v <path>`.
- Reading source, docs, configs, and `.env.example`.

**Requires explicit human approval before running** (they install, execute, download,
train, or hit networks/markets — never run them autonomously):
- Dependency / environment: `uv sync` (+ `--extra dev|docs|llm|chronos|all-extras`),
  any `pip`/`uv pip install`.
- Tests: `uv run pytest tests/ -v` (and the coverage variant). The suite uses mocks but
  also exercises `datasets`/HF and example scripts — treat as approval-gated.
- Docs: `mkdocs serve` / `mkdocs build` / `mkdocs gh-deploy`.
- Anything under `examples/` (training, backtests, **live trading**), model training,
  data downloads (HF `datasets`), `wandb` runs, and any LLM inference (`vllm`,
  `openai`, `transformers`).
- Release/publish (`hatchling` build, PyPI upload), and any GitHub Actions trigger.

Default posture: **propose the command and wait** for a human to run or approve it.

## 6. Validation for code changes

When changes *are* approved to be validated, the canonical commands are:

```bash
uv sync --extra dev                 # set up dev environment (approval-gated)
uv run pytest tests/ -v             # run the test suite
uv run pytest tests/ -v --cov=torchtrade --cov-report=html   # with coverage
mkdocs serve                        # preview docs locally
```

Keep new code consistent with the surrounding style; add or update tests under
`tests/` for behavior changes; keep public APIs and example defaults backward- and
safety-compatible.

## 7. Secret-scan discipline before any commit

Even for docs-only changes, scan staged content before committing. Inspect **added
lines only** and **filenames**, never secret values themselves:

```bash
# Staged filenames that look credential/data/artifact-shaped (review, don't commit):
git status --short | cut -c4- | grep -iE '\.env($|\.)|\.pem$|\.pth$|/data/|real_live_data|outputs/|replay_buffer|\.ipynb$|secret|token|api[_-]?key' || echo "clean: no sensitive filenames staged"

# Added-lines-only scan for secret-shaped values:
git diff --cached | grep -nE '^\+' | grep -iE '(api[_-]?key|secret|token|passphrase|password|bearer|wandb|hf_[a-z0-9])' || echo "clean: no secret-shaped added lines"
```

If either prints a hit, stop and review by hand before committing.

## 8. Scope discipline for agents

- Touch the **minimum** set of files needed; do not opportunistically refactor on a
  shared OSS repo.
- Do not edit `.gitignore`, CI workflows under `.github/workflows/`, `pyproject.toml`
  dependency pins, or release config unless that is the explicit task.
- Do not introduce new dependencies without a human decision.
- A repo-local Claude Code operating kit (`CLAUDE.md`, `.claude/commands/`) may exist
  in a maintainer's working tree. Those files are **intentionally git-ignored**
  (see `.gitignore`) and are developer-local — do **not** force-add or commit them, and
  do not assume other contributors have them. This `AGENTS.md` is the shared,
  committed source of truth.
