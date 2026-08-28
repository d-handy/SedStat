# SedStat Contributing Guide

## Welcome

Welcome to the SedStat Contributing Guide. SedStat is a grain size
distribution analysis tool: statistics, classification, and plotting for
laser granulometer, sieve, and pipette lab data. It is developed in the
Cologne Geomorphological Software Laboratory at the Institute of
Geography, University of Cologne.

Contributions we accept:

* **Bug reports**
  * Incorrect statistical output
  * File-format parsing failures (`.$av` exports, sieve/pipette CSV)
  * GUI crashes, incorrect plots, or incorrect classification results
* **Feature development**
  * GUI/UX improvements
  * New plot types or export formats
  * New statistical methods
* **Tests**
  * Unit tests for statistics/classification logic
  * Widget/GUI tests (`pytest-qt`)
* **Documentation**
  * Docstrings for non-obvious methods
  * This guide and the README

At this time, we do not accept:

* Changes to statistical or classification behavior without a
  corresponding user-configurable setting and a discussion in the issue
  tracker. See *Best practices* below.
* New third-party dependencies without prior agreement
* Swapping the GUI toolkit (PySide6) or plotting library
  (matplotlib/seaborn) for an alternative

## Ground rules

* Be respectful in all written communication: issues, pull requests, and commit messages.
* Open an issue before starting significant work so the approach can be agreed on.
* One logical change per pull request. Do not bundle unrelated fixes.
* All new behaviour must be covered by tests.

## AI usage

> "In the kernel community we do open source because it results in
> better technology, not because of religious reasons. And so we make
> decisions primarily based on technical merit. Not fear of new tools."
>
> — Linus Torvalds, on AI-assisted contributions to the Linux kernel
> ([Ars Technica](https://arstechnica.com/ai/2026/07/linus-torvalds-to-critics-of-ai-coding-in-linux-fork-it-or-just-walk-away/), July 2026)

Using AI tools to help write code is allowed and welcomed. It does not
change any of the standards in this guide: judge the code on its merits,
not on how it was produced.

* **Verify by running the code, not by assuming it works.** Run the
  relevant tests or a short live script before calling something done,
  especially for numerical output. A plausible-looking value is not the
  same as a checked one.
* **Statistical or classification choices are not the AI's to decide.**
  If a change has more than one defensible answer (a rounding
  convention, an interpolation method, a reconciliation strategy), stop
  and ask before implementing it, then expose the choice as a real
  parameter or setting rather than picking one and hardcoding it.
* Every AI-generated change needs a human to actually read it before it
  is committed. Do not submit code you have not reviewed and understood
  yourself.
* AI-generated code follows the same rules as any other code, see *Best
  practices* below, especially "no speculative abstractions": prefer the
  simplest solution that solves the problem over a generated one that
  handles cases which cannot occur here.
* Install the pre-commit hooks (`uv run pre-commit install`). They run
  `ruff`, `mypy`, and the `xenon` complexity gate before each commit, the
  same checks listed under *Tests*, and catch generated code that is
  needlessly complex or fails linting/type checks before it ever reaches
  a pull request.

## Issue management

Issues are tracked in the GitHub issue tracker and follow a structured format.

When filing a bug:
1. State the expected and actual behaviour.
2. Name the affected file and line number if known.
3. Provide a minimal reproduction: a sample file or CSV snippet, GUI steps
   to reproduce, or a short Python snippet.

When filing a feature request:
1. State the problem it solves, not just the desired solution.
2. Describe the proposed change at the level of functions, widgets, or
   plot parameters.
3. Note any prerequisites (other features, new dependencies).

## Environment setup

1. Clone the repository and install [uv](https://docs.astral.sh/uv/):

   ```bash
   git clone <repo-url>
   cd SedStat
   uv sync --extra gui --dev
   ```

2. Run the test suite:

   ```bash
   uv run pytest tests/ -q
   ```

3. Run the desktop app:

   ```bash
   uv run sedstat
   ```

SedStat has no server-side component, so there is no database or
credentials to configure. Tests run headlessly
(`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py`), so no display
is required either.

## Best practices

* **No unnecessary comments.** Only add a comment when the *why* is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug. Do not describe what the code does.
* **No speculative abstractions.** Three similar lines are better than a premature helper. Only generalise when there are three or more concrete call sites.
* **No broad exception handling.** Catch only the specific exception types that can actually occur. Never use bare `except Exception`.
* **Validate at system boundaries only.** Trust numpy/scipy/Qt guarantees internally. Validate user input and data parsed from external files (`.$av`, CSV).
* **Methodological decisions are user-configurable, not hardcoded.** A statistical or classification choice that isn't the single obviously-correct answer (a reconciliation strategy, a rounding convention, an interpolation method) needs a discussion first and should end up as a real parameter or setting, not something picked once and hardcoded. 

## Coding style

* **Type hints everywhere, clean under `mypy`.** Every function
  parameter and return type, every dataclass/class field, and every
  variable whose type isn't obvious from the assignment. Modern syntax
  only: `list[float]`, `X | None`, `Literal[...]`. Not `typing.List`,
  `typing.Optional`, or `typing.Union`. `mypy` runs in its default mode
  here, not `--strict` (see `pyproject.toml`, no `[tool.mypy]` section
  beyond the third-party ignore overrides).
* **`from __future__ import annotations`** at the top of every module.
* **Google-style docstrings** (`Args:` / `Returns:` / `Raises:`) where a
  docstring is warranted at all, consistent with *Best practices*' "no
  unnecessary comments" above. Most private helpers need none.
* **Private helpers are prefixed with `_`** and not exported from
  `__init__.py`. Only the public API surface is unprefixed.
* `ruff check .` and `mypy sedstat` must both pass clean. Line length is
  100 columns (`pyproject.toml`'s `[tool.ruff]`).


## Contribution workflow

### Branch creation

Branches follow the pattern `<type>/<short-description>`:

* `bug/fix-ternary-label-overlap` for bug fixes
* `feat/plot-export-dialog` for features
* `test/sieve-pipette-overlap-strategy` for test-only changes
* `chore/remove-legacy-classification-table` for cleanup

### Commit messages

Write commit messages in the imperative mood, present tense:

```
Fix D50 annotation using the wrong class edge in phi mode

Add overlap_strategy parameter to combine_sieve_pipette
```

* First line: 50 characters max, no trailing period.
* Optional body: explain *why*, not *what*. Reference issue numbers (`Closes #XX`).
* Do not amend published commits.

### Pull requests

* Open against `main`.
* Title mirrors the commit message style.
* Description must state: what changed, why, and how it was tested.
* Link the corresponding issue (`Closes #XX`).
* At least one approving review is required before merging.

### Tests

Run the test suite with:

```bash
uv run pytest tests/ -q
```

Also expected to stay clean on any touched code:

```bash
uv run ruff check .
uv run mypy sedstat
uv run xenon --max-absolute B --max-modules B --max-average A sedstat
```

On Linux, PySide6 needs a handful of system Qt libraries even under the
offscreen platform. See `.github/workflows/ci.yml` for the exact list if
imports fail.

### Code organisation

| Package | Scope |
|---|---|
| `sedstat.core` | Statistics, classification, fractions, percentiles, Stokes' law, bimodality |
| `sedstat.io` | File readers/writers: Beckman Coulter LS (`.$av`), sieve/pipette CSV |
| `sedstat.gui` | Desktop app: main window, dialogs, embedded plot widgets |
| `sedstat.plotting` | Standalone, Qt-free matplotlib figure factories, shared by the GUI and the Python API |

`sedstat.plotting` functions must stay usable without a `QApplication`,
since they are also called directly as the standalone Python API.

### Releases

There is no fixed release cadence. 