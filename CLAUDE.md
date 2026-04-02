# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The Sampling Book is a [Jupyter Book](https://jupyterbook.org/) containing tutorials on sampling algorithms built with the [BlackJAX](https://github.com/blackjax-devs/blackjax) library. Notebooks cover both algorithm showcases and model-specific examples.


## Commands

All commands should be run via `mamba run -n sampling_book <command>`.

### Install dependencies
```bash
mamba run -n sampling_book pip install -r requirements.txt
```

### Build the book
```bash
cd book && mamba run -n sampling_book jupyter book build --execute --html
```

Static HTML output goes to `book/_build/html/`. Execution results are cached in
`book/_build/execute/` (auto-invalidates when cell code changes; clear with
`mamba run -n sampling_book jupyter book clean --execute`).

For local preview (dev server):
```bash
cd book && mamba run -n sampling_book jupyter book start --execute
```

### Linting / formatting
Pre-commit hooks handle formatting for both Python source and notebooks:
```bash
mamba run -n sampling_book pre-commit run --all-files
```

Tools configured: `black`, `isort` (black profile), `flake8` (ignores E501, E203, E731, W503), `mypy`, `pyupgrade`, and their `nbQA` notebook equivalents (`nbqa-black`, `nbqa-isort`, `nbqa-flake8`).

### PR Workflow

Always branch from `origin/main`:
```bash
git fetch origin
git checkout -b my-branch origin/main
```

Before opening or updating a PR, always rebase onto the latest `origin/main`
and force-push:
```bash
git fetch origin && git rebase origin/main
git push --force-with-lease
```

## Architecture

### Notebook format
All tutorial notebooks live under `book/` as MyST Markdown files (`.md` with jupytext frontmatter, `format_name: myst`). They are **not** `.ipynb` files — jupytext converts them at build time. The TOC is defined in the `toc` section of `book/myst.yml`.

- `book/algorithms/` — notebooks showcasing specific BlackJAX samplers (NUTS, SMC, MCLMC, MEADS, LAPS, etc.)
- `book/models/` — notebooks showing BlackJAX applied to specific statistical models (GPs, logistic regression, BNNs, ODE solvers, etc.)

### Shared Python utilities (`src/`)
A small installable package (`samplingbook`, installed as `-e .`) provides reusable helpers:
- `src/models/sparse_regression.py` — Aesara/Aeppl model definition and log-density helper for the German Credit sparse regression example

This package is declared in `pyproject.toml` and installed via `requirements.txt` (`-e .`). Add shared model/utility code here when multiple notebooks need it.

### Key dependencies
- **BlackJAX** — installed from `main` branch (`git+https://github.com/blackjax-devs/blackjax.git@main`)
- **JAX / jaxlib** — numerical backend
- **Aesara / Aeppl** — used in some older notebooks for model specification
- **ArviZ** — posterior diagnostics and plotting
- **NumpPyro, TFP** — used in select notebooks for comparison or model building

### CI
`.github/workflows/test.yml` — builds the book on PRs to `main`, using a cache keyed by PR number.
`.github/workflows/publish.yml` — publishes the built book (triggered separately).

