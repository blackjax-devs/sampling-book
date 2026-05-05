# The Sampling Book

The Sampling Book is a series of tutorials on sampling algorithms built with the [BlackJAX](https://github.com/blackjax-devs/blackjax) library. The book is built with [Jupyter Book 2](https://jupyterbook.org/) and published to GitHub Pages.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Install dependencies:
   ```bash
   make install  # runs: uv sync --group book
   ```
   This creates a local `.venv/` and installs all dependencies. No need to activate — prefix commands with `uv run`, or run `source .venv/bin/activate` once per shell session.

   **GPU/CUDA users:** `uv sync` installs the CPU build of JAX by default. For GPU work, either run `uv pip install "jax[cuda12]"` afterwards, or use mamba to manage the CUDA toolkit and run `uv sync` inside that conda environment.

## Building the book

Build the static site with notebook execution:

```bash
make build  # runs: cd book && uv run jupyter book build --execute --html
```

Static HTML output goes to `book/_build/html/`.

For a local preview with live reload:

```bash
make preview  # runs: cd book && uv run jupyter book start --execute
```

This serves the book at `http://localhost:3000`.

## Contributing

All tutorial notebooks are MyST Markdown files (`.md`) under `book/`, organized into:

- `book/algorithms/` — sampling algorithm showcases
- `book/models/` — model-specific examples

### Converting between `.ipynb` and `.md`

Notebooks are stored as MyST Markdown (`.md`) via [jupytext](https://jupytext.readthedocs.io/). To convert between formats during development:

```bash
# .md → .ipynb (for interactive editing in Jupyter)
jupytext --to ipynb book/algorithms/my_notebook.md

# .ipynb → .md (convert back before committing)
jupytext --to myst book/algorithms/my_notebook.ipynb
```

Only commit the `.md` files — `.ipynb` files are for local development only.

### Linting

Pre-commit hooks enforce formatting (black, isort, flake8) for both Python source and notebooks:

```bash
make lint  # runs: uv run pre-commit run --all-files
```
