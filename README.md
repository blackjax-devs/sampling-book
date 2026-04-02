# The Sampling Book

The Sampling Book is a series of tutorials on sampling algorithms built with the [BlackJAX](https://github.com/blackjax-devs/blackjax) library. The book is built with [Jupyter Book 2](https://jupyterbook.org/) and published to GitHub Pages.

## Setup

Install dependencies into a conda/mamba environment:

```bash
mamba create -n sampling_book python=3.13
mamba activate sampling_book
pip install -r requirements.txt
```

## Building the book

Build the static site with notebook execution:

```bash
cd book && jupyter book build --execute --html
```

Static HTML output goes to `book/_build/html/`.

For a local preview with live reload:

```bash
cd book && jupyter book start --execute
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
pre-commit run --all-files
```
