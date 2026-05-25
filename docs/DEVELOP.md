# Contributing

## Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/developmentseed/lazymerge.git
cd lazymerge
uv sync --group dev --group docs
```

## Running tests

```bash
uv run pytest
```

## Linting and type checking

```bash
uv run ruff check .
uv run mypy lazymerge/ tests/
```

Or run everything through pre-commit:

```bash
uv run pre-commit run --all-files
```

## Building docs

```bash
uv run mkdocs serve
```

This starts a local dev server at `http://127.0.0.1:8000/`.

## Pre-commit hooks

Install the hooks so they run automatically on commit:

```bash
uv run pre-commit install
```
