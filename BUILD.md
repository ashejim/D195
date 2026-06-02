# Building & Publishing the D195 / D502 Site

This repo builds with [Jupyter Book](https://jupyterbook.org/) inside a
[uv](https://docs.astral.sh/uv/)-managed Python environment, defined by
`pyproject.toml` + `uv.lock` (both committed), so it reproduces identically on any
machine — no manual Python or pip setup.

## One-time, per machine

Install **uv**: <https://docs.astral.sh/uv/getting-started/installation/>
(Windows: `winget install astral-sh.uv`, or the install script on that page.)

That is the only prerequisite — uv fetches the pinned Python (3.12) and all
dependencies for you.

## Build & publish

Run the deploy script from this folder:

    update_D195_site.bat

It prepares the env (`uv sync`), pulls latest from GitHub, builds the book
(`uv run jupyter-book build .`), commits + pushes the source to `main`, then
publishes `_build/html` to the `gh-pages` branch (`uv run ghp-import ...`).

Build only, no publish:

    uv run jupyter-book build .

## Using this in your other local clones

After `git pull`, run the `.bat` (or `uv sync` once) and uv recreates the local
`.venv` from `uv.lock`. The `.venv` is not committed (it is in `.gitignore`); uv
rebuilds it per machine. Prerequisites per machine: uv installed + internet for the
first `uv sync`.

## Dependencies / "requirements file"

With uv the dependencies live in `pyproject.toml` and are pinned in `uv.lock` — those
are the "requirement files." Add or upgrade with `uv add <package>` (commit both).

- Jupyter Book is pinned **`<2`** — 2.x is the incompatible mystmd rewrite.
- Python is pinned to **3.12** via `.python-version` (Jupyter Book / Sphinx do not yet
  fully support 3.14). uv fetches 3.12 automatically on any machine that lacks it.
