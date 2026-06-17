# Python Standards & CLI Implementation Guidelines

Read this when editing or creating any `**/*.py` files.

## 1. Type Hints and Modern Python

- **Python 3.10+.** Use modern syntax (`X | Y`, `list[int]`, etc.).
- **Type hints on every function signature.**
- Use `from typing import Annotated` for Typer parameter annotations.

## 2. Import Placement

**Imports go at the top of the module** (PEP 8 order: stdlib, third-party,
first-party). This holds for `core/`, `utils/` and `operators/`. Do **not** scatter
`import` statements inside function bodies out of habit — top-level imports make a
module's dependencies legible at a glance, and the container fallback already tolerates
heavy top-level imports in `core/` (the `cli` wrapper imports `core` inside a
`try/except ImportError`). Ordinary stdlib and heavy third-party deps (`numpy`,
`astropy`, `healpy`, `xarray`, `tempfile`, …) belong at the **top** of `core/`/`utils/`
modules — there is no reason to defer them.

Put an import inside a function body **only** for one of these specific reasons, and add
a one-line comment saying which:

1. **`cli/` → `core/` (required).** A CLI wrapper imports its core implementation
   *inside the function body*, never at module scope. This is the hinge of the
   container-fallback pattern (`architecture.md` §3) and keeps the lightweight install
   importable without the heavy deps. CLI wrappers must stay free of heavy/domain
   imports at module scope entirely.
2. **GPU-only deps in a module that must stay importable without a GPU.** `cupy` and
   `holoscan` are imported at module top *only* in `operators/` and other inherently
   GPU-only modules. A `core/`/`utils/` module that has a non-GPU path (e.g.
   `core/smoovie.py`, which falls back to CPU imaging) keeps them lazy — import the
   GPU module inside the GPU branch — so the CPU path stays importable on a machine
   without `cupy`/`holoscan`. (A core command that is *inherently* GPU-only, like
   `core/stream_msv4.py`, may import them at top.)
3. **fsspec backends stay lazy.** Never import `s3fs`, `gcsfs`, or `adlfs` directly;
   fsspec loads the matching backend on demand when a remote UPath is first accessed.
4. **Breaking a genuine import cycle.** Prefer to remove the cycle by moving the shared
   code into `utils/` (see `architecture.md` §1) over papering over it with a
   function-body import.

## 3. Typer Option / Argument Syntax (CRITICAL)

**Never pass `None` as the positional default to `typer.Option()`** — it raises
`AttributeError`. Follow these exact patterns:

- **Required:** `Annotated[T, typer.Option(..., help="...")]` (no `= default`).
- **Optional w/ default:** `Annotated[T, typer.Option(help="...")] = default`.
- **Optional None:** `Annotated[T | None, typer.Option(help="...")] = None`.

## 4. hip-cargo Types

- **Comma-separated lists:** use `ListInt`, `ListFloat`, `ListStr` from
  `hip_cargo`, with their matching `parse_list_*` parsers. Typer cannot natively
  handle variable-length lists as a single option; these `NewType` wrappers wrap
  `str` for Typer but parse into `list[int]` etc. at runtime.
- **UPath-backed path types:** `File`, `Directory`, `MS`, `URI` are
  `NewType(..., UPath)`. Generated CLIs use `parser=parse_upath` so the same
  signature accepts local paths and remote URIs. User functions receive a
  `universal_pathlib.UPath` and call `.open()` / `.exists()` directly.
- **The `stimela` metadata dict:** `Annotated[..., StimelaMeta(...)]` overrides
  inferred cab metadata. Use `StimelaMeta(skip=True)` to exclude a parameter from
  the generated cab YAML entirely.

## 5. Architectural Style

- Prefer functional, explicit-over-implicit code. Use classes when state or
  polymorphism genuinely helps.
- Keep `core/` implementations straightforward; let exceptions propagate. Use
  `typer.Exit(code=1)` for CLI errors in `cli/` (never in `core/`).
- Use Google-style docstrings (document Args, Returns, Raises). Keep them
  concise. Add short inline comments only when intent isn't obvious.

## 6. Private vs. Shared Names

A **leading underscore means module-private**: `_helper` may only be used inside the
module that defines it. If a function (or constant/class) is imported by *any* other
module, it is part of an API and must **not** carry a leading underscore.

- A helper used across modules is **shared code**: give it a public (non-underscore)
  name and move it to `utils/` (per `architecture.md` §1) — not a `_name` reached out of
  a `core/` command. Example: `_partition` / `_utc` defined in `core/smoovie.py` and
  imported from `utils/satellites.py` is wrong twice over — wrong layer (a `core` command
  acting as a utility provider; see `architecture.md` §1) *and* wrong visibility (a
  "private" name used across modules).
- Reserve `_name` for helpers that genuinely stay within their defining module.

So `utils/` and `operators/` expose public, non-underscore functions; `core/<cmd>.py`
keeps its `_name` helpers truly local, or promotes them to `utils/` when a second module
needs them.
