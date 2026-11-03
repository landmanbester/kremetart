# Python standards

Read this when editing or creating any `**/*.py` file.

## 1. Type hints

- Python 3.11+ syntax (`X | Y`, `list[int]`).
- Type hints on every function signature, tests included. Arrays that may be NumPy or CuPy use the
  `Array` alias from `kremetart.imaging.snapshot`; array modules are `types.ModuleType`.

## 2. Import placement

Imports go at the top of the module, in PEP 8 order (stdlib, third-party, first-party), so a module's
dependencies are legible at a glance. Heavy dependencies (`numpy`, `xarray`, `astropy`, `zarr`, …)
belong at the top too.

An import goes inside a function body only for one of these reasons, with a one-line comment naming it:

1. **`cli.py` → `core`**: a command imports its `core` function inside its body so `kremetart --help`
   stays fast.
2. **GPU-only dependencies in a module with a CPU path**: `cupy` and `holoscan` sit at module top only in
   inherently GPU modules (`operators/`). A module that also runs on NumPy imports them inside the GPU
   branch, so it stays importable on a machine without a GPU (e.g. `utils/beam.py`'s Bessel function).
3. **Breaking a genuine import cycle**: first try moving the shared code down the dependency chain
   (`architecture.md`).

## 3. Typer options

`typer.Option()` takes keyword arguments only; a positional `None` default raises `AttributeError`.

- Required: `Annotated[T, typer.Option(help="...")]` with no `= default`.
- Optional with default: `Annotated[T, typer.Option(help="...")] = default`.
- Optional, may be `None`: `Annotated[T | None, typer.Option(help="...")] = None`.

## 4. Style

- Prefer functional, explicit code; use a class when state or polymorphism earns it (stages hold
  per-snapshot state, so they are classes).
- Let exceptions propagate from `core/`, `imaging/` and `utils/`; `typer.Exit(code=1)` belongs in
  `cli.py` only.
- Google-style docstrings (Args, Returns, Raises), concise. Inline comments only where intent is not
  obvious.

## 5. Private vs shared names

A leading underscore means module-private: `_helper` is used only inside the module that defines it.
Anything imported by another module is part of an interface and has a public name. A helper that a
second module needs gets a public name and moves down the dependency chain to the lowest package both
can import (`architecture.md`).
