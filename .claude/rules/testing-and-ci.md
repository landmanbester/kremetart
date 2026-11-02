# Testing and CI

Read this when editing `tests/**` or `.github/workflows/**`.

## 1. Test surfaces

- Test a module through its interface: stages through `image`/`solve`/`update` on synthetic snapshots
  (`tests/test_imaging_stages.py`), the CLI through Typer's `CliRunner` with `core` stubbed
  (`tests/test_cli.py`), runners through the runner interface (`tests/test_runners.py`), and command
  orchestration through `smoovie()` on the loop runner with only the web server faked
  (`tests/test_smoovie.py`). A test that has to reach past an interface is a sign the module is the wrong
  shape.
- New behaviour in a stage gets a NumPy test; GPU agreement is checked once per chain, not per test.

## 2. GPU, real-data and parity tests

These are gated so the default suite runs on CPU-only CI runners:

| Test | Runs when |
|---|---|
| Holoscan/CuPy tests (Holoscan runner in `test_runners.py`, fan-out in `test_serve_integration.py`, CuPy agreement in `test_imaging_stages.py`) | a CUDA device is present (skip otherwise) |
| CyberEther runs in `test_cyberether_runner.py` (its CUDA case also needs a CUDA device) | `KREMETART_CYBERETHER=<AppImage>` or `cyberether` on `PATH` |
| `test_stefcal_realdata.py` | `KREMETART_REALDATA=1` |
| `test_calibration.py` tart2ms oracle | `KREMETART_MS_ORACLE=1` |
| `test_imaging_parity.py` | `KREMETART_GOLDEN=<dir>` holding reference runs; `KREMETART_GOLDEN_NFRAMES` sets how many snapshots |

Parity tolerances follow measured numerics and their reasons are recorded in the test docstring. A
parity failure gets investigated; the tolerance moves only with a new measurement and reason.

Holoscan needs a 32 MB stack: run GPU tests after `ulimit -s 32768`.

## 3. Test hygiene

Write temporary files under `tmp_path` (or `tempfile.TemporaryDirectory()`); the repository stays
clean after a test run. Test data under `tests/data/` is gitignored and fetched on demand
(`tests/conftest.py`); `KREMETART_OFFLINE=1` skips the download.

## 4. Dev workflow

After every code change:

```bash
uv run ruff format . && uv run ruff check . --fix
```

The pre-commit hook and CI enforce the same configuration.

## 5. Commits

[Conventional Commits](https://www.conventionalcommits.org/): `<type>: <description>` with `feat`,
`fix`, `refactor`, `perf`, `docs`, `test`, `ci`, `deps` or `chore`. Imperative mood, first line under
72 characters.
