# Testing and CI

Read this when editing `tests/**` or `.github/workflows/**`.

## 1. Test surfaces

- Test a module through its interface: stages through `image`/`solve`/`update` on synthetic snapshots
  (`tests/test_imaging_stages.py`), the CLI through Typer's `CliRunner` with `core` stubbed
  (`tests/test_cli.py`), runners through the runner interface (`tests/test_runners.py`), and command
  orchestration through `kremetart()` on the loop runner with only the web server faked
  (`tests/test_kremetart.py`). A test that has to reach past an interface is a sign the module is the wrong
  shape.
- New behaviour in a stage gets a NumPy test; GPU agreement is checked once per chain, not per test.
- One test per behaviour, at the highest interface that shows it. Test a lower-level function directly only
  for a property the interface test cannot see (e.g. the Hessian's self-adjointness in
  `test_imaging_healpix.py`). Investigations against other tools are one-off scripts, not tests.

## 2. Platforms, GPU, real-data and parity tests

CI runs the suite on Ubuntu (Python 3.11–3.13), macOS 15 on Apple Silicon (3.12) and Windows (3.12),
none with a GPU. Each job downloads that platform's portable CyberEther build and sets
`KREMETART_CYBERETHER`; Linux and Windows get software Vulkan (Mesa lavapipe) for it, macOS uses Metal. A dependency that is missing on a
platform (healpy and the Measurement Set readers on Windows, CuPy and Holoscan off Linux) is skipped with
`pytest.importorskip` where the test uses it, so the rest of the module still runs there.

These are gated so the default suite runs on CPU-only CI runners:

| Test | Runs when |
|---|---|
| Holoscan/CuPy tests (Holoscan runner and its viewer fan-out in `test_runners.py`, CuPy agreement in `test_imaging_stages.py`) | a CUDA device is present (skip otherwise) |
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

The documentation (Sphinx, hosted on Read the Docs) is a quick read for people, generated, never written:
the home page from `README.md` (repo-relative links pointed at GitHub, via `docs/readme_page.py`), the
command page from the Typer help strings (as tables, via `docs/command_page.py`), the Python reference from docstrings (Parameters, Raises and Returns as tables,
via `docs/field_tables.py`), and the glossary from `GLOSSARY.md` (one table per section, without the
`_Avoid_` lines, via `docs/glossary_page.py`). The ADRs, design notes and `docs/agents/` are the agents'
record and stay out of the site. Pages under `docs/` only arrange the sources. Build it after changing a
docstring, help string, the README or the glossary:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

CI's Code Quality job runs the same build; a warning fails it. The build also writes `llms.txt` and
`llms-full.txt`, the whole site as Markdown for agents.

## 5. Commits

[Conventional Commits](https://www.conventionalcommits.org/): `<type>: <description>` with `feat`,
`fix`, `refactor`, `perf`, `docs`, `test`, `ci`, `deps` or `chore`. Imperative mood, first line under
72 characters.
