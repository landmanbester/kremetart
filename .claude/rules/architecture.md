# Architecture

Read this when editing `src/kremetart/**`.

## Layout

| Package | Holds | Imports from |
|---|---|---|
| `cli.py` | The Typer app: one thin function per command, forwarding its options to `core`. | `core` (inside the command, so `kremetart --help` stays fast). |
| `core/` | Host orchestration of a command: data preparation, catalog, web server, profiling, running the streaming app. | `operators`, `utils`. |
| `operators/` | Holoscan **adapters** (receive arrays from ports, build a `Snapshot`, call a stage, emit arrays), plus the zarr reader/writer and the web sink. | `imaging`, `utils`. |
| `imaging/` | The `Snapshot`, the `SkyGrid`, the four **stages** (dirty imager, Tikhonov solver, L1 solver, IWP filter) and the `SnapshotImager` that chains them. All `xp`-injectable. | `opt`, `utils`. |
| `opt/` | Generic solvers (CG, FISTA), `xp`-injectable. | itself. |
| `utils/` | Host helpers: sky geometry, HDF preparation, calibration and gains, satellites, the web server and frame encoding. | itself. |
| `static/` | The three.js web viewer. | |

Dependencies point one way, down this chain: `cli → core → operators → imaging → {opt, utils}`, with
`core` also using `utils`. A helper that a lower package needs moves down into that package under a
public name (`python-standards.md` §5).

## Stages and adapters

Per-snapshot logic (beam, no-data handling, regularisation scaling, warm starts, Kalman state) belongs
in a stage in `imaging/`, where it is tested on NumPy through the stage's interface. Stages exchange
only plain arrays, and each builds its own DFT kernel; `docs/adr/0001-independent-imaging-stages.md`
records why. A stage takes its array module as an explicit `xp` and keeps its state on that device.

A runtime reaches the stages only through thin adapters: a Holoscan operator today, a CyberEther Python
block after the port. Behaviour changes go in the stage; plumbing changes go in the adapter.

## Commands

`cli.smoovie` and `core.smoovie.smoovie` take exactly the same parameters (`tests/test_cli.py` checks
this), so a user-facing knob goes on both. Environment differences such as GPU vs CPU are detected
inside the code; tests force a path by monkeypatching the detector.

## Packaging and releases

- One dependency list in `pyproject.toml`: `pip install kremetart` installs everything, including the
  CUDA 13 stack (Holoscan, CuPy, cuBLAS).
- `Dockerfile` builds the runtime image; `.github/workflows/publish-container.yml` pushes it to
  `ghcr.io/landmanbester/kremetart` (version tag on release tags, `latest` on `main`, per-branch on PRs).
- Release with `uv run tbump <version>`: it regenerates `CHANGELOG.md`, refreshes `uv.lock`, then tags
  and pushes. The tag triggers the PyPI and container publishes.
