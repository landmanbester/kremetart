# Architecture

Read this when editing `src/kremetart/**`.

## Layout

| Package | Holds | Imports from |
|---|---|---|
| `cli.py` | The Typer app: one thin function per command, forwarding its options to `core`. | `core` (inside the command, so `kremetart --help` stays fast). |
| `core/` | Host orchestration of a command (validation, data preparation, catalog, web server, profiling) and the **runners** that drive the stages (see Runners). | `imaging`, `utils`; `operators` from `holoscan_runner.py` only. |
| `blocks/` | CyberEther **adapters**: one class per Python block (a stage, the snapshot source, the map-store sink), built from a JSON config inside the CyberEther process. | `imaging`, `utils`. |
| `operators/` | Holoscan **adapters** (receive arrays from ports, build a `Snapshot`, call a stage, emit arrays), plus the zarr reader/writer and the web sink. | `imaging`, `utils`. |
| `imaging/` | The `Snapshot`, the `SkyGrid`, the four **stages** (dirty imager, Tikhonov solver, L1 solver, IWP filter) and the `SnapshotImager` that chains them. All `xp`-injectable. | `opt`, `utils`. |
| `opt/` | Generic solvers (CG, FISTA), `xp`-injectable. | itself. |
| `utils/` | Host helpers: sky geometry, HDF preparation, calibration and gains, satellites, the map store, the web server and frame encoding. | itself. |
| `static/` | The three.js web viewer. | |

Dependencies point one way, down this chain: `cli → core → {operators, blocks} → imaging → {opt, utils}`, with
`core` also using `imaging` and `utils`. A helper that a lower package needs moves down into that package under a
public name (`python-standards.md` §5).

## Stages and adapters

Per-snapshot logic (beam, no-data handling, regularisation scaling, warm starts, Kalman state) belongs
in a stage in `imaging/`, where it is tested on NumPy through the stage's interface. Stages exchange
only plain arrays, and each builds its own DFT kernel; `docs/adr/0001-independent-imaging-stages.md`
records why. A stage takes its array module as an explicit `xp` and keeps its state on that device.

A runtime reaches the stages only through thin adapters: a Holoscan operator today, a CyberEther Python
block after the port. Behaviour changes go in the stage; plumbing changes go in the adapter.

## Runners

A runner drives the stages over a prepared snapshot zarr and delivers every map to the map store
(`utils/map_store.py`) and, when serving, the live viewer (`utils/healpix_viz.publish_maps`). Its
interface is `core.runner.Runner`: `run(prepared, output, settings, holder)`, with `ImagingSettings`
in `core/settings.py`. Adapters:

| Runner (`--runner`) | Module | Picked by `auto` when |
|---|---|---|
| `loop`: `SnapshotImager` in a Python loop, `xp` injectable (NumPy by default) | `core/loop_runner.py` | no usable CUDA device |
| `holoscan`: the Holoscan graph, one operator per stage | `core/holoscan_runner.py` | `core.runner.cuda_available()` |
| `cyberether`: a CyberEther flowgraph, one Python block per stage, in a `cyberether --headless` subprocess | `core/cyberether_runner.py` | never (opt-in); GPU when `cupy_device_available()` |
| `cyberether-gui`: the same flowgraph in a CyberEther window, left open until the user closes it | `core/cyberether_runner.py` (`window=True`) | never (opt-in) |

`core.runner.select_runner(name)` maps the `--runner` choice to an adapter; `core.runner.RUNNERS` is the
list of names, mirrored by `cli.RunnerName`. Which device a runner uses is always detected, never chosen. Every
runner yields the same maps (`tests/test_runners.py`), so output-format and naming changes go in the
shared map store and frame publishing, not in a runner. Only `holoscan_runner.py` imports `holoscan`
or `operators/`; every other `core` module stays importable without a GPU.

### The CyberEther runner

It writes the flowgraph YAML (each block's code is a stub that builds a `blocks/` adapter, laid out left to
right for the window), runs it on this interpreter (`--python-runtime`), and follows the sink's progress
file. `docs/adr/0002-cyberether-as-a-subprocess-flowgraph.md` records why it is a subprocess. CyberEther
behaviours it handles, each recorded where it is handled in `core/cyberether_runner.py`:

- Importing pyarrow, dask or astropy while CyberEther loads block code hangs it, so adapters are built on
  the first compute cycle.
- The AppImages bundle CUDA 12's NVRTC; the runner preloads the newest CUDA 13+ NVRTC it finds (CUDA
  wheels, then `$CUDA_PATH`, `$CUDA_HOME`, `/usr/local/cuda*`).
- A failing block, a rejected flowgraph, or a hang does not stop CyberEther: block stubs record
  tracebacks, and the runner fails on them, on `[ERROR]` log lines, and (headless) when no snapshot lands
  for `$KREMETART_CYBERETHER_STALL_SECONDS` (default 300).
- The graph has no end-of-stream: the source returns `SKIP` when done and the runner ends the process with
  SIGTERM (SIGINT is unreliable) once the sink reports the last snapshot. On Linux the kernel also
  SIGTERMs CyberEther if kremetart dies, so no window is orphaned.

Block stubs can only reach the adapters by import, so a change to a `blocks/` adapter needs no change to
the runner; a change to a block's ports or config needs both (`tests/test_cyberether_runner.py` checks the
wiring without CyberEther).

## Commands

`cli.smoovie` and `core.smoovie.smoovie` take exactly the same parameters (`tests/test_cli.py` checks
this), so a user-facing knob goes on both. Environment differences such as GPU vs CPU are detected
inside the code (`core.runner.cuda_available`); tests force a path by monkeypatching the detector.

## Packaging and releases

- One dependency list in `pyproject.toml`: `pip install kremetart` installs everything the platform can
  run. Linux, macOS (Apple Silicon, 14+) and Windows x64 are supported; platform markers keep the CUDA 13
  stack (Holoscan, CuPy, cuBLAS) on Linux and healpy off Windows. Every runtime dependency must ship
  wheels for all three: check a new one with
  `uv pip compile --python-platform <x86_64-pc-windows-msvc|aarch64-apple-darwin> --only-binary :all:`.
- HEALPix geometry comes from astropy-healpix (`utils/healpix_geometry.py`), which has wheels everywhere;
  healpy has none for Windows and is only used by the parked Mollweide renderer and as a test oracle.
  Measurement Set readers (`arcae`, `xarray-ms`) are test-only for the same reason.
- `Dockerfile` builds the runtime image; `.github/workflows/publish-container.yml` pushes it to
  `ghcr.io/landmanbester/kremetart` (version tag on release tags, `latest` on `main`, per-branch on PRs).
- Release with `uv run tbump <version>`: it regenerates `CHANGELOG.md`, refreshes `uv.lock`, then tags
  and pushes. The tag triggers the PyPI and container publishes.
