# kremetart

**K**alman **R**eal-time **E**vidence **M**onitoring **E**xtractor for **TART**.

Holoscan-driven applications to image — and, in time, detect transients in — data from the
[Transient Array Radio Telescope (TART)](https://github.com/tart-telescope). The streaming
inference machinery is GPU-resident (CuPy + Holoscan); **a CUDA GPU is highly recommended**, and
required for the imaging app.

> Status: early / pathfinder. Imaging and the per-pixel quiescent filter are in place; the full
> calibration EKF and the sequential transient detector described in the design note are still to
> come. See `docs/tex/unified/kremetart_design.tex` for the complete design.

## What it can currently do

`kremetart` is a [`hip-cargo`](https://github.com/landmanbester/hip-cargo) package: a Typer CLI
whose commands double as [Stimela](https://github.com/caracal-pipeline/stimela) cab definitions, so
the same command runs from the shell or from a Stimela recipe. Two commands are available today:

| Command | What it does |
|---|---|
| `kremetart smoovie` | Render a sequence of TART HDF snapshots into a HEALPix all-sky movie, and run a per-pixel **IWP–Kalman whitening filter** over the resulting light curves. |
| `kremetart stream-msv4` | Stream an MSv4 Measurement Set into a Zarr dataset. |

```bash
kremetart --help
kremetart smoovie --help
```

### `smoovie` in detail

`smoovie` concatenates the HDF snapshots in a directory into a single MSv4 Zarr, then streams it
through a GPU Holoscan app:

```
reader → imager (HEALPix DFT) → IWP–Kalman filter → writer
```

Every sub-integration is imaged into a *fixed equatorial (ICRS) HEALPix grid* with a common phase
centre, so each pixel carries a sidereally-fixed light curve. A per-pixel **q=1 integrated
Wiener-process (IWP) Kalman filter** then whitens each pixel's light curve frame by frame (the
quiescent model of the design note, §sec:iwp/§sec:kf): it tracks the slowly-varying flux and emits

- the **filtered flux** `x_{k|k}[0]` — a denoised version of the dirty map, and
- the **normalised innovation** `z_k = e_k / √S_k` — the whitened residual, which is ≈ 𝒩(0,1) per
  pixel under the quiescent model and is the precursor to the transient detector.

The filter honours irregular cadence exactly: the transition/process-noise matrices are rebuilt
from the per-frame Δt of the timestamp stream every frame (no fixed-step assumption).

**Outputs** (for `--movie out.mp4`):

| Path | Contents |
|---|---|
| `out.mp4` | Mollweide movie of the **dirty** maps (fixed colour scale). |
| `out.mp4.filtered.mp4` | Movie of the IWP **filtered flux**. |
| `out.mp4.znorm.mp4` | Movie of the **normalised innovation** `z_k` (diverging scale, centred on 0). |
| `out.mp4.zarr` | Durable `(TIME, PIX)` Zarr holding the `dirty`, `filtered` and `znorm` HEALPix maps, left in place for inspection. |

## Installation

`kremetart` has two install modes:

```bash
pip install kremetart          # lightweight: hip-cargo + typer only
pip install "kremetart[full]"  # full native stack (CuPy/Holoscan, healpy, xarray, tart-tools, …)
```

The **lightweight** install is enough to *invoke* any command: when the heavy native deps are
absent, the CLI transparently dispatches the command into the project's container image (needs a
container runtime; `smoovie` still needs a CUDA GPU). The **full** install runs everything natively.

For `smoovie` you also need [`ffmpeg`](https://ffmpeg.org/) on `PATH` (for the mp4 encode) and, for
native execution, a CUDA GPU with the CuPy/Holoscan stack from `[full]`.

### From a source checkout (development)

```bash
uv sync --extra full     # create the venv with the full stack
uv run kremetart --help
```

## Test data

The TART snapshots live under `tests/data/` (nine `vis_*.hdf` files, matching tart2ms Measurement
Sets, and a bundled satellite-catalogue cache `catalog.zarr`). They are **gitignored** and fetched
on demand from Google Drive the first time you run the test suite:

```bash
uv run pytest tests/test_smoovie.py -q   # downloads tests/data/ on session start (via gdown)
```

Set `KREMETART_OFFLINE=1` to skip the download (air-gapped runs). `smoovie` works on *any* directory
of TART `*.hdf` snapshots, not just `tests/data/`.

## Running `smoovie` on the test data

Once `tests/data/` is populated, render the nine bundled snapshots from a source checkout:

```bash
# Basic: image + IWP-filter all sub-integrations into a HEALPix movie at nside=64.
uv run kremetart smoovie \
    --hdf-dir tests/data \
    --movie /tmp/tart.mp4 \
    --nside 64 \
    --fps 2

# Produces /tmp/tart.mp4 (dirty), /tmp/tart.mp4.filtered.mp4 (IWP filtered flux),
#          /tmp/tart.mp4.znorm.mp4 (normalised innovation), and /tmp/tart.mp4.zarr.
```

A few useful variations:

```bash
# Quick preview: cap the number of frames, and apply the per-antenna gain solution before imaging.
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 \
    --nside 64 --nframes 4 --correct-gains --overwrite

# Overlay catalogued satellite tracks, reusing the bundled cache (built at 45° elevation,
# so no network call is made):
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 \
    --nside 64 --overlay-catalog \
    --catalog-cache tests/data/catalog.zarr --catalog-elevation-deg 45

# Tune the IWP filter: --iwp-sigma is the driving variance σ² (smaller = stiffer / slower to
# follow changes); --iwp-noise is the measurement-noise variance R.
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 \
    --nside 64 --iwp-sigma 1e-4 --iwp-noise 1e-2 --overwrite
```

Notes:

- `smoovie` refuses to clobber an existing `<movie>.zarr` — pass `--overwrite` to replace it.
- Inspect the per-pixel maps directly: `xarray.open_zarr("/tmp/tart.mp4.zarr")` gives a
  `(TIME, PIX)` dataset with `dirty`, `filtered` and `znorm` variables.
- `--phase-ra-deg` / `--phase-dec-deg` override the common phase centre (default: the zenith
  RA/Dec at the global mid-time); `--profile` prints a per-stage timing summary.

## `stream-msv4`

Stream a Measurement Set (MSv4) into a Zarr dataset:

```bash
uv run kremetart stream-msv4 --ms tests/data/vis_2026-06-09_08_11_43.476804.ms --output-dataset /tmp/vis.zarr
```

## Use from Stimela

Each command ships a generated cab under `src/kremetart/cabs/` (e.g. `smoovie.yml`), so the same
commands can be driven from a Stimela recipe with the same inputs/outputs. The cabs are
auto-generated from the CLI source — do not edit them by hand.

## Links

- TART telescope: <https://github.com/tart-telescope>
- hip-cargo (CLI ↔ cab tooling): <https://github.com/landmanbester/hip-cargo>
- Stimela: <https://github.com/caracal-pipeline/stimela>
- Design note: `docs/tex/unified/kremetart_design.tex`
