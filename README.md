# kremetart

**K**alman **R**eal-time **E**vidence **M**onitoring **E**xtractor for **TART**.

Holoscan-driven applications to image — and, in time, detect transients in — data from the
[Transient Array Radio Telescope (TART)](https://github.com/tart-telescope). The streaming
inference machinery is GPU-resident (CuPy + Holoscan); a CUDA GPU is highly recommended.

## Installation

```bash
pip install kremetart          # lightweight: hip-cargo + typer only
pip install "kremetart[full]"  # full native stack (CuPy/Holoscan, healpy, xarray, tart-tools, …)
```

The **lightweight** install is enough to invoke any command: when the heavy native deps are absent,
the CLI transparently dispatches the command into the project's container image. The **full**
install runs everything natively.

## Usage

```bash
kremetart --help
```

## Producing a movie from the test data

The bundled TART snapshots under `tests/data/` are gitignored and fetched on demand from Google
Drive the first time you run the test suite. Populate them once (set `KREMETART_OFFLINE=1` to skip
the download):

```bash
uv run pytest tests/test_smoovie.py -q
```

Then render the snapshots into a HEALPix all-sky movie with `smoovie` (needs a CUDA GPU and
[`ffmpeg`](https://ffmpeg.org/) on `PATH`). `smoovie` images **one frame per sub-integration** —
the bundled data is nine ~1-minute snapshots of ~1-second sub-integrations, i.e. **540 frames** — so
render the whole sequence at a watchable frame rate:

```bash
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 --nside 64 --fps 12 --correct-gains --overlay-catalog
```

Rendering dominates the runtime (~0.8 s/frame for the three movies), so the full 540-frame render
takes a few minutes. For a quick look, cap the number of frames with `--nframes` — but note this
caps the *total* imaged sub-integrations, and only a few seconds of sky barely moves, so use a
generous value (e.g. the first ~2 minutes of sky):

```bash
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/preview.mp4 --nside 64 --fps 12 --nframes 120 --correct-gains --overlay-catalog
```

Either command writes the dirty all-sky movie (`/tmp/tart.mp4`) alongside `*.filtered.mp4` (the IWP
filtered flux), `*.znorm.mp4` (the normalised innovation), and a durable `*.zarr` holding the
`dirty`/`filtered`/`znorm` `(TIME, PIX)` maps. `smoovie` refuses to overwrite an existing
`<movie>.zarr` — pass `--overwrite` to replace it. It works on any directory of TART `*.hdf`
snapshots; see `kremetart smoovie --help` for all options.
