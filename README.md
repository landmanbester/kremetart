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
[`ffmpeg`](https://ffmpeg.org/) on `PATH`):

```bash
uv run kremetart smoovie --hdf-dir tests/data --movie /tmp/tart.mp4 --nside 64 --nframes 4
```

This writes `/tmp/tart.mp4` (the dirty all-sky movie) alongside `tart.mp4.filtered.mp4`,
`tart.mp4.znorm.mp4`, and a durable `tart.mp4.zarr`. `smoovie` refuses to overwrite an existing
`<movie>.zarr` — pass `--overwrite` to replace it. `smoovie` works on any directory of TART `*.hdf`
snapshots; see `kremetart smoovie --help` for all options.
