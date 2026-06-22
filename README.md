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

## Running in a container (GPU)

If you'd rather not install the full native stack, run any command straight from the published
image. The imaging pipeline is GPU-resident (CuPy + Holoscan) with no CPU fallback, so the
container **must** be launched with GPU access. This requires the host to have an NVIDIA GPU, recent
drivers, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed:

```bash
mkdir -p out
docker run --rm --gpus all --ulimit stack=33554432 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/tests/data:/data:ro" -v "$PWD/out:/out" \
  ghcr.io/landmanbester/kremetart:0.0.1 \
  kremetart smoovie --hdf-dir /data --movie /out/tart.mp4 --nside 64 --fps 12 --correct-gains
```

The non-obvious flags:

- **`--gpus all`** — expose the host GPU(s) to the container; needs the NVIDIA Container Toolkit. The
  pipeline aborts with *"no CUDA-capable device is detected"* without it.
- **`--ulimit stack=33554432`** — give threads a 32 MiB stack. Holoscan warns at import and can
  segfault below this.
- **`--user … -e HOME=/tmp`** — write outputs as you rather than `root`, and give CuPy/matplotlib a
  writable directory for their on-disk caches.
- **`-v …`** — mount the input HDF snapshots read-only and a directory to receive the `.mp4`/`.zarr`.

Swap `:0.0.1` for `:latest` to track `main`. Under Apptainer/Singularity the GPU flag is `--nv`
instead (`apptainer exec --nv docker://ghcr.io/landmanbester/kremetart:0.0.1 kremetart …`). Add
`--overlay-catalog` to overlay the TART satellite catalogue on each frame — it needs outbound
network access to the TART catalogue API, which the default Docker bridge network provides.
