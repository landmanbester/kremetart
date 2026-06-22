# kremetart

**K**alman **R**eal-time **E**vidence **M**onitoring **E**xtractor for **TART**.

Holoscan-driven applications to image and detect transients in data from the
[Transient Array Radio Telescope (TART)](https://github.com/tart-telescope). The streaming
inference machinery is GPU-resident (CuPy + Holoscan); a CUDA GPU is currently required.

## Installation

```bash
pip install "kremetart[full]"  # full native stack (CuPy/Holoscan, healpy, xarray, tart-tools, …)
```

## Usage

```bash
kremetart --help
kremetart smoovie --help
```

If you have a directory containing pre-downloaded `.hdf` files you should be able to run

```bash
kremetart smoovie --hdf-dir path/to/hdf/dir --nside 64 --correct-gains --overlay-catalog --open-browser --output /tmp/test.zarr
```

Point your browser to `http://localhost:8080/` to watch the output stream. This fetches and overlays the satellite catalog on the fly which might drop frames.

## Producing a movie from the test data

The bundled TART snapshots under `tests/data/` are gitignored and fetched on demand from Google
Drive the first time you run the test suite. You can download them by cloning the repository and installing the test dependencies:

```bash
git clone https://github.com/landmanbester/kremetart.git
cd kremetart
uv sync --group test
```

Download the test data with (set `KREMETART_OFFLINE=1` to skip the download):

```bash
uv run pytest tests/test_smoovie.py -q
```

Then render the snapshots into a HEALPix all-sky movie with `smoovie` which images **one frame per sub-integration**. The bundled data consists of nine ~1-minute snapshots of ~1-second sub-integrations, i.e. **540 frames** — so
render the whole sequence at a watchable frame rate:

```bash
uv run kremetart smoovie --hdf-dir tests/data --output /tmp/test.zarr --nside 64 --correct-gains --overlay-catalog --catalog-cache tests/data/catalog.zarr --open-browser
```

Browser should open automatically, browse to `http://localhost:8080/` if not.

## Running in a container (Not yet ready!)

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
