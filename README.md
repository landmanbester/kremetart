# kremetart

**K**alman **R**eal-time **E**vidence **M**onitoring **E**xtractor for **TART**.

Applications to image and detect transients in data from the
[Transient Array Radio Telescope (TART)](https://github.com/tart-telescope). On a machine with a CUDA
GPU the imaging streams through Holoscan (CuPy); without one it falls back to an in-process NumPy loop,
which gives the same maps but is much slower. Deconvolution is opt-in for that reason: pass `--l2 0.01`
(Tikhonov) and/or `--l1 0.01` (reweighted L1) to enable it; by default the Kalman filter runs on the
dirty maps.

## Installation

```bash
pip install kremetart
```

kremetart needs Python 3.11 or later and runs on Linux, macOS (Apple Silicon) and Windows. What it can run on depends on the platform:

| Platform | Runners | Notes |
|---|---|---|
| Linux x86-64 and arm64 (e.g. DGX Spark) | `loop`, `holoscan`, `cyberether` | `pip install` also brings the CUDA 13 stack (Holoscan, CuPy, cuBLAS); the GPU is used when present. |
| macOS 14+, Apple Silicon | `loop`, `cyberether` | CPU only: no CUDA on macOS. |
| Windows x64 | `loop`, `cyberether` | CPU only: CyberEther has no CUDA build for Windows. |

`--runner auto` (the default) picks Holoscan on a Linux CUDA machine and the NumPy loop everywhere else.

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

`--runner` picks what drives the imaging. `auto` (the default) uses Holoscan on a CUDA machine and the
NumPy loop otherwise; `loop` and `holoscan` force one of those; `cyberether` and `cyberether-gui` are
described in [Running through CyberEther](#running-through-cyberether). Every runner produces the same maps.

Holoscan recommends a stack size of at least 32 MB and warns (and may crash on deep call stacks) below that. Most Linux shells default to 8 MB, so raise the limit in the shell before running natively:

```bash
ulimit -s 32768
```

This lasts for the current shell session, so add it to your shell profile to make it permanent. Apptainer/Singularity containers launched from that shell inherit the limit.

## Trying it on the test data

The bundled TART data under `tests/data/` is gitignored and fetched on demand from Google Drive the first
time you run the test suite. Clone the repository and install the test dependencies:

```bash
git clone https://github.com/landmanbester/kremetart.git
cd kremetart
uv sync --group test
```

Then download the data by running a test (set `KREMETART_OFFLINE=1` to skip the download):

```bash
uv run pytest tests/test_smoovie.py -q
```

The bundle holds nine ~1-minute HDF files of ~1-second snapshots, i.e. 540 snapshots. `smoovie` images
each one and streams the maps to the live viewer:

```bash
uv run kremetart smoovie --hdf-dir tests/data --output /tmp/test.zarr --nside 64 --correct-gains --overlay-catalog --catalog-cache tests/data/catalog.zarr --open-browser
```

The browser should open on its own; if not, browse to `http://localhost:8080/`.

## Running through CyberEther

`--runner cyberether` runs the stages as blocks of a headless
[CyberEther](https://github.com/luigifcruz/CyberEther) flowgraph, on the GPU when CuPy sees one. You need
CyberEther 1.19 or later. On a GPU machine use a CUDA build (`CyberEther-linux-x64-cuda.AppImage`, or
`CyberEther-linux-arm64-cuda.AppImage` on the DGX Spark): the plain `CyberEther-linux-x64.AppImage` has no
CUDA backend and only suits CPU-only machines. Make the AppImage executable and point
`KREMETART_CYBERETHER` at it (or put it on `PATH` as `cyberether`), adjusting the path to wherever you
saved it:

```bash
chmod +x ~/Downloads/CyberEther-linux-x64-cuda.AppImage
export KREMETART_CYBERETHER=~/Downloads/CyberEther-linux-x64-cuda.AppImage
kremetart smoovie --hdf-dir path/to/hdf/dir --output /tmp/test.zarr --runner cyberether
```

kremetart writes the flowgraph, starts CyberEther on its own Python (so the blocks import kremetart from
the same environment), and stops it after the last snapshot. Each run spends about 10 s starting up. The
live web viewer works as with any other runner.

**To see the flowgraph inside CyberEther**, use `--runner cyberether-gui`. It runs the same flowgraph in a
CyberEther window instead of headless, so you can watch the blocks execute:

```bash
kremetart smoovie --hdf-dir path/to/hdf/dir --output /tmp/test.zarr --nside 32 --l2 0.01 --runner cyberether-gui
```

The window shows a note explaining the graph and one Python block per stage, left to right: `source`
(the prepared snapshots), `imager` (dirty map), `tikhonov` and `l1` (only when `--l2`/`--l1` are given),
`iwp` (the Kalman filter) and `sink` (the map store); pin tooltips show each tensor's attributes. Each
block's code is a short stub that builds the stage adapter from `kremetart.blocks`; the maths itself is in
`kremetart.imaging`. Editing the stub in the window reloads that block. When every snapshot is imaged,
kremetart prints `close the CyberEther window to finish`; close the window to let `smoovie` write its
profile and, when serving, freeze the web viewer.

Troubleshooting:

- **`CyberEther not found`**: set `KREMETART_CYBERETHER` to the AppImage.
- **A block's traceback**: kremetart stops CyberEther and raises the error from the failing block.
- **`CyberEther wrote no snapshot for 300 s`**: CyberEther hung, or a snapshot really takes that long (L1 at
  high nside on the CPU); raise `KREMETART_CYBERETHER_STALL_SECONDS` for the latter.
- **CUDA errors such as `cuda_fp6.hpp: expected a ";"`**: the AppImage's bundled CUDA 12 compiler is in use.
  kremetart preloads CUDA 13's NVRTC from your CuPy wheels or CUDA toolkit; if it finds neither, install
  CUDA 13 or set `CUDA_PATH`.

## Running in a container

The published image (`ghcr.io/landmanbester/kremetart`, x86-64) carries the whole Linux stack, CUDA
included. Started with GPU access it images on the GPU (Holoscan); without it, it falls back to the CPU
loop. GPU access needs an NVIDIA GPU, recent drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

A batch run, writing the map store to `./out`:

```bash
mkdir -p out
docker run --rm --gpus all --ulimit stack=33554432 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/tests/data:/data:ro" -v "$PWD/out:/out" \
  ghcr.io/landmanbester/kremetart:latest \
  kremetart smoovie --hdf-dir /data --output /out/tart.zarr --nside 64 --correct-gains --no-serve
```

- **`--gpus all`**: expose the host GPU. Without it the run prints `imaging on CPU (NumPy loop)`.
- **`--ulimit stack=33554432`**: a 32 MiB thread stack. Holoscan warns at import and can segfault below it.
- **`--user … -e HOME=/tmp`**: write outputs as you rather than `root`, and give CuPy a writable cache
  directory.
- **`-v …`**: mount the HDF files read-only and a directory to receive the `.zarr`.

To watch the live viewer, drop `--no-serve` and run on the host's network (`--network host` instead of
`-p 8080:8080`, because the viewer listens on localhost only), then browse to `http://localhost:8080/`.
Host networking also gives `--overlay-catalog` the access it needs to the TART catalogue API.

`:latest` tracks `main`; releases are tagged by version (e.g. `:0.1.0`). Under Apptainer/Singularity the
GPU flag is `--nv` and the host network is the default:

```bash
apptainer exec --nv docker://ghcr.io/landmanbester/kremetart:latest kremetart smoovie --hdf-dir … --output …
```
