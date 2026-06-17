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
