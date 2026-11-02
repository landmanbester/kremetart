# The CyberEther runner drives a generated flowgraph in a subprocess

The CyberEther runner writes the flowgraph as YAML, launches `cyberether --headless` on kremetart's own
Python interpreter, and follows the run from outside: the map-store sink writes each snapshot and then a
progress count, the runner publishes new rows to the live viewer from the map store, and it ends the
process with SIGTERM once the last snapshot is written. Each Python block's code is a stub that imports a
`kremetart.blocks` adapter on its first compute cycle.

This looks roundabout next to the Holoscan runner, which builds its graph in-process. It is deliberate:
CyberEther 1.19 has no Python API for building or running a flowgraph, so a YAML file and its CLI are the
only stable way in, and a flowgraph has no end-of-stream signal, so completion has to be observed rather
than awaited.

## Considered options

- **Native CyberEther modules (`.cep` plugins).** Faster and in-process, but the plugin ABI is
  experimental and would mean rewriting the stages in C++; revisit once the Python blocks are proven on
  the Spark and a stage is a measured bottleneck.
- **Embed the whole snapshot imager in one Python block.** Simpler wiring, but it gives up one block per
  stage, which ADR-0001 keeps so stages can be ported, profiled and replaced one at a time.
- **Write block code into the YAML.** The stubs stay tiny instead, so the adapters are ordinary,
  importable, testable modules; and importing pyarrow, dask or astropy while CyberEther loads block code
  hangs it, which building on the first compute cycle avoids.
