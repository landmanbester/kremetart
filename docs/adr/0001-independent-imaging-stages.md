# Imaging stages are independent and exchange only plain arrays

The snapshot imager is split into four stages: dirty imager, Tikhonov solver, L1 solver and IWP filter.
Each holds all of its own per-snapshot logic and state. Stages exchange only plain arrays: snapshot fields
and maps. They never pass callables, solver objects or cached operators to each other. This lets each
stage map one-to-one onto a Holoscan operator today, and onto a CyberEther Python block (and later a
native CyberEther module) after the port, without changing the stage. CyberEther blocks can only pass
tensors between them.

The consequence that looks odd is that the dirty imager, Tikhonov solver and L1 solver each build the same
`(baseline, channel, pixel)` DFT kernel for every snapshot, rather than building it once and sharing it.
This is deliberate.

## Considered options

- **Pass the kernel downstream as a tensor.** It is zero-copy on GPU, but it puts a 200–400 MB array and
  its layout into every stage's interface.
- **Merge the dirty imager and both solvers into one deconvolution stage** that shares the kernel
  internally. Rejected because it gives up being able to port, profile and replace stages one at a time.

Measured cost of the rebuild on CPU (NumPy, 276 baselines, one channel): about 0.1 s per stage at nside 32
and about 0.37 s at nside 64. On GPU it is a few milliseconds. If profiling on the DGX Spark or on CPU shows
that it matters, cache inside a stage. Do not share across stages.
