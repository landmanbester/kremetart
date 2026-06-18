# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-06-18

### Added

- Expose iwp_sigma/iwp_noise/overwrite on the smoovie CLI
- Wire per-pixel IWP filter into smoovie with durable zarr and 3 movies
- HealpixWriterOperator writes dirty/filtered/znorm variables
- Add per-pixel IWP-Kalman Holoscan operator
- Add xp-generic IWP transition and Kalman recursion kernels
- Make smoovie purely holoscan driven. test: refactor tests to eliminate CPU only path
- Route smoovie imaging through the GPU Holoscan app with CPU fallback
- GPU healpix imager consumes streamed b_rot; add healpix reader/writer ops
- Add host prepare-step writing an imaging-ready smoovie zarr
- Add some progress trackers to smoovie. remove satelite names
- Expose catalog-cache, profile, and nframes on smoovie CLI
- Add per-stage profiling and nframes/catalog-cache wiring to smoovie
- Add nframes cap to frame_dirty_maps
- Cache TART catalogue to a time-indexed zarr dataset
- Expose gain-correction and catalog-overlay on smoovie CLI
- Thread gain-correction and catalog-overlay options through smoovie
- Overlay satellite tracks in smoovie render_frames
- Build ICRS satellite tracks from the TART catalogue
- Apply inverse gains in smoovie frame_dirty_maps
- Add inverse per-antenna gain correction helper
- Expose phase-direction params on smoovie CLI
- Anchor smoovie frames to a common phase direction
- Image every sub-integration in smoovie frame_dirty_maps
- Add common_phase_direction helper to smoovie core
- Add smoovie CLI command, cab, and round-trip test
- Add smoovie rendering, ffmpeg encode, and orchestrator
- Add smoovie core frame_dirty_maps (per-HDF HEALPix imaging)
- Add HealpixDFTOperator Holoscan wrapper
- Add end-to-end HEALPix image_frame (C(t) + dirty map)
- Add astropy C(t) baseline rotation for HEALPix imaging
- Add HEALPix dirty-map convenience (weighted adjoint)
- Add HEALPix DFT forward/adjoint transpose pair
- Add HEALPix pixel-grid (direction cosines) for the DFT imager
- Add rephasing + start of healpix dft operators
- Add read_tart_hdf.py module. Add __init__.py for operators and utils modules. deps: add netcdf4 dependency
- Add stream-msv4 command

### Changed

- Drop unused overwrite arg from image_via_app; document smoovie outputs
- Type-hint smoovie app, eager zarr read-back, test gpu-probe fallback
- Tighten healpix writer typing + document out_times coupling
- Type-annotate prepare_msv4_zarr params and cover nframes file-boundary break
- Split healpix image_frame into device-pure image_frame_prerotated
- Avoid per-satellite full redraws in smoovie overlay
- Promote itrs_baselines to a public shared helper

### Dependencies

- Declare nvidia-cublas (CUDA-13) for cupy GPU imaging kernels
- Declare dask[array] for the streaming writer operators

### Documentation

- Update readme with --correct-gains and --overlay-catalog params
- Fix smoovie movie example (nframes 4 -> full 540-frame render)
- Add instructions to produce a movie from the test data
- Trim README to project aim and install modes
- Document current capabilities and smoovie usage on test data
- Fix stale prepare_msv4_zarr cross-ref in HealpixZarrReaderOperator
- Update HealpixWriterOperator docstring for filtered/znorm outputs
- Note intentional float64 state and in-order-frame assumption in IWP operator
- Implementation plan for per-pixel IWP-Kalman filter in smoovie
- Design spec for per-pixel IWP-Kalman filter in smoovie
- Add smoovie GPU Holoscan (Phase 2) implementation plan
- Add smoovie performance Phase 1 plan (catalog cache + profiling)
- Cache catalog as a time-indexed xarray dataset in smoovie perf spec
- Add smoovie performance + GPU Holoscan design
- Record confirmed tart2ms weighted-corrected DATA convention
- Add smoovie gain-correction and catalog-overlay plan
- Add smoovie gain-correction and catalog-overlay design
- Add reworked smoovie spec and plan docs
- Refresh smoovie module docstring for common-frame behavior
- Commit the smoovie spec and plan
- Fix itrs_baselines reference after rename
- Add HEALPix DFT operator implementation plan
- Update imaging and calibration document
- Start adding design docs and feature specs

### Fixed

- Remove unnecessary importorskips and move most imports to the top of files
- Remove checks for protext from tests as text is no longer plotted

### Miscellaneous

- Update container tag
- Initial project scaffold

### Other

- Update uv.lock for gdown and pyproj test deps
- Remove 3.10 from test matrix and update container tag
- Add full dependencies

### Testing

- GPU e2e for filtered/znorm outputs and overwrite fail-fast
- IWP innovations whiten with mean NIS ~ 1 on synthetic data
- GPU smoovie end-to-end and equivalence to the CPU baseline
- Add opt-in oracle comparing weighted corrected vis to calibrated MS
- Download whole test-data bundle from Google Drive
- Auto-download test data from Google Drive via gdown
- Add steady-source pixel-stability test (sidereally-fixed grid)
- Add L2 sub-pixel imaging accuracy verification vs tart2ms geometry
- Add recovery/offset/analytic metrics for accuracy verification
- Add sky/source + truth-visibility simulation helpers
- Add L1 antenna-position accuracy verification vs PROJ truth
- Add pyproj truth-geometry helpers for accuracy verification


[0.0.1]: https://github.com/landmanbester/kremetart/releases/tag/v0.0.1

