# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-06

### CI

- Move the macOS job to macos-15; GitHub retired macos-14

### Documentation

- Show group options, argument shape and README reference links
- Sync the docs generators with the autodocs skill
- Say the smooth map's unit follows the map it smooths
- Link the Read the Docs site from the README and package metadata
- Tabulate options and glossary; calmer colours; keep agent records out
- Render the command options as tables
- Generate Sphinx reference docs from docstrings, help strings and records
- Name releases after African trees; drop the closed #47 reference


## [0.1.0] - 2026-11-05

### Added

- Support macOS and Windows for the loop and CyberEther runners
- Add a cyberether-gui runner, a stall timeout, and the docs
- Add a --runner option to pick loop, holoscan or cyberether
- Add a CyberEther runner
- Run smoovie on CPU behind a runner seam; make L2 opt-in
- Add unit-labelled colorbar to the live viewer
- Always image dirty/Tikhonov/L1; --l2/--l1 strengths
- Constant 5-channel viewer set with per-channel units
- Add Tikhonov-vs-L1 deconvolution comparison script
- Select tikhonov or l1 deconvolution via --regulariser
- Add L1ReweightOperator (FISTA deconvolution sibling of Tikhonov)
- Add fista_quadratic image-space reweighted-L1 solver
- Fista deconv function
- Export fista and add sky-model recovery test
- Add reweighted-L1 outer loop to FISTA
- Add inner FISTA with backtracking line search
- Add weighted-L1 soft-threshold prox
- T_int solution interval (time-pooled StefCAL)
- Optional primary-beam weighting in model visibilities
- Add catalogue glue + opt-in StefCAL real-data check
- Add acquisition StefCAL solver with gauge fixing
- Add unit-flux ENU sky model for calibration
- Add (beam corrected) Tikhonov regularised stream
- Airy beam in measurement operator
- Vendor renderer with live stream, freeze, and satellite overlay
- Rewrite smoovie CLI for live web viewer (serve/port/output)
- Stream smoovie maps to web sink; drop png/ffmpeg path
- Add WebStreamSinkOperator terminal sink
- Add FrameServer and /stream websocket handler
- Add geometry and tracks payload builders for web viz
- Add LatestFrameHolder and frame encoder for web viz
- Split out visualisation module for future sub-command that will render cude to old Mollwiede projection

### CI

- Publish the container for arm64 (DGX Spark) as well as amd64
- Test the matrix Python; probe CyberEther Python blocks on Windows
- Fail stalled CyberEther runs after 120 s
- Register Windows' software Vulkan driver in the registry
- Give Windows runners Mesa's software Vulkan for CyberEther
- Extract CyberEther with relative paths so Windows tar works
- Bump the github-actions group across 1 directory with 3 updates

### Changed

- Make kremetart a single command, dropping the smoovie subcommand
- Remove the pre-stage imaging path and the tart2ms study
- Retire hip-cargo packaging
- Delete the stream_msv4 lm-plane imaging lineage
- Move per-snapshot imaging logic into four stage modules
- Split device-side HEALPix imaging maths from host geometry
- Make reweighted-L1 (FISTA) deconvolution opt-in
- Read TART HDF files through tart-msv4
- Drive fista on shared value/value_grad smooth-term core

### Dependencies

- Drop the Mollweide renderer and unused dependencies
- Bump the python-minor-patch group across 1 directory with 8 updates
- Bump the python-minor-patch group across 1 directory with 4 updates
- Update uv-build requirement
- Bump the python-minor-patch group across 1 directory with 2 updates

### Documentation

- Update the README, agent rules and design notes for 0.1.0
- Tell native users to raise the stack size for Holoscan
- Delete stale calimage and detector documents now in unified
- Align operator/spec docs with parallel-deconvolver data flow
- Correct L1ReweightOperator port docstring after parallel rewiring
- Implementation plan for three-stream comparison viewer
- Design spec for three-stream comparison viewer
- Reconcile fista_quadratic default (max_reweight=0) and b==0 short-circuit
- Implementation plan for reweighted-L1 deconvolution
- Design spec for reweighted-L1 deconvolution operator
- Add reweighted-L1 FISTA implementation plan
- Add reweighted-L1 FISTA solver design spec
- Implementation plan for StefCAL calibration core
- Design for StefCAL calibration core (acquisition)
- Design for Tikhonov-regularised imaging via conjugate gradients
- Design for Airy beam in the measurement operator
- Update README
- Record localhost-default server bind in web visualizer spec
- Add live web visualizer implementation plan
- Note parked Mollweide renderer in web visualizer spec
- Add live web visualizer design spec for smoovie

### Fixed

- Read and write text as UTF-8; show CyberEther's log on errors
- Check the CyberEther executable before any work starts
- Find NVRTC in CUDA toolkits as well as CUDA wheels
- Label smoothed dirty maps in dirty units; comment GPU imports
- Address review on the imaging stages
- **fista**: Move w into the sum
- Compare dirty concentration on the same clipped basis as tikhonov/l1
- Clamp negative max_reweight in fista to plain L1
- GPU = True in _container_image.py so that containerised backends can see the GPU
- Finalise symlinks at build time by by adding RUN python -c 'import holoscan' to Dockerfile
- Harden IWP evolution when frames are fully flagged
- Harden stream_handler against name mismatch; correct overlay docstrings
- Bind web server to localhost by default (no LAN exposure)
- Flush clean end on imaging error so clients freeze not reconnect
- Stop web server on imaging error; clarify serve in catalog tests
- Explicit tracks None-check and type hints on stream_handler

### Miscellaneous

- Configure agent skills (issue tracker, triage labels, domain docs)
- Pin the pre-commit ruff hook to the locked ruff (0.16.6)
- Respond to copilot comments
- Add StefCAL gain calibration/validation scripts
- Merge in main
- Update container tag

### Other

- Bump hip-cargo version
- Update uv.lock for fastapi and uvicorn
- Get Dockerfile to install system dependencies

### Testing

- Strip colour from the help output before checking it
- Check the Kalman covariance stays symmetric PSD in the IWP loop
- Keep one test per behaviour at the interface
- Break the CyberEther source block in a platform-independent way
- Run the CyberEther CUDA case wherever CuPy sees a device
- Make the CLI tests GPU-free and check every forwarded option
- Module-level imports and type hints in new tests
- Cover complex-operator gradient path in FISTA
- Real-data check uses beam + pooled t_int; sync spec with findings
- Cover web fan-out and HTTP/WS serving end-to-end
- Relocate Mollweide overlay tests to utils.visualisation
- Assert geometry corners are native floats and cover second track point
- Derive thread-safety max and cover empty-frame encoding


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


[0.1.1]: https://github.com/landmanbester/kremetart/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/landmanbester/kremetart/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/landmanbester/kremetart/releases/tag/v0.0.1

