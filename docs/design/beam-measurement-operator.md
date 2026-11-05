# Airy Beam in the Measurement Operator — Design

**Date:** 2026-06-23
**Status:** Approved (brainstorming) — ready for implementation
**Scope:** Pre-compute the per-frame antenna boresight during the host reprojection
stage, carry it through the streaming imaging zarr into the GPU HEALPix imager, and
apply the analytic Airy primary beam (`kremetart.utils.beam.airy_power_beam`) inside
the measurement operator so the per-frame dirty maps are beam-aware (oriented toward
the intrinsic sky rather than the apparent sky).

---

## 1. Purpose

The GPS patch antennas are zenith-pointing with a broad, frequency-dependent primary
beam approximated by the Airy pattern of the 125 mm ground-plane aperture
(`airy_power_beam`, added in the prior task). The streaming HEALPix imager currently
treats the visibilities as samples of the *apparent* sky `B(t) ⊙ sky` but never models
`B`, so the dirty maps mix the true sky with the (time-varying, sidereally-rotating)
beam.

This change wires `B` into the measurement operator `R = A_dft ∘ diag(B)` so the
imager forms `Rᴴ W vis = B(t) ⊙ (A_dftᴴ W vis)` per frame. That down-weights low-gain
directions, zeroes the unobservable back hemisphere, and is the correct foundation for
forward-modelling / deconvolution toward the intrinsic sky.

---

## 2. Key decisions (from brainstorming)

1. **Store the generator, not the beam.** The per-frame **boresight unit vector**
   (instantaneous zenith in the ICRS grid frame) is stored as a data_var
   `BORESIGHT (time, xyz)` — the exact analog of the existing `B_ROT`. The beam itself
   (`(nchan, npix)`, large) is **computed on the GPU per frame** from
   `airy_power_beam(pix_vec, boresight, freqs)`. Rationale: the beam is `n_time × nchan
   × npix` (hundreds of MB to ~GB), recomputing it on-GPU is negligible against the DFT,
   and `prepare_msv4_zarr` then needs no `nside`. This mirrors the codebase's host/device
   split (small per-frame astropy bookkeeping on the host; heavy per-pixel math on the
   GPU).
2. **Beam-weighted adjoint** (measurement-operator semantics), *not* primary-beam
   division. Forward multiplies the sky by `B`; the per-frame dirty map becomes
   `B ⊙ (Aᴴ W vis / Σ W)`. Division (PB-correction) is rejected: it amplifies noise at
   the horizon and divides by zero in the back hemisphere.
3. **Imaging path only.** Modify `prepare_msv4_zarr` (precompute), the
   `HealpixZarrReaderOperator` (carry), and `HealpixDFTOperator` / `healpix_dft` (apply).
   `rephase_to_dir` (the tart2ms-comparison DataTree utility) is left unchanged — it does
   not feed the DFT operators.
4. **On by default, CLI-configurable.** Add `--apply-beam/--no-apply-beam` (default on)
   and `--ground-plane-diameter` (default 0.125 m) to the `smoovie` CLI; the cab
   regenerates automatically.

---

## 3. Frame / geometry

The HEALPix grid (`make_pixel_grid`) lives in the equatorial **ICRS** frame, and
`b_rot = b_itrs @ R(t)` satisfies `b_rot · s_icrs = b_itrs · s_itrs` (verified: the
physical geometric delay). Therefore the beam boresight must be the **instantaneous
zenith expressed as an ICRS unit vector**: `cos θ = pix_vec · boresight(t)` is then the
true zenith angle of each pixel, and the beam peaks (value 1) at the zenith pixel.

`boresight(t)` = `SkyCoord(AltAz(alt=90°, az=0°, obstime=t, location=site)).icrs` →
`[cos δ cos α, cos δ sin α, sin δ]`.

---

## 4. Components & changes

### 4.1 `utils/healpix_dft.py`
- **New host helper** `zenith_icrs_vectors(times, lat_deg, lon_deg, alt_m) -> (n_time, 3)`:
  instantaneous-zenith ICRS unit vectors (astropy, lazy import). Lives here next to
  `equatorial_baselines` as frame geometry; placed here (not `rephasing.py`) to avoid the
  `read_tart_hdf → rephasing` import cycle.
- **Thread optional `beam=None`** through `dft_forward`, `dft_adjoint`, `dirty_map`,
  `image_frame_prerotated`, `image_frame`. `beam=None` reproduces today's behaviour
  exactly (existing tests untouched). With `beam` of shape `(nchan, npix)`:
  - forward: `vis = (kernel * beam[None]) @ image`
  - adjoint: `einsum("rcj,rc,cj->j", kernel, vis, beam)` — beam applied per-channel
    *before* the channel sum. Conjugate kernel + the same real `beam` ⇒ the exact
    Hermitian transpose is preserved.
  - `dirty_map`: `dft_adjoint(weights*vis, ..., beam=beam).real / weights.sum()` — scalar
    normalization unchanged (see §6).

### 4.2 `utils/read_tart_hdf.py::prepare_msv4_zarr`
- For each file compute `BORESIGHT` via `zenith_icrs_vectors` from the site `info`
  (`site_latitude_deg` / `site_longitude_deg` / `site_altitude_m`), concatenate across
  files exactly like `B_ROT`, and write it as `("time", "xyz")` (float64). No `nside`
  parameter is added.

### 4.3 `operators/io.py::HealpixZarrReaderOperator`
- Add a `BORESIGHT` output port; emit `s["BORESIGHT"].values` (`(1, 3)`) per frame.

### 4.4 `operators/dft_healpix.py::HealpixDFTOperator`
- New ctor args `apply_beam: bool = True`, `ground_plane_diameter: float = 0.125`.
- Add a `BORESIGHT` input port (always received to keep the port from stalling).
- In `compute`, when `apply_beam`, build
  `beam = airy_power_beam(self.pix_vec, boresight, self.freqs, diameter=self.ground_plane_diameter, xp=cp)`
  and pass `image_frame_prerotated(..., beam=beam)`. When off, `beam=None`.

### 4.5 `core/smoovie.py` + `cli/smoovie.py`
- Thread `apply_beam` and `ground_plane_diameter` through `smoovie` → `image_via_app`
  → `SmooviePipeline` → `HealpixDFTOperator`.
- Add `("BORESIGHT", "BORESIGHT")` to the reader→imager `add_flow` set.
- Add the two parameters to the CLI wrapper (`--apply-beam/--no-apply-beam`,
  `--ground-plane-diameter`); the cab regenerates via the pre-commit hook.

---

## 5. Data flow

```
prepare_msv4_zarr (host)            HealpixZarrReader (GPU)      HealpixDFTOperator (GPU)
  B_ROT     (time, bl, xyz)   ──►     emit B_ROT          ──►      beam = airy_power_beam(
  BORESIGHT (time, xyz)  NEW  ──►     emit BORESIGHT  NEW  ──►        pix_vec, boresight, freqs)
  VISIBILITY, WEIGHT, time           emit the rest                 image_frame_prerotated(
                                                                     ..., beam=beam)
```

---

## 6. Normalization & the "intrinsic sky" claim (honest scope)

The per-frame output `B ⊙ (Aᴴ W vis / Σ W)` is the adjoint of the beam measurement
operator — beam-aware and back-hemisphere-zeroed. It is **not**, on its own, a fully
PB-corrected intrinsic-sky image: full recovery requires accumulating frames with a
`Σₜ Bₜ²` (per-pixel) normalization or a deconvolution, which is downstream of the
current per-pixel IWP-Kalman *temporal* filter. This design delivers the correct
measurement-operator wiring and documents the accumulation/normalization step as future
work rather than overclaiming.

---

## 7. Testing (all CPU, `xp=numpy`)

- `healpix_dft`:
  - Beam-threaded adjointness: `<R x, y> == <x, Rᴴ y>` with a non-trivial `(nchan, npix)`
    beam.
  - Beam-weighted `dirty_map`: a forward-modelled single-pixel source peaks at that pixel
    scaled by the beam value there; pixels with `beam == 0` (below horizon) are zero.
  - `beam=None` path leaves existing results byte-identical (regression).
- `zenith_icrs_vectors`: unit norm; a source placed at the returned vector transforms back
  to alt≈90°.
- `prepare_msv4_zarr`: `BORESIGHT` present with dims `("time", "xyz")`, unit-norm rows,
  values equal to `zenith_icrs_vectors`. Update the existing schema test to expect it.
- GPU operators (`HealpixDFTOperator`) are not CPU-unit-tested; their device-pure core
  (`healpix_dft`) is. Existing host-wiring smoovie tests continue to cover the app graph.

---

## 8. Out of scope

- `rephase_to_dir` / the tart2ms-comparison DataTree path.
- `Σₜ Bₜ²` accumulation / deconvolution for full intrinsic-sky reconstruction.
- Polarized / Jones beams (the power beam suffices for the Stokes-I dirty map here).
- Per-antenna or non-circular beam models.
