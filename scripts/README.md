# scripts

Host/CPU utilities for validating the StefCAL acquisition calibration against TART's own gain
solutions. None of these need a GPU or Holoscan — they call `kremetart.utils` directly.

| Script | Purpose |
|---|---|
| `stefcal_calibrate.py` | Shared helper. `solve_file_gains()` solves one TART file's StefCAL gains (beam-weighted model, pooled over the whole file). |
| `validate_tart_gains.py` | Image one frame with no-cal / TART / our gains and report the pixel correlation of each against the TART-calibrated image. A high `corr(ours, TART)` means our gains recover the TART-quality image. |
| `overwrite_tart_gains.py` | Solve and write **phase-only** gains (unit amplitude live, 0 for dead) back into a directory of HDFs, so `kremetart smoovie --correct-gains` images with our calibration. |

## Typical workflow

Work on a **copy** of the data (e.g. `tests/data_stefcal`), never the pristine `tests/data` — the
overwrite is in place.

```bash
# 1. Numerically check our gains reproduce the TART-calibrated image:
python scripts/validate_tart_gains.py tests/data_stefcal/vis_2026-06-09_08_11_43.476804.hdf

# 2. Overwrite the copy's gain solutions with ours:
python scripts/overwrite_tart_gains.py tests/data_stefcal

# 3. Image + overlay the catalogue tracks; bright satellites should sit on the tracks:
uv run kremetart smoovie --hdf-dir tests/data_stefcal \
    --output /tmp/view_ours.zarr --overwrite \
    --correct-gains --overlay-catalog \
    --catalog-cache tests/data_stefcal/catalog.zarr --open-browser
```

The `solve_file_gains` logic here is the seed of the future streaming calibration operator; it lives
under `scripts/` for now as a validation aid.
