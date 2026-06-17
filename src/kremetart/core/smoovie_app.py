"""GPU Holoscan smoovie imaging app: a prepared zarr -> a streamed HEALPix dirty-map zarr.

Mirrors :mod:`kremetart.core.stream_msv4`. All host work (gain correction, the per-frame ``b_rot(t)``
rotation, the catalogue) is done by the prepare-step (:mod:`kremetart.core.smoovie_prepare`) and the
catalogue cache; this module is the pure-GPU imaging backbone: reader -> HealpixDFTOperator ->
writer. Movie rendering/encoding happens on the host *after* ``app.run()`` (see :func:`image_via_app`).

``holoscan`` and ``cupy`` import at module top, so importing this module requires a GPU. The CPU
``smoovie`` path in :mod:`kremetart.core.smoovie` never imports it -- it is imported lazily, only when
:func:`kremetart.core.smoovie._gpu_imaging_available` is true.
"""

from pathlib import Path

import holoscan as hs
import numpy as np
import xarray as xr
from holoscan.conditions import CountCondition

from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator


class SmooviePipeline(hs.core.Application):
    """Stream a prepared imaging zarr through the GPU HEALPix imager into a ``(TIME, npix)`` zarr."""

    def __init__(self, prepared_zarr, output_zarr, nside, *args, nest=True, **kwargs):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
        super().__init__(*args, **kwargs)

        import healpy as hp

        ds = xr.open_zarr(self.prepared_zarr)
        self.ntime = int(ds.time.size)
        self.out_times = ds.time.values
        self.freqs = ds.frequency.values
        self.npix = hp.nside2npix(nside)

    def compose(self):
        reader = HealpixZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.prepared_zarr,
        )
        imager = HealpixDFTOperator(self, self.nside, self.freqs, name="imager", nest=self.nest)
        writer = HealpixWriterOperator(
            self,
            self.ntime,
            self.npix,
            name="writer",
            output_dataset=self.output_zarr,
            out_times=self.out_times,
        )
        self.add_flow(
            reader,
            imager,
            {("VISIBILITY", "VISIBILITY"), ("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("time", "time")},
        )
        self.add_flow(imager, writer, {("cube", "cube"), ("time_out", "time_out")})


def image_via_app(
    hdf_paths,
    nside,
    *,
    correct_gains=False,
    phase_ra_deg=None,
    phase_dec_deg=None,
    nframes=None,
    nest=True,
):
    """Image the HDF sequence through the GPU app; return ``(maps, stamps)``.

    Drop-in for the imaging half of :func:`kremetart.core.smoovie.frame_dirty_maps`: returns a list
    of ``(npix,)`` dirty maps (one per frame, in order) and a list of UTC stamp strings. Runs the
    host prepare-step into a temp zarr, streams it through :class:`SmooviePipeline`, then loads the
    ``(TIME, npix)`` output zarr back to host.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        correct_gains: apply inverse per-antenna gains in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on frames.
        nest: NESTED HEALPix ordering (default True; matches the CPU path).

    Returns:
        ``(maps, stamps)``.
    """
    import tempfile

    from kremetart.core.smoovie import _utc
    from kremetart.core.smoovie_prepare import prepare_msv4_zarr

    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        output = Path(td) / "dirty.zarr"
        config = Path(td) / "config.yaml"
        config.touch()  # an empty Holoscan config is valid

        prepare_msv4_zarr(
            hdf_paths,
            prepared,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
        )

        app = SmooviePipeline(prepared, output, nside, nest=nest)
        app.config(str(config))
        app.run()

        ds = xr.open_zarr(str(output))
        dirty = np.asarray(ds["dirty"].values)  # (ntime, npix)
        times = np.asarray(ds["TIME"].values)

    maps = [dirty[i] for i in range(dirty.shape[0])]
    stamps = [_utc(t) for t in times]
    return maps, stamps
