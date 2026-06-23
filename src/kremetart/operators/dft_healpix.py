"""Holoscan operator: full-sky HEALPix dirty-map imager (GPU-resident, xp=cupy).

Pure cupy: consumes the per-frame equatorial-rotated baselines ``b_rot(t)`` precomputed by the host
prepare-step (:mod:`kremetart.core.smoovie_prepare`), so there is no astropy round-trip inside
``compute()``. Thin wrapper around :func:`kremetart.utils.healpix_dft.image_frame_prerotated`; images
onto a fixed full-sky equatorial HEALPix grid built once on the device.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam
from kremetart.utils.healpix_dft import image_frame_prerotated, make_pixel_grid


class HealpixDFTOperator(Operator):
    """Adjoint (dirty-map) HEALPix DFT imager, pure cupy.

    When ``apply_beam`` is set, the per-frame Airy primary beam is built on the GPU from the
    streamed boresight and folded into the measurement operator, so the dirty map is the
    beam-weighted ``B (.) (A_dft^H W vis)`` oriented toward the intrinsic sky.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        nest: NESTED HEALPix ordering (default True; index locality for the streaming detector).
        apply_beam: apply the Airy primary beam in the measurement operator (default True).
        ground_plane_diameter: Airy aperture (ground plane) diameter in metres.
    """

    def __init__(
        self,
        fragment,
        nside,
        freqs,
        *args,
        nest=True,
        apply_beam=True,
        ground_plane_diameter=GROUND_PLANE_DIAMETER,
        **kwargs,
    ):
        self.nside = nside
        self.freqs = cp.asarray(freqs)
        self.nest = nest
        self.apply_beam = apply_beam
        self.ground_plane_diameter = ground_plane_diameter
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Build the fixed equatorial pixel grid once, on device.
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)

    def setup(self, spec: OperatorSpec):
        spec.input("VISIBILITY")
        spec.input("WEIGHT")
        spec.input("B_ROT")
        spec.input("BORESIGHT")
        spec.input("time")
        spec.output("cube")
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        vis = cp.asarray(op_input.receive("VISIBILITY"))  # (1, nbl, nchan)
        wgt = cp.asarray(op_input.receive("WEIGHT"))  # (1, nbl, nchan)
        b_rot = cp.asarray(op_input.receive("B_ROT"))  # (1, nbl, 3)
        boresight = cp.asarray(op_input.receive("BORESIGHT"))  # (1, 3)
        times = cp.asarray(op_input.receive("time"))  # (1,)

        beam = None
        if self.apply_beam:
            beam = airy_power_beam(
                self.pix_vec, boresight[0], self.freqs, diameter=self.ground_plane_diameter, xp=cp
            )  # (nchan, npix)
        dmap = image_frame_prerotated(vis, wgt, b_rot, self.pix_vec, self.freqs, beam=beam, xp=cp)  # (npix,)

        # Output layout: (ntime_out=1, npix) -- one dirty-map row per frame.
        op_output.emit(hs.as_tensor(dmap[None, :]), "cube")
        op_output.emit(hs.as_tensor(cp.mean(times, keepdims=True)), "time_out")
