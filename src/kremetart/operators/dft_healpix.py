"""Holoscan operator: full-sky HEALPix dirty-map imager (GPU-resident, xp=cupy).

Thin wrapper around :func:`kremetart.utils.healpix_dft.image_frame`. Receives unstopped residual
visibilities in the raw-visibility layout plus their timestamps, forms the per-frame equatorial
baseline rotation C(t) on the host, and images onto a fixed full-sky equatorial HEALPix grid.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.healpix_dft import image_frame, make_pixel_grid


class HealpixDFTOperator(Operator):
    """Adjoint (dirty-map) HEALPix DFT imager.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        itrs_baselines: ``(nbl, 3)`` ITRS baseline vectors (constant for the array).
        freqs: ``(nchan,)`` frequencies in Hz.
        nest: NESTED HEALPix ordering (default True).
        ctime_backend: ``C(t)`` backend ("astropy" now, "native" later).
    """

    def __init__(self, fragment, nside, itrs_baselines, freqs, *args, nest=True, ctime_backend="astropy", **kwargs):
        self.nside = nside
        self.itrs_baselines = cp.asnumpy(itrs_baselines)  # host; rotation runs on host per frame
        self.freqs = cp.asarray(freqs)
        self.nest = nest
        self.ctime_backend = ctime_backend
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # Build the fixed equatorial pixel grid once, on device.
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)

    def setup(self, spec: OperatorSpec):
        spec.input("VISIBILITY")
        spec.input("WEIGHT")
        spec.input("time")
        spec.output("cube")
        spec.output("time_out")
        spec.output("freq_out")

    def compute(self, op_input, op_output, context):
        vis = cp.asarray(op_input.receive("VISIBILITY"))  # (n_time, nbl, nchan)
        wgt = cp.asarray(op_input.receive("WEIGHT"))  # (n_time, nbl, nchan)
        times = cp.asarray(op_input.receive("time"))  # (n_time,)

        dmap = image_frame(
            vis,
            wgt,
            cp.asnumpy(times),
            self.itrs_baselines,
            self.pix_vec,
            self.freqs,
            ctime_backend=self.ctime_backend,
            xp=cp,
        )

        # Output layout: (ncorr=1, ntime_out=1, nfreq_out=1, npix)
        cube = dmap[None, None, None, :]
        time_out = cp.mean(times, keepdims=True)
        freq_out = cp.mean(self.freqs, keepdims=True)
        op_output.emit(hs.as_tensor(cube), "cube")
        op_output.emit(hs.as_tensor(time_out), "time_out")
        op_output.emit(hs.as_tensor(freq_out), "freq_out")
