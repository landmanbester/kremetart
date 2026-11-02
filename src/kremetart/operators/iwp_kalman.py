"""Holoscan adapter for the IWP filter stage (:class:`kremetart.imaging.iwp_filter.IWPFilter`, CuPy)."""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.imaging.iwp_filter import IWPFilter


class IWPKalmanOperator(Operator):
    """Map (``cube``) and timestamp (``time_out``) in; the input map (passthrough ``cube``), smooth map
    (``filtered``), normalised innovation (``znorm``) and timestamp out, each with a leading length-1 axis.

    Args:
        fragment: Holoscan fragment.
        npix: number of pixels.
        sigma2: IWP driving variance.
        noise: measurement-noise variance.
    """

    def __init__(self, fragment, npix, *args, sigma2, noise, **kwargs):
        self.npix = npix
        self.sigma2 = float(sigma2)
        self.noise = float(noise)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        self.stage = IWPFilter(self.npix, sigma2=self.sigma2, noise=self.noise, xp=cp)

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("time_out")
        spec.output("cube")
        spec.output("filtered")
        spec.output("znorm")
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        cube = cp.asarray(op_input.receive("cube"))  # (1, npix)
        time_out = cp.asarray(op_input.receive("time_out"))  # (1,)
        smooth, innovation = self.stage.update(cube[0], float(time_out[0]))
        op_output.emit(hs.as_tensor(cube), "cube")
        op_output.emit(hs.as_tensor(smooth[None, :]), "filtered")
        op_output.emit(hs.as_tensor(innovation[None, :]), "znorm")
        op_output.emit(hs.as_tensor(time_out), "time_out")
