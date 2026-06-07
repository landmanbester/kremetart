import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from scipy.constants import c as LIGHTSPEED


class CoreAlgorithmOperator(Operator):
    """
    Receives GPU pointer, computes on GPU, emits GPU pointer.
    """

    def __init__(self, fragment, cellx, celly, nx, ny, *args, pol: str = "linear", **kwargs):
        self.pol = pol
        self.cellx = cellx
        self.celly = celly
        self.nx = nx
        self.ny = ny
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # these operators map stokes_to_corr/corr_to_stokes
        if self.pol == "linear":
            self.s2c = cp.array([[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0j], [0, 0, 1.0, -1.0j], [1.0, -1.0, 0, 0]])
            self.c2s = cp.array([[0.5, 0.0, 0, 0.5], [0.5, 0, 0.0, -0.5], [0, 0.5, 0.5, 0], [0, -0.5j, 0.5j, 0]])
        else:
            raise NotImplementedError

    def setup(self, spec: OperatorSpec):
        spec.input("UVW")
        spec.input("VISIBILITY")
        spec.input("WEIGHT")
        spec.input("FLAG")
        spec.input("FREQ")
        spec.input("time")

        spec.output("cube")
        spec.output("time_out")
        spec.output("freq_out")

    def compute(self, op_input, op_output, context):
        # Receive tensor (GPU pointer)
        uvw = cp.asarray(op_input.receive("UVW"))
        vis = cp.asarray(op_input.receive("VISIBILITY"))
        wgt = cp.asarray(op_input.receive("WEIGHT"))
        flag = cp.asarray(op_input.receive("FLAG"))
        freq = cp.asarray(op_input.receive("FREQ"))
        time = cp.asarray(op_input.receive("time"))

        # Core algorithm here. This all on seems to happen in the GPU
        cube, time_out, freq_out = self._process_on_gpu(uvw, vis, wgt, flag, freq, time)

        op_output.emit(hs.as_tensor(cube), "cube")
        op_output.emit(hs.as_tensor(time_out), "time_out")
        op_output.emit(hs.as_tensor(freq_out), "freq_out")

    def _corr_to_stokes(self, vis, wgt):
        # this can actually be done analytically including Jones matrix application
        stokes_wgt = self.s2c.conj().T @ (wgt[:, None] * self.s2c)
        stokes_vis = cp.linalg.solve(stokes_wgt, self.s2c.conj().T @ (wgt * vis))
        return stokes_vis, cp.diag(stokes_wgt)  # keep only diagonal wgt

    def _process_on_gpu(
        self,
        uvw: cp.ndarray,
        vis: cp.ndarray,
        wgt: cp.ndarray,
        flag: cp.ndarray,
        freqs: cp.ndarray,
        times: cp.ndarray,
    ) -> cp.ndarray:
        """
        Eventually need to duplicate functionality here

        https://github.com/ratt-ru/pfb-imaging/blob/main/pfb/utils/stokes2im.py
        """
        # currently assume reduction over time and freq
        time_out = cp.mean(times, keepdims=True)
        freq_out = cp.mean(freqs, keepdims=True)
        x, y = cp.meshgrid(*[-ss / 2 + cp.arange(ss) for ss in [self.nx, self.ny]], indexing="ij")
        x *= self.cellx
        y *= self.celly
        eps = x**2 + y**2
        apply_w = True  # probably not necessary
        if apply_w:
            nm1 = -eps / (cp.sqrt(1.0 - eps) + 1.0)
            n = (nm1 + 1)[None, :, :]
        else:
            nm1 = 0.0
            n = 1.0
        ntime, nbl, nchan, ncorr = vis.shape
        res = cp.zeros((ncorr, self.nx, self.ny), dtype=wgt.dtype)
        # this should be vectorized for better paralellism
        for t in range(ntime):  # probably only one of these
            for bl in range(nbl):
                for chan in range(nchan):
                    # skip if any corrrelation is flagged
                    if flag[t, bl, chan].any():
                        continue
                    # convert to corrected Stokes vis and weights (harder to vectorize with gains)
                    stokes_vis, stokes_wgt = self._corr_to_stokes(vis[t, bl, chan], wgt[t, bl, chan])
                    u, v, w = uvw[t, bl]
                    phase = freqs[chan] / LIGHTSPEED * (x * u + y * v - w * nm1)
                    cphase = cp.exp(2j * cp.pi * phase)
                    res += (stokes_vis[:, None, None] * stokes_wgt[:, None, None] * cphase[None, :, :]).real
        res /= n
        # axes placeholders
        res = res[:, None, None, :, :]
        return res, time_out, freq_out
