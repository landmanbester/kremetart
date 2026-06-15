import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec
from scipy.constants import c as LIGHTSPEED

from kremetart.utils.stokes_expr_cupy import CONVERT_FNS


class CoreAlgorithmOperator(Operator):
    """
    Receives GPU pointer, computes on GPU, emits GPU pointer.
    """

    def __init__(
        self,
        fragment,
        cellx,
        celly,
        nx,
        ny,
        *args,
        pol: str = "linear",
        memory_limit: int = None,
        compute_dtype=cp.float32,
        **kwargs,
    ):
        self.pol = pol
        self.cellx = cellx
        self.celly = celly
        self.nx = nx
        self.ny = ny
        self.memory_limit = memory_limit
        self.compute_dtype = compute_dtype
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        if self.pol not in ("linear", "circular"):
            raise NotImplementedError(f"Polarisation type '{self.pol}' not supported")

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
        uvw = cp.asarray(op_input.receive("UVW"))
        vis = cp.asarray(op_input.receive("VISIBILITY"))
        wgt = cp.asarray(op_input.receive("WEIGHT"))
        flag = cp.asarray(op_input.receive("FLAG"))
        freq = cp.asarray(op_input.receive("FREQ"))
        time = cp.asarray(op_input.receive("time"))

        cube, time_out, freq_out = self._process_on_gpu(uvw, vis, wgt, flag, freq, time)

        op_output.emit(hs.as_tensor(cube), "cube")
        op_output.emit(hs.as_tensor(time_out), "time_out")
        op_output.emit(hs.as_tensor(freq_out), "freq_out")

    def _compute_tile_params(self, nbl, nchan, nstokes, elem_size):
        if self.memory_limit is None:
            free, _ = cp.cuda.Device().mem_info
            available = int(0.8 * free)
        else:
            available = self.memory_limit

        # Subtract resident arrays: weighted_stokes (nbl, nchan, nstokes) complex
        resident = nbl * nchan * nstokes * elem_size * 2  # complex
        available = max(available - resident, elem_size)

        # Per-tile memory: geo_phase (nbl, tx, ty) + kernel (nbl, nchan, tx, ty) complex
        # = nbl * tile_pixels * (elem + nchan * elem * 2)
        # = nbl * tile_pixels * elem * (1 + 2 * nchan)
        bytes_per_pixel = nbl * elem_size * (1 + 2 * nchan)
        tile_pixels = available // bytes_per_pixel
        tile_size = int(tile_pixels**0.5)
        tile_size = max(1, min(tile_size, self.nx, self.ny))

        nchan_batch = nchan
        if tile_size < 4:
            tile_size = min(32, self.nx, self.ny)
            tile_pixels = tile_size * tile_size
            # nbl * tile_pixels * elem * (1 + 2 * nchan_batch) <= available
            nchan_batch = max(1, (available // (nbl * tile_pixels * elem_size) - 1) // 2)

        return tile_size, nchan_batch

    def _corr_to_stokes_batch(self, vis, wgt, flag):
        """Vectorized Stokes conversion + flagging + weighting.

        Operates on full (ntime, nbl, nchan) arrays at once using
        analytic expressions from _stokes_expr_cupy.

        Returns weighted_stokes: (ntime, nbl, nchan, nstokes) with
        flagged entries zeroed.
        """
        mask = ~flag.any(axis=-1, keepdims=True)  # (ntime, nbl, nchan, 1)

        v00, v01, v10, v11 = [vis[..., c] for c in range(4)]
        w00, w01, w10, w11 = [wgt[..., c] for c in range(4)]

        jones_key = "NOJONES"  # will be "JONES" when gains are applied
        pol_key = self.pol.upper()

        stokes_list = []
        for s in ["I", "Q", "U", "V"]:
            vis_fn = CONVERT_FNS[("VIS", pol_key, jones_key, s)]
            wgt_fn = CONVERT_FNS[("WEIGHT", pol_key, jones_key, s)]
            sv = vis_fn(v00, v01, v10, v11)
            sw = wgt_fn(w00, w01, w10, w11)
            stokes_list.append(sv * sw)

        weighted_stokes = cp.stack(stokes_list, axis=-1) * mask
        return weighted_stokes

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
        Vectorized DFT imager with spatial tiling.

        Computes the dirty image as:
          I(x,y) = Re[ sum_{bl,chan} weighted_stokes * exp(i * 2pi * freq * (u*x + v*y - w*nm1) / c) ] / n

        Uses spatial tiling to fit within GPU memory budget, with optional
        channel batching as fallback for very large problems.
        """
        time_out = cp.mean(times, keepdims=True)
        freq_out = cp.mean(freqs, keepdims=True)

        cdtype = self.compute_dtype
        ntime, nbl, nchan, ncorr = vis.shape
        nstokes = 4
        elem_size = cp.dtype(cdtype).itemsize
        # complex type matching compute precision
        cdtype_complex = cp.result_type(cdtype, cp.complex64)

        # --- Image-plane coordinates ---
        x_full = (-self.nx / 2 + cp.arange(self.nx, dtype=cdtype)) * self.cellx
        y_full = (-self.ny / 2 + cp.arange(self.ny, dtype=cdtype)) * self.celly

        # --- W-correction term (full grid) ---
        x2d, y2d = cp.meshgrid(x_full, y_full, indexing="ij")
        eps = x2d**2 + y2d**2
        nm1 = -eps / (cp.sqrt(1.0 - eps) + 1.0)
        n = nm1 + 1.0

        # --- Vectorized Stokes conversion + flagging ---
        weighted_stokes = self._corr_to_stokes_batch(vis, wgt, flag)
        # Flatten time into baseline dim, cast to compute complex dtype
        weighted_stokes = weighted_stokes.reshape(-1, nchan, nstokes).astype(cdtype_complex)
        nbl_total = weighted_stokes.shape[0]

        # --- UVW: flatten and cast ---
        uvw_flat = uvw.reshape(-1, 3).astype(cdtype)
        u = uvw_flat[:, 0]
        v = uvw_flat[:, 1]
        w = uvw_flat[:, 2]

        freqs_c = freqs.astype(cdtype)

        # --- Tiling parameters ---
        tile_size, nchan_batch = self._compute_tile_params(nbl_total, nchan, nstokes, elem_size)

        # --- Accumulation buffer (complex, take .real at end) ---
        res = cp.zeros((nstokes, self.nx, self.ny), dtype=cdtype_complex)

        # --- Tiled DFT ---
        for ix0 in range(0, self.nx, tile_size):
            ix1 = min(ix0 + tile_size, self.nx)
            x_tile = x_full[ix0:ix1]
            nm1_rows = nm1[ix0:ix1, :]

            for iy0 in range(0, self.ny, tile_size):
                iy1 = min(iy0 + tile_size, self.ny)
                y_tile = y_full[iy0:iy1]
                nm1_tile = nm1_rows[:, iy0:iy1]
                tx, ty = ix1 - ix0, iy1 - iy0

                # Geometric phase: (nbl_total, tx, ty)
                geo_phase = (2.0 * cp.pi / LIGHTSPEED) * (
                    u[:, None, None] * x_tile[None, :, None]
                    + v[:, None, None] * y_tile[None, None, :]
                    - w[:, None, None] * nm1_tile[None, :, :]
                )

                for ic0 in range(0, nchan, nchan_batch):
                    ic1 = min(ic0 + nchan_batch, nchan)
                    freq_batch = freqs_c[ic0:ic1]
                    ws_batch = weighted_stokes[:, ic0:ic1, :]

                    # Full phase: (nbl_total, cb, tx, ty)
                    full_phase = freq_batch[None, :, None, None] * geo_phase[:, None, :, :]

                    # Complex exponential kernel
                    kernel = cp.exp(1j * full_phase)

                    # Contract via matmul: (nstokes, nbl*cb) @ (nbl*cb, tx*ty) -> (nstokes, tx*ty)
                    ws_flat = ws_batch.reshape(-1, nstokes).T
                    kernel_flat = kernel.reshape(-1, tx * ty)
                    tile_result = ws_flat @ kernel_flat

                    res[:, ix0:ix1, iy0:iy1] += tile_result.reshape(nstokes, tx, ty)

        # Take real part and apply n-term correction
        res = res.real / n[None, :, :]

        # Output axes: (nstokes, nfreq_out, ntime_out, nx, ny)
        res = res[:, None, None, :, :]
        return res, time_out, freq_out
