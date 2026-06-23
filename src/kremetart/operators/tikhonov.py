"""Holoscan operator: per-frame Tikhonov-regularised deconvolution via CG (GPU-resident, xp=cupy).

Sits between the HEALPix imager and the IWP filter. Solves ``(H + λI) x = b`` per frame, where
``H = B Mᴴ W M B`` is the image-space Hessian (:func:`kremetart.utils.healpix_dft.hessian_healpix`),
``b`` is the un-normalised dirty image (the imager's normalised dirty times ``Σw``), and
``λ = eta·Σw`` makes ``eta`` a frame-invariant fraction of the central PSF value ``Σw``. It needs no
visibilities — only the weights/geometry/beam that build ``H`` and the imager's dirty map as the RHS.

The Jacobi preconditioner ``1/(diag(H) + λ)`` and warm-start (seed each frame with the previous
frame's solution, kept on-device) are optional flags; the reference/self-adjointness tests exercise
:func:`kremetart.opt.cg.cg` and :func:`hessian_healpix` directly with both disabled. See
docs/superpowers/specs/2026-06-23-tikhonov-cg-regularisation-design.md.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.opt.cg import cg
from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam
from kremetart.utils.healpix_dft import hessian_healpix, make_pixel_grid


class TikhonovOperator(Operator):
    """Per-frame Tikhonov deconvolution (regularised dirty image) via preconditioned CG.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        eta: regularisation strength as a fraction of ``Σw`` (``λ = eta·Σw``); must be > 0.
        nest: NESTED HEALPix ordering (default True).
        apply_beam: build the Airy beam into ``H`` (must match the imager's setting).
        ground_plane_diameter: Airy aperture diameter in metres.
        maxiter: maximum CG iterations per frame.
        tol: CG relative-residual tolerance.
        use_preconditioner: apply the Jacobi preconditioner ``1/(diag(H)+λ)``.
        use_warm_start: seed each frame's CG with the previous frame's solution.
    """

    def __init__(
        self,
        fragment,
        nside,
        freqs,
        eta,
        *args,
        nest=True,
        apply_beam=True,
        ground_plane_diameter=GROUND_PLANE_DIAMETER,
        maxiter=100,
        tol=1e-5,
        use_preconditioner=True,
        use_warm_start=True,
        **kwargs,
    ):
        self.nside = nside
        self.freqs = cp.asarray(freqs)
        self.eta = float(eta)
        self.nest = nest
        self.apply_beam = apply_beam
        self.ground_plane_diameter = ground_plane_diameter
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.use_preconditioner = use_preconditioner
        self.use_warm_start = use_warm_start
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)
        self.x_prev = None  # device-resident warm-start state

    def setup(self, spec: OperatorSpec):
        spec.input("cube")  # imager dirty map = RHS (normalised)
        spec.input("WEIGHT")
        spec.input("B_ROT")
        spec.input("BORESIGHT")
        spec.input("time_out")
        spec.output("cube")  # regularised image -> IWP
        spec.output("dirty")  # raw dirty passthrough -> writer
        spec.output("time_out")

    def compute(self, op_input, op_output, context):
        dirty = cp.asarray(op_input.receive("cube"))  # (1, npix)
        weights = cp.asarray(op_input.receive("WEIGHT"))  # (1, nbl, nchan)
        b_rot = cp.asarray(op_input.receive("B_ROT"))  # (1, nbl, 3)
        boresight = cp.asarray(op_input.receive("BORESIGHT"))  # (1, 3)
        time_out = cp.asarray(op_input.receive("time_out"))  # (1,)

        w = weights[0]  # (nbl, nchan)
        wsum = w.sum()
        # Fully-flagged frame: λ = eta·Σw = 0 makes H + λI singular and the imager's dirty is the
        # all-zero no-data map, so there is nothing to solve. Pass the zero map straight through (the
        # IWP reads it as a no-data frame and coasts); leave the warm-start untouched so the next
        # live frame still seeds from the last good solution.
        if float(wsum) == 0.0:
            zeros = cp.zeros(self.pix_vec.shape[0], dtype=cp.float64)
            op_output.emit(hs.as_tensor(zeros[None, :]), "cube")
            op_output.emit(hs.as_tensor(dirty), "dirty")
            op_output.emit(hs.as_tensor(time_out), "time_out")
            return

        beam = None
        if self.apply_beam:
            beam = airy_power_beam(
                self.pix_vec, boresight[0], self.freqs, diameter=self.ground_plane_diameter, xp=cp
            )  # (nchan, npix)

        rows = b_rot[0]  # (nbl, 3)
        hmv, hdiag = hessian_healpix(rows, self.pix_vec, self.freqs, w, beam=beam, xp=cp)
        lam = self.eta * wsum

        def a_matvec(x):
            return hmv(x) + lam * x

        precond = None
        if self.use_preconditioner:
            inv_mdiag = 1.0 / (hdiag + lam)

            def precond(r):
                return r * inv_mdiag

        x0 = None
        if self.use_warm_start and self.x_prev is not None and bool(cp.all(cp.isfinite(self.x_prev))):
            x0 = self.x_prev

        b = dirty[0] * wsum  # un-normalise the imager's normalised dirty to the Hessian RHS
        x = cg(a_matvec, b, x0=x0, M=precond, maxiter=self.maxiter, tol=self.tol, xp=cp)
        self.x_prev = x

        op_output.emit(hs.as_tensor(x[None, :]), "cube")
        op_output.emit(hs.as_tensor(dirty), "dirty")
        op_output.emit(hs.as_tensor(time_out), "time_out")
