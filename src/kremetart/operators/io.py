import cupy as cp
import dask.array as da
import holoscan as hs
import numpy as np
import xarray as xr
from holoscan.core import Operator, OperatorSpec
from numpy.typing import NDArray


class HealpixZarrReaderOperator(Operator):
    """Stream a prepared imaging zarr (VISIBILITY, WEIGHT, B_ROT, time) one frame at a time to the GPU.

    Reads the host prepare-step output (:func:`kremetart.utils.read_tart_hdf.prepare_msv4_zarr`),
    which carries the precomputed ``B_ROT`` and no UVW/FLAG (the HEALPix imager does not need them).
    """

    def __init__(self, fragment, *args, zarr_path: str, **kwargs):
        self.zarr_path = zarr_path
        self.current_index = 0
        self.ntime = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("VISIBILITY")
        spec.output("WEIGHT")
        spec.output("B_ROT")
        spec.output("BORESIGHT")
        spec.output("time")

    def start(self):
        self.dataset = xr.open_zarr(self.zarr_path)
        self.ntime = self.dataset["time"].size
        self.out_times = self.dataset["time"].values
        self.current_index = 0

    def compute(self, op_input, op_output, context):
        if self.current_index >= self.ntime:
            return
        s = self.dataset.isel({"time": slice(self.current_index, self.current_index + 1)})
        op_output.emit(hs.as_tensor(cp.asarray(s["VISIBILITY"].values)), "VISIBILITY")
        op_output.emit(hs.as_tensor(cp.asarray(s["WEIGHT"].values)), "WEIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(s["B_ROT"].values)), "B_ROT")
        op_output.emit(hs.as_tensor(cp.asarray(s["BORESIGHT"].values)), "BORESIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(s["time"].values)), "time")
        self.current_index += 1

    def stop(self):
        pass


class HealpixWriterOperator(Operator):
    """Write per-frame HEALPix maps to a ``(TIME, PIX)`` zarr (dask scaffold + region="auto").

    The stored variables are configured by ``var_specs`` (one ``(input_port, variable_name)`` pair
    per map): a lazy dask scaffold is written once, then each frame is written into its region of the
    flat ``(TIME, npix)`` variables.

    ``out_times`` MUST equal the streamed frames' ``time`` values (the prepared zarr's ``time``
    coordinate): ``region="auto"`` locates each frame by matching its emitted ``time_out`` against
    the scaffold's ``TIME`` coordinate, so a mismatch raises ``KeyError`` at write time. The app
    builds both the scaffold (via this ``out_times``) and the reader's stream from the same zarr, so
    they agree by construction.
    """

    def __init__(
        self,
        fragment,
        ntime,
        npix,
        *args,
        output_dataset: str | None = None,
        out_times: NDArray = None,
        var_specs: tuple[tuple[str, str], ...] = (("cube", "dirty"), ("filtered", "filtered"), ("znorm", "znorm")),
        **kwargs,
    ):
        self.output_dataset = output_dataset
        self.ntime = ntime
        self.npix = npix
        self.out_times = out_times if out_times is not None else np.arange(ntime)
        self.pix = np.arange(npix)
        # var_specs maps (input_port -> stored_variable_name); the default below is a minimal schema.
        # smoovie's pipeline overrides it with dirty/tikhonov/l1/filtered/znorm sourced from the
        # imager, both deconvolvers and the IWP (see kremetart.core.smoovie.SmooviePipeline.compose).
        self.var_specs = tuple(var_specs)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # dask scaffold: da.empty allocates no data, only the zarr structure to write regions into.
        data_vars = {
            name: (("TIME", "PIX"), da.empty((self.ntime, self.npix), chunks=(1, self.npix), dtype=np.float32))
            for _, name in self.var_specs
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), self.out_times), "PIX": (("PIX",), self.pix)},
        )
        ds.to_zarr(self.output_dataset, mode="w", compute=True)

    def setup(self, spec: OperatorSpec):
        for port, _ in self.var_specs:
            spec.input(port)
        spec.input("time_out")

    def compute(self, op_input, op_output, context):
        time_out = cp.asnumpy(cp.asarray(op_input.receive("time_out")))  # (1,)
        data_vars = {
            name: (("TIME", "PIX"), cp.asnumpy(cp.asarray(op_input.receive(port))).astype(np.float32))
            for port, name in self.var_specs
        }
        dso = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), time_out), "PIX": (("PIX",), self.pix)},
        )
        dso.to_zarr(self.output_dataset, region="auto")

    def stop(self):
        pass
