import cupy as cp
import holoscan as hs
import xarray as xr
from holoscan.core import ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext
from numpy.typing import NDArray

from kremetart.utils.map_store import create_map_store, write_maps


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
    """Write each snapshot's maps into the durable map store (:mod:`kremetart.utils.map_store`).

    ``var_specs`` maps input ports to stored variable names, one ``(input_port, variable_name)`` pair
    per map. ``out_times`` must equal the streamed snapshots' times: each row is located by matching
    the received ``time_out`` against them, so a mismatch raises ``KeyError`` at write time.
    """

    def __init__(
        self,
        fragment,
        ntime: int,
        npix: int,
        *args,
        output_dataset: str,
        out_times: NDArray,
        var_specs: tuple[tuple[str, str], ...],
        **kwargs,
    ) -> None:
        self.output_dataset = output_dataset
        self.ntime = ntime
        self.npix = npix
        self.out_times = out_times
        self.var_specs = tuple(var_specs)
        super().__init__(fragment, *args, **kwargs)

    def start(self) -> None:
        create_map_store(self.output_dataset, self.out_times, self.npix, [name for _, name in self.var_specs])

    def setup(self, spec: OperatorSpec) -> None:
        for port, _ in self.var_specs:
            spec.input(port)
        spec.input("time_out")

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext) -> None:
        time_out = float(cp.asnumpy(cp.asarray(op_input.receive("time_out")))[0])
        maps = {name: cp.asnumpy(cp.asarray(op_input.receive(port)))[0] for port, name in self.var_specs}
        write_maps(self.output_dataset, time_out, maps)
