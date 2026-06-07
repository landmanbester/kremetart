from pathlib import Path

import holoscan as hs
import numpy as np
import xarray as xr
from holoscan.conditions import CountCondition
from scipy.constants import c as LIGHTSPEED

from kremetart.operators.dft_lm import CoreAlgorithmOperator
from kremetart.operators.io import ResultWriterOperator, XarrayZarrReaderOperator


class StreamingPipeline(hs.core.Application):
    def __init__(
        self,
        MSv4_zarr_path: Path,
        output_dataset: Path,
        *args,
        data_column: str = "VISIBILITY",
        stokes: str = "IQUV",
        **kwargs,
    ):
        self.zarr_path = MSv4_zarr_path
        self.output_dataset = output_dataset
        self.data_column = data_column
        self.stokes = np.array(list(stokes))
        super().__init__(*args, **kwargs)

        # get view of data
        dz = xr.open_datatree(self.zarr_path, engine="zarr", chunks="auto")
        # only the first data group for now (will need to iterate)
        self.dataset = dz[list(dz.children)[0]].ds
        self.nstokes = self.stokes.size
        self.ntime = self.dataset.time.size
        self.out_times = self.dataset.time.values
        self.ntime_out = self.out_times.size
        self.out_freqs = np.mean(self.dataset.frequency.values, keepdims=True)
        self.nfreq_out = self.out_freqs.size

        # get image size
        uvw = self.dataset.UVW.values
        freq = self.dataset.frequency.values
        umax = np.abs(uvw[:, :, 0]).max()
        vmax = np.abs(uvw[:, :, 1]).max()
        uv_max = np.maximum(umax, vmax)
        cell = 1.0 / (2 * uv_max * freq.max() / LIGHTSPEED)  # Nyquist in radians
        field_of_view_deg = 1.0
        nx = int(np.ceil(np.deg2rad(field_of_view_deg) / cell))
        if nx % 2:
            nx += 1
        self.nx = nx
        self.ny = nx
        self.cellx = cell
        self.celly = cell

    def compose(self):
        reader = XarrayZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.zarr_path,
            data_column=self.data_column,
            stokes=self.stokes,
        )

        algorithm = CoreAlgorithmOperator(
            self,
            self.cellx,
            self.celly,
            self.nx,
            self.ny,
            name="core_algorithm",
        )

        writer = ResultWriterOperator(
            self,
            self.nstokes,
            self.nfreq_out,
            self.ntime_out,
            self.ny,
            self.nx,
            name="writer",
            output_dataset=self.output_dataset,
            out_stokes=self.stokes,
            out_freqs=self.out_freqs,
            out_times=self.out_times,
        )

        # Connect the pipeline
        self.add_flow(
            reader,
            algorithm,
            {
                ("UVW", "UVW"),
                ("VISIBILITY", "VISIBILITY"),
                ("WEIGHT", "WEIGHT"),
                ("FLAG", "FLAG"),
                ("FREQ", "FREQ"),
                ("time", "time"),
            },
        )
        self.add_flow(algorithm, writer, {("cube", "cube"), ("time_out", "time_out"), ("freq_out", "freq_out")})


def stream_msv4(
    ms: Path = Path("/data/test_ascii_1h60.0s.zarr"),
    output_dataset: Path = Path("/data/test_stream.zarr"),
):
    app = StreamingPipeline(
        "/data/test_ascii_1h60.0s.zarr",  # input MSv4.zarr
        "/data/test_stream.zarr",
    )  # output_dataset.zarr
    app.config("config.yaml")
    app.run()
