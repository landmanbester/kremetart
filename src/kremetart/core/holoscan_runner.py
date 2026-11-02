"""The Holoscan runner: image a prepared snapshot zarr through a GPU Holoscan graph, one operator per stage.

reader -> imager -> Tikhonov and L1 (each when enabled) -> writer, with the IWP filter on the Tikhonov
stream (the dirty stream when Tikhonov is off), plus a web sink when a holder is given. Holoscan
broadcasts shared output ports, so the writer and the sink both receive every map.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import holoscan as hs
import xarray as xr
from holoscan.conditions import CountCondition

from kremetart.core.settings import ImagingSettings
from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
from kremetart.operators.iwp_kalman import IWPKalmanOperator
from kremetart.operators.l1_reweight import L1ReweightOperator
from kremetart.operators.snapshot_ports import snapshot_flow
from kremetart.operators.tikhonov import TikhonovOperator
from kremetart.operators.web_sink import WebStreamSinkOperator
from kremetart.utils.healpix_viz import LatestFrameHolder


class SmooviePipeline(hs.core.Application):
    """The Holoscan graph that streams a prepared snapshot zarr into the map store.

    Args:
        prepared: the prepared snapshot zarr.
        output: the map store to create.
        settings: the imaging settings.
        holder: when given, a terminal web sink publishes every snapshot's maps to it.
    """

    def __init__(
        self,
        prepared: Path,
        output: Path,
        settings: ImagingSettings,
        holder: LatestFrameHolder | None = None,
        **kwargs: object,
    ) -> None:
        self.prepared = str(prepared)
        self.output = str(output)
        self.settings = settings
        self.holder = holder
        super().__init__(**kwargs)

        ds = xr.open_zarr(self.prepared)
        self.times = ds.time.values
        self.freqs = ds.frequency.values

    def compose(self) -> None:
        s = self.settings
        grid = {"nest": s.nest, "apply_beam": s.apply_beam, "ground_plane_diameter": s.ground_plane_diameter}
        reader = HealpixZarrReaderOperator(
            self, CountCondition(self, len(self.times)), name="reader", zarr_path=self.prepared
        )
        imager = HealpixDFTOperator(self, s.nside, self.freqs, name="imager", **grid)
        # Both deconvolvers are opt-in: they dominate the runtime.
        tikhonov = None if s.l2 is None else TikhonovOperator(self, s.nside, self.freqs, s.l2, name="tikhonov", **grid)
        l1 = None if s.l1 is None else L1ReweightOperator(self, s.nside, self.freqs, s.l1, name="l1reweight", **grid)
        iwp = IWPKalmanOperator(self, s.npix, name="iwp", sigma2=s.iwp_sigma, noise=s.iwp_noise)
        writer = HealpixWriterOperator(
            self,
            len(self.times),
            s.npix,
            name="writer",
            output_dataset=self.output,
            out_times=self.times,
            var_specs=tuple((name, name) for name in s.stored_names),
        )

        # Every stage operator receives the whole snapshot; the solvers also take the dirty map.
        self.add_flow(reader, imager, snapshot_flow())
        for solver in (tikhonov, l1):
            if solver is not None:
                self.add_flow(reader, solver, snapshot_flow())
                self.add_flow(imager, solver, {("cube", "cube")})
        # The IWP filter smooths the Tikhonov stream, or the dirty stream when Tikhonov is off; it takes
        # the snapshot time from the imager. Every output port is connected: an unconnected one stalls
        # the graph.
        self.add_flow(imager, iwp, {("time_out", "time_out")})
        self.add_flow(imager if tikhonov is None else tikhonov, iwp, {("cube", "cube")})

        # (target, the target's port for the smooth map): the writer stores it as "filtered".
        sinks = [(writer, "filtered")]
        if self.holder is not None:
            sink = WebStreamSinkOperator(self, name="websink", holder=self.holder, names=s.stream_names)
            sinks.append((sink, "smooth"))
        for target, smooth_port in sinks:
            self.add_flow(iwp, target, {("filtered", smooth_port), ("znorm", "znorm"), ("time_out", "time_out")})
            self.add_flow(imager, target, {("cube", "dirty")})
            for name, solver in (("tikhonov", tikhonov), ("l1", l1)):
                if solver is not None:
                    self.add_flow(solver, target, {("cube", name)})


def run_holoscan(
    prepared: Path, output: Path, settings: ImagingSettings, holder: LatestFrameHolder | None = None
) -> None:
    """Image every snapshot in ``prepared`` into the map store at ``output`` (the runner interface)."""
    with tempfile.TemporaryDirectory() as td:
        config = Path(td) / "config.yaml"
        config.touch()  # an empty Holoscan config is valid
        app = SmooviePipeline(prepared, output, settings, holder)
        app.config(str(config))
        app.run()
