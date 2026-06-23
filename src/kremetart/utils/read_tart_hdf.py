from __future__ import annotations

import datetime
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from tart_tools import api_handler

from kremetart.utils import partition_datatree
from kremetart.utils.calibration import correct_file_gains
from kremetart.utils.healpix_dft import equatorial_baselines, zenith_icrs_vectors


def read_hdf_as_xr(path, filter_elevation=45):
    # open hdf as xarray dataset
    ds = xr.open_dataset(path, engine="netcdf4")

    # get the config
    # eg for the Mauritius array, this is the config dict:
    # {'name': 'Bel Air - Mauritius',
    # 'frequency': 1575420000.0,
    # 'L0_frequency': 1571328000.0,
    # 'baseband_frequency': 4092000.0,
    # 'sampling_frequency': 16368000.0,
    # 'bandwidth': 2500000.0,
    # 'lat': -20.2587508,
    # 'lon': 57.7591989,
    # 'alt': 20.0,
    # 'num_antenna': 24,
    # 'orientation': 0.0,
    # 'axes': ['East', 'North', 'Up']}
    config = json.loads(ds.config.values[0])

    # attributes
    attrs = {}
    for key, value in config.items():
        attrs[key] = value
    attrs["phase_elaz_deg"] = ds.phase_elaz.values

    # coordinates
    n_baselines = ds.baselines.shape[0]
    n_antenna = ds.antenna_positions.shape[0]
    n_times = ds.timestamp.shape[0]
    coords = {
        "time": ("time", ds.timestamp.values),
        "frequency": ("frequency", np.array([config["frequency"]])),
        "baseline_id": ("baseline_id", np.arange(n_baselines)),
        "antenna_id": ("antenna_id", np.arange(n_antenna)),
        "antenna1": ("antenna1", ds.baselines.values[:, 0]),
        "antenna2": ("antenna2", ds.baselines.values[:, 1]),
        "polarisation": ("polarisation", np.array(["LL"])),
        "enu": ("enu", np.array(config["axes"])),
    }

    # flag faulty antenna
    faulty_idx = np.where(ds.gains.values == 0)[0]
    flag = np.zeros(n_baselines, dtype=bool)
    for idx in faulty_idx:
        flag |= (ds.baselines.values[:, 0] == idx) | (ds.baselines.values[:, 1] == idx)
    flag = np.broadcast_to(flag[None, :], (n_times, n_baselines))

    # data variables
    shape = (ds.timestamp.shape[0], ds.baselines.shape[0], 1)  # add polarisation dim
    data_vars = {
        "antenna_positions": (("antenna_id", "enu"), ds.antenna_positions.values),
        "vis": (("time", "baseline_id", "polarisation"), ds.vis.values[:, :, None]),
        "weight": (("time", "baseline_id", "polarisation"), np.ones(shape, dtype=np.float32)),
        "flag": (("time", "baseline_id", "polarisation"), flag[:, :, None]),
    }

    # query the API to get the satelite positions
    lat = config["lat"]
    lon = config["lon"]
    api = api_handler.APIhandler("")
    nretry = 5
    catalog = []
    for t in ds.timestamp.values:
        cat_url = (
            api.catalog_url(
                lon,
                lat,
                datestr=t,
            )
            + f"&elevation={filter_elevation}"
        )
        retry = 0
        while retry < nretry:
            try:
                cat = api.get_url(cat_url)
                break
            except Exception as e:
                print(f"Error fetching catalog (attempt {retry + 1}/{nretry}): {e}")
                retry += 1
        else:
            raise RuntimeError(f"Failed to fetch catalog after {nretry} attempts.")
        catalog.append(cat)

    # append source data vars
    data_vars["source_name"] = (("time",), [cat[0]["name"] for cat in catalog])
    data_vars["source_elevation_deg"] = (("time",), [cat[0]["el"] for cat in catalog])
    data_vars["source_azimuth_deg"] = (("time",), [cat[0]["az"] for cat in catalog])
    data_vars["source_flux_jy"] = (("time",), [cat[0]["jy"] for cat in catalog])
    data_vars["source_height_m"] = (("time",), [cat[0]["r"] for cat in catalog])

    # create xarray dataset
    xds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    return xds


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to a geocentric ITRS/ECEF position.

    Args:
        lat_deg: Geodetic latitude in degrees.
        lon_deg: Geodetic longitude in degrees.
        alt_m: Height above the WGS84 ellipsoid in metres.

    Returns:
        The (x, y, z) ITRS position in metres as a length-3 array.
    """
    a = 6378137.0  # WGS84 semi-major axis (m)
    f = 1.0 / 298.257223563  # WGS84 flattening
    e2 = f * (2.0 - f)  # first eccentricity squared
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)  # prime vertical radius of curvature
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * np.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def _enu_to_ecef(enu: np.ndarray, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Map local ENU offsets at a site to geocentric ITRS/ECEF positions.

    Args:
        enu: ``(n, 3)`` array of East/North/Up offsets (m) relative to the site origin.
        lat_deg: Site geodetic latitude in degrees.
        lon_deg: Site geodetic longitude in degrees.
        alt_m: Site height above the WGS84 ellipsoid in metres.

    Returns:
        An ``(n, 3)`` array of geocentric ITRS positions in metres.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    site = _geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    # Rotation from local ENU to ECEF axes (standard topocentric->geocentric matrix).
    rot = np.array(
        [
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0.0, np.cos(lat), np.sin(lat)],
        ]
    )
    return site + enu @ rot.T


def read_hdf_as_msv4(path: str | Path) -> xr.DataTree:
    """Read a TART visibility HDF chunk into an MSv4-compliant xarray DataTree.

    A TART HDF file stores a ~one-minute chunk of single-polarisation, single-channel
    visibilities, the array configuration, and a snapshot of TART's own gain solution. This
    reader converts that into the Measurement Set v4 (MSv4) layout used throughout the
    pipeline so every downstream operator consumes one canonical schema.

    Design decisions (rationale in the inline comments):
      * Returns a ``DataTree``: TART has a single spectral window and a single polarisation, so
        there is exactly one partition node. It holds the main visibility xds plus the MSv4
        ``antenna_xds`` and ``field_and_source`` sub-nodes, and a non-standard ``gain_xds``.
      * No network access. The calibrator/source catalogue requires ephemerides (the TART API
        or sgp4) and is built by a separate sky-model step; reading a file must stay hermetic.
      * Antenna positions are stored both as geocentric ITRS (the MSv4 standard) and as the
        native local ENU vectors the calibration and imaging operators actually consume.
      * Visibilities are tagged dimensionless (``units='1'``): TART visibilities are normalised
        correlation coefficients, not Jy, consistent with the relative self-calibrated scale.

    Args:
        path: Path to a TART ``vis_*.hdf`` file.

    Returns:
        An ``xarray.DataTree`` rooted at a single partition node, MSv4-compliant for the
        visibility/antenna/field-and-source groups, with a TART-specific gain sub-node.
    """
    path = Path(path)

    # The TART HDF is netCDF4-readable, so reuse xarray's reader rather than dropping to h5py
    # (keeps parity with read_hdf_as_xr above).
    raw = xr.open_dataset(path, engine="netcdf4")

    # --- array configuration ---------------------------------------------------------------
    config = json.loads(raw.config.values[0])
    freq = float(config["frequency"])  # L1 centre (~1.575 GHz)
    bandwidth = float(config["bandwidth"])
    lat, lon, alt = float(config["lat"]), float(config["lon"]), float(config["alt"])
    telescope_name = "TART"  # the instrument
    station_name = config.get("name", "TART")  # the site, e.g. "Bel Air - Mauritius"

    # --- dimensions and antenna naming -----------------------------------------------------
    n_ant = raw.antenna_positions.shape[0]
    n_bl = raw.baselines.shape[0]
    baselines = raw.baselines.values  # (n_bl, 2) integer antenna indices, i < j
    # MSv4 keys baselines by antenna *name*, resolved against antenna_xds, not by index.
    antenna_names = np.array([f"ant{p:02d}" for p in range(n_ant)])

    # --- time: ISO-8601 strings -> float64 unix seconds (UTC) ------------------------------
    # The filter rests on Delta_k between frames; unix-second floats make that a plain
    # subtraction and match the MSv4 'time' convention (units s, format unix, scale utc).
    def _as_str(t: object) -> str:
        return t.decode() if isinstance(t, (bytes, bytearray)) else str(t)

    iso = [_as_str(t) for t in raw.timestamp.values]
    time_unix = np.array([datetime.datetime.fromisoformat(t).timestamp() for t in iso], dtype=np.float64)
    n_time = time_unix.size
    # Per-frame integration time is not recorded; infer it from the median cadence.
    integration_time = float(np.median(np.diff(time_unix))) if n_time > 1 else float("nan")

    # --- dead-antenna flagging -------------------------------------------------------------
    # TART marks dead antennas with gain == 0 (e.g. antenna 15 in the validation archive).
    # Per the design these are pinned out of the calibration state; here we flag every baseline
    # touching them and zero its weight.
    gains_amp = raw.gains.values  # (n_ant,) real amplitude
    gains_phase = raw.phases.values  # (n_ant,) real phase (rad)
    dead = np.where(gains_amp == 0)[0]
    bl_flag = np.zeros(n_bl, dtype=bool)
    for idx in dead:
        bl_flag |= (baselines[:, 0] == idx) | (baselines[:, 1] == idx)
    # MSv4 data arrays are 4-D: (time, baseline_id, frequency, polarization).
    flag = np.broadcast_to(bl_flag[None, :, None, None], (n_time, n_bl, 1, 1)).astype(np.uint8)

    # --- data arrays, shaped (time, baseline_id, frequency, polarization) ------------------
    # TART is single-channel, single-pol -> size-1 frequency and polarization axes. netCDF4
    # exposes the complex visibilities as a compound (r, i) dtype, so reassemble if needed.
    vis_raw = raw.vis.values
    if vis_raw.dtype.names is not None and {"r", "i"} <= set(vis_raw.dtype.names):
        vis_raw = vis_raw["r"] + 1j * vis_raw["i"]
    vis = vis_raw[:, :, None, None].astype(np.complex64)
    weight = np.ones((n_time, n_bl, 1, 1), dtype=np.float32)
    weight[flag.astype(bool)] = 0.0  # flag wins: dead baselines carry zero weight

    # --- UVW -------------------------------------------------------------------------------
    # TART points at zenith and does not fringe-stop, so the phase centre is the local zenith.
    # With w along the zenith the (u, v, w) frame coincides with local ENU, hence UVW is simply
    # the ENU baseline vector (East, North, Up) and is time-independent (the array is fixed).
    # The per-frame rotation onto the equatorial grid is applied downstream by the imager, not
    # here. Sign convention: b = pos(ant1) - pos(ant2); this must be cross-checked against the
    # correlator's phase convention during imaging validation.
    enu = raw.antenna_positions.values.astype(np.float64)  # (n_ant, 3): East, North, Up (Up == 0)
    bl_enu = enu[baselines[:, 0]] - enu[baselines[:, 1]]  # (n_bl, 3)
    uvw = np.broadcast_to(bl_enu[None, :, :], (n_time, n_bl, 3)).astype(np.float64)

    # --- main visibility xds ---------------------------------------------------------------
    # TART GPS patch antennas are right-hand circularly polarised -> 'RR' (not 'LL').
    pol = np.array(["RR"])
    main = xr.Dataset(
        data_vars={
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                vis,
                {"type": "quantity", "units": "1"},  # dimensionless correlation coefficients
            ),
            "WEIGHT": (("time", "baseline_id", "frequency", "polarization"), weight),
            "FLAG": (("time", "baseline_id", "frequency", "polarization"), flag),
            "UVW": (
                ("time", "baseline_id", "uvw_label"),
                uvw,
                {"type": "uvw", "units": "m", "frame": "ENU-zenith"},
            ),
            # Per-baseline time bookkeeping: cheap broadcasts that improve MSv4 fidelity.
            "TIME_CENTROID": (
                ("time", "baseline_id"),
                np.broadcast_to(time_unix[:, None], (n_time, n_bl)).astype(np.float64),
            ),
            "EFFECTIVE_INTEGRATION_TIME": (
                ("time", "baseline_id"),
                np.full((n_time, n_bl), integration_time, dtype=np.float64),
            ),
        },
        coords={
            "time": (
                "time",
                time_unix,
                {
                    "type": "time",
                    "units": "s",
                    "format": "unix",
                    "scale": "utc",
                    "integration_time": {"attrs": {"type": "quantity", "units": "s"}, "data": integration_time},
                },
            ),
            "baseline_id": ("baseline_id", np.arange(n_bl)),
            "baseline_antenna1_name": ("baseline_id", antenna_names[baselines[:, 0]]),
            "baseline_antenna2_name": ("baseline_id", antenna_names[baselines[:, 1]]),
            # MSv4 per-time labels. TART has one field (the zenith) and one scan per chunk, both
            # constant across the frames; field_name stays "zenith" since we do not rephase here.
            "field_name": ("time", np.full(n_time, "zenith")),
            "scan_name": ("time", np.full(n_time, "1")),
            "frequency": (
                "frequency",
                np.array([freq], dtype=np.float64),
                {
                    "type": "spectral_coord",
                    "units": "Hz",
                    "observer": "TOPO",
                    "reference_frequency": {
                        "attrs": {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
                        "data": freq,
                    },
                    "channel_width": {"attrs": {"type": "quantity", "units": "Hz"}, "data": bandwidth},
                },
            ),
            "polarization": ("polarization", pol),
            "uvw_label": ("uvw_label", np.array(["u", "v", "w"])),
        },
        attrs={
            "schema_version": "4.0.0",
            "type": "visibility",
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "flag": "FLAG",
                    "weight": "WEIGHT",
                    "uvw": "UVW",
                    "description": "TART single-polarisation L1 visibilities",
                }
            },
            "observation_info": {
                "telescope_name": telescope_name,
                "site_latitude_deg": lat,
                "site_longitude_deg": lon,
                "site_altitude_m": alt,
            },
            "processor_info": {"type": "TART", "sub_type": "correlator"},
            "creator": {"software_name": "kremetart"},
            "source_hdf": path.name,
            # creation_date is deliberately omitted so the output is deterministic for tests.
        },
    )

    # --- antenna_xds -----------------------------------------------------------------------
    ecef = _enu_to_ecef(enu, lat, lon, alt)  # geocentric ITRS positions (MSv4 standard)
    antenna = xr.Dataset(
        data_vars={
            "ANTENNA_POSITION": (
                ("antenna_name", "cartesian_pos_label"),
                ecef,
                {
                    "type": "location",
                    "units": "m",
                    "frame": "ITRS",
                    "coordinate_system": "geocentric",
                    "origin_object_name": "earth",
                },
            ),
            # Non-standard but load-bearing: the calibration and imaging operators work natively
            # in the local ENU frame, so the original ENU positions are kept alongside ITRS.
            "ANTENNA_POSITION_ENU": (
                ("antenna_name", "enu_label"),
                enu,
                {"type": "location", "units": "m", "frame": "ENU"},
            ),
            # TART uses small GPS patch antennas (no dish); diameter is nominal/unused.
            "ANTENNA_DISH_DIAMETER": (("antenna_name",), np.zeros(n_ant), {"type": "quantity", "units": "m"}),
            # Patch antennas: no dish, no meaningful receptor angle. Present (as zeros) for MSv4
            # schema parity with the canonical converter.
            "ANTENNA_EFFECTIVE_DISH_DIAMETER": (
                ("antenna_name",),
                np.zeros(n_ant),
                {"type": "quantity", "units": "m"},
            ),
            "ANTENNA_RECEPTOR_ANGLE": (
                ("antenna_name", "receptor_label"),
                np.zeros((n_ant, 1)),
                {"type": "quantity", "units": "rad"},
            ),
        },
        coords={
            "antenna_name": ("antenna_name", antenna_names),
            "cartesian_pos_label": ("cartesian_pos_label", np.array(["x", "y", "z"])),
            "enu_label": ("enu_label", np.array(["e", "n", "u"])),
            "receptor_label": ("receptor_label", np.array(["pol_0"])),
            # 'R' = the receptor's circular handedness (RHCP); the correlation product is 'RR'.
            "polarization_type": (("antenna_name", "receptor_label"), np.full((n_ant, 1), "R")),
            "mount": ("antenna_name", np.full(n_ant, "X-Y")),  # nominal; TART elements are fixed
            "station_name": ("antenna_name", np.full(n_ant, station_name)),
            "telescope_name": ("antenna_name", np.full(n_ant, telescope_name)),
        },
        attrs={"type": "antenna", "overall_telescope_name": telescope_name, "relocatable_antennas": False},
    )

    # --- field_and_source ------------------------------------------------------------------
    # TART stares at the local zenith and does not track, so the phase centre is the zenith
    # (phase_elaz = [el=90, az=0]). In ra/dec it is therefore time-dependent (LST-driven) and a
    # single fixed FIELD_PHASE_CENTER_DIRECTION is not well defined, so we record the zenith
    # convention rather than fabricate a sky direction. The full calibrator catalogue (many
    # sources, with rise/set) is a separate sky-model node built downstream.
    phase_elaz = np.asarray(raw.phase_elaz.values, dtype=np.float64)  # [el_deg, az_deg]
    field = xr.Dataset(
        data_vars={
            "PHASE_CENTER_ELAZ": (
                ("field_name", "elaz_label"),
                phase_elaz[None, :],
                {"type": "sky_coord", "units": "deg", "frame": "topocentric-AZEL"},
            ),
        },
        coords={
            "field_name": ("field_name", np.array(["zenith"])),
            "source_name": ("field_name", np.array(["zenith"])),
            "elaz_label": ("elaz_label", np.array(["el", "az"])),
        },
        attrs={"type": "field_and_source", "phase_center_convention": "local zenith (no tracking)"},
    )

    # --- gain_xds (non-standard) -----------------------------------------------------------
    # TART's own gain solution: one snapshot per file (per antenna), retained as the validation
    # oracle and as initialisation means for the IWP-EKF. Dead antennas carry amplitude 0. This
    # node is NOT part of the MSv4 visibility schema.
    gain = (gains_amp * np.exp(1j * gains_phase)).astype(np.complex64)
    gain_xds = xr.Dataset(
        data_vars={
            "GAIN": (("antenna_name",), gain),
            "GAIN_AMPLITUDE": (("antenna_name",), gains_amp.astype(np.float64)),
            "GAIN_PHASE": (("antenna_name",), gains_phase.astype(np.float64), {"units": "rad"}),
            "ANTENNA_FLAG": (("antenna_name",), gains_amp == 0),
        },
        coords={"antenna_name": ("antenna_name", antenna_names)},
        attrs={
            "type": "tart_gain_snapshot",
            "description": "Per-file TART gain solution; validation oracle / EKF initialisation means",
            "gain_time_unix": float(np.median(time_unix)),
        },
    )

    raw.close()

    # Assemble the MSv4 DataTree: one partition node with the standard sub-nodes plus gains.
    partition = "partition_000"
    tree = xr.DataTree.from_dict(
        {
            f"/{partition}": main,
            f"/{partition}/antenna_xds": antenna,
            f"/{partition}/field_and_source_base_xds": field,
            f"/{partition}/gain_xds": gain_xds,
        }
    )
    return tree


def prepare_msv4_zarr(
    hdf_paths: Iterable[Path | str],
    out_zarr: Path | str,
    *,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
):
    """Write the HDF sequence to one imaging-ready zarr; return the output path.

    Args:
        hdf_paths: ordered iterable of TART HDF paths (frames are emitted in this order).
        out_zarr: output zarr path (overwritten if present).
        correct_gains: divide vis/weights by the per-antenna gain product before writing.
        phase_ra_deg: common phase-direction RA (deg, ICRS); stored in attrs (``NaN`` if unset).
        phase_dec_deg: common phase-direction Dec (deg, ICRS); stored in attrs (``NaN`` if unset).
        nframes: optional cap on the total number of frames written (profiling/preview aid).

    Returns:
        The ``out_zarr`` path.

    Raises:
        FileNotFoundError: if no frames are produced.
    """

    out_zarr = Path(out_zarr)
    vis_all: list[np.ndarray] = []
    wgt_all: list[np.ndarray] = []
    brot_all: list[np.ndarray] = []
    bore_all: list[np.ndarray] = []
    time_all: list[np.ndarray] = []
    freqs = None
    info = None

    for path in hdf_paths:
        if nframes is not None and sum(v.shape[0] for v in vis_all) >= nframes:
            break
        node = partition_datatree(read_hdf_as_msv4(path))
        main = node.ds
        times = np.asarray(main.time.values)
        antenna = node["antenna_xds"].to_dataset(inherit=False)
        pos = antenna.ANTENNA_POSITION.values  # (n_ant, 3) ITRS
        names = list(antenna.antenna_name.values)
        index = {name: i for i, name in enumerate(names)}
        a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
        a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
        bl = np.asarray(np.asarray(pos[a1] - pos[a2]))  # (nbl, 3) host
        vis = np.asarray(main.VISIBILITY.values)[..., 0]  # (n_time, nbl, nchan)
        wgt = np.asarray(main.WEIGHT.values)[..., 0]
        if freqs is None:
            freqs = np.asarray(main.frequency.values)
            info = main.attrs["observation_info"]
        if correct_gains:
            vis, wgt = correct_file_gains(node, vis, wgt, xp=np)
        b_rot = equatorial_baselines(bl, times, xp=np)  # (n_time, nbl, 3)
        obs = main.attrs["observation_info"]  # site is shared across files; use this file's times
        boresight = zenith_icrs_vectors(
            times, obs["site_latitude_deg"], obs["site_longitude_deg"], obs["site_altitude_m"]
        )  # (n_time, 3) ICRS zenith unit vectors -> Airy beam boresight
        vis_all.append(np.asarray(vis))
        wgt_all.append(np.asarray(wgt))
        brot_all.append(np.asarray(b_rot))
        bore_all.append(np.asarray(boresight))
        time_all.append(times)

    if not vis_all:
        raise FileNotFoundError("no HDF frames to prepare")

    vis_c = np.concatenate(vis_all, axis=0)
    wgt_c = np.concatenate(wgt_all, axis=0)
    brot = np.concatenate(brot_all, axis=0)
    bore = np.concatenate(bore_all, axis=0)
    tt = np.concatenate(time_all, axis=0)
    if nframes is not None:
        vis_c, wgt_c, brot, bore, tt = (
            vis_c[:nframes],
            wgt_c[:nframes],
            brot[:nframes],
            bore[:nframes],
            tt[:nframes],
        )

    ds = xr.Dataset(
        data_vars={
            "VISIBILITY": (("time", "baseline", "frequency"), vis_c.astype(np.complex64)),
            "WEIGHT": (("time", "baseline", "frequency"), wgt_c.astype(np.float32)),
            "B_ROT": (("time", "baseline", "xyz"), brot.astype(np.float64)),
            "BORESIGHT": (("time", "xyz"), bore.astype(np.float64)),
        },
        coords={
            "time": ("time", tt.astype(np.float64)),
            "frequency": ("frequency", np.asarray(freqs, dtype=np.float64)),
            "xyz": ("xyz", np.array(["x", "y", "z"])),
        },
        attrs={
            "phase_ra_deg": float("nan") if phase_ra_deg is None else float(phase_ra_deg),
            "phase_dec_deg": float("nan") if phase_dec_deg is None else float(phase_dec_deg),
            "site_latitude_deg": float(info["site_latitude_deg"]),
            "site_longitude_deg": float(info["site_longitude_deg"]),
            "site_altitude_m": float(info["site_altitude_m"]),
        },
    )
    if out_zarr.exists():
        shutil.rmtree(out_zarr)
    ds.to_zarr(out_zarr, mode="w")
    return out_zarr
