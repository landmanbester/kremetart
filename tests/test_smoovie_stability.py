"""A fixed celestial source lands in the same HEALPix pixel across all frames (sidereally-fixed grid)."""

import numpy as np

from kremetart.utils import partition_datatree
from kremetart.utils.healpix_dft import image_frame, make_pixel_grid
from kremetart.utils.read_tart_hdf import read_hdf_as_msv4
from kremetart.utils.rephasing import itrs_baselines
from tests.accuracy_helpers import (
    recovered_direction_and_flux,
    simulate_visibilities,
    source_svec,
    sources_spanning_zenith,
)

NSIDE = 64


def test_steady_source_holds_one_pixel_across_frames(hdf_paths):
    import healpy as hp

    paths = hdf_paths
    pix = make_pixel_grid(NSIDE, xp=np)

    # A fixed celestial source ~60 deg elevation at the first frame's mid time.
    first = partition_datatree(read_hdf_as_msv4(paths[0]))
    info = first.ds.attrs["observation_info"]
    site = (info["site_latitude_deg"], info["site_longitude_deg"], info["site_altitude_m"])
    t0 = np.asarray(first.ds.time.values)
    ra, dec = sources_spanning_zenith(t0[t0.size // 2 : t0.size // 2 + 1], *site, els_deg=[60.0])
    svec = source_svec(ra, dec)

    peak_pixels = []
    for path in paths:
        node = partition_datatree(read_hdf_as_msv4(path))
        times = np.asarray(node.ds.time.values)
        tmid = times[times.size // 2 : times.size // 2 + 1]
        bl = itrs_baselines(node, np)
        freqs = np.asarray(node.ds.frequency.values)
        vis = simulate_visibilities([1.0], svec, bl, tmid, freqs)
        dmap = image_frame(vis, np.ones_like(vis.real), tmid, bl, pix, freqs)
        rec, _ = recovered_direction_and_flux(dmap, pix, NSIDE, near=svec[0], search_radius_deg=10.0)
        peak_pixels.append(int(hp.vec2pix(NSIDE, *rec, nest=True)))

    assert len(set(peak_pixels)) == 1  # identical pixel in every frame
