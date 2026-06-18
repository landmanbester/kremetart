import numpy as np

from kremetart.utils.gains import apply_inverse_gains


def correct_file_gains(node, vis, wgt, *, xp=np):
    """Divide a file's vis/weight by the per-antenna gain product (``gain_xds.GAIN``).

    Maps each baseline to its two antenna gains the same way :func:`itrs_baselines` maps
    antennas, then delegates to :func:`kremetart.utils.gains.apply_inverse_gains`. The gain
    snapshot is per-file (time-independent), so this runs once before the sub-integration loop.
    """

    antenna = node["antenna_xds"].to_dataset(inherit=False)
    index = {name: i for i, name in enumerate(antenna.antenna_name.values)}
    a1 = np.array([index[n] for n in node.ds.baseline_antenna1_name.values])
    a2 = np.array([index[n] for n in node.ds.baseline_antenna2_name.values])
    gains = node["gain_xds"].to_dataset(inherit=False).GAIN.values
    return apply_inverse_gains(vis, wgt, gains, a1, a2, xp=xp)
