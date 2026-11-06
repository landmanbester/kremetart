"""Host-side helpers: sky geometry, HDF preparation, calibration and gains, satellites, the map store and
the live viewer."""


def partition_datatree(dt):
    return dt[list(dt.children)[0]]
