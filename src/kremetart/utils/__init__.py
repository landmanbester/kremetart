"""Utils for kremetart."""

import datetime


def partition_datatree(dt):
    return dt[list(dt.children)[0]]


def unix_to_utc(unix_seconds) -> str:
    dt = datetime.datetime.fromtimestamp(float(unix_seconds), tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
