import itertools
from typing import Iterable, Tuple, TypeVar

_T = TypeVar("_T")


def pairwise(iterable: Iterable[_T]) -> Iterable[Tuple[_T, _T]]:
    "(s0, s1, s2, s3, ...) -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_region_coords(region: str):
    if region.lower() == "cr":
        region_coords = ["-83.812_10.39"]
    elif region.lower() == "in":
        region_coords = ["73.125_18.8143"]
    elif region.lower() == "crin":
        region_coords = ["-83.812_10.39", "73.125_18.8143"]
    else:
        raise AssertionError("--region must be 'cr', 'in' or 'crin'")

    return region_coords
