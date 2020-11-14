import itertools
from typing import Iterable, Tuple, TypeVar

_T = TypeVar("_T")


def pairwise(iterable: Iterable[_T]) -> Iterable[Tuple[_T, _T]]:
    "(s0, s1, s2, s3, ...) -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
