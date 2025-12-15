from enum import Enum, StrEnum
from uuid import uuid4
from collections import deque
import dataclasses as dc
import itertools as it
from collections import Counter
import operator as op
import re
import functools as ft
from typing import Any, Callable, Generator, Iterable, Iterator, Literal, NewType, Self, Sequence, TypeVar, overload
from numbers import Number

import aoc

T = TypeVar('T')
type PresShape = tuple[tuple[bool, bool, bool], tuple[bool, bool, bool], tuple[bool, bool, bool]]
type WidLen = tuple[int, int]


def main():
    print(num_fit())


def num_fit(inp: str | None = None):
    return sum(1 for wl, s in parsed(inp) if can_raw_area_fit(wl, s))


def parsed(inp: str | None = None):
    inp = inp or aoc.input_str()
    *pres_grid, treespaces = inp.split('\n\n')

    def parse_shape(pg: str) -> tuple[int, PresShape]:
        idx, *grid = pg.splitlines()
        return int(idx.removesuffix(':')), three(three(v == '#' for v in r) for r in grid)

    pres_shapes = dict(parse_shape(pg) for pg in pres_grid)

    def parse_ts(ts: str) -> tuple[WidLen, Counter[PresShape]]:
        dims, pres_counts = ts.split(':')
        w, l = map(int, dims.split('x'))
        return (w, l), Counter({pres_shapes[i]: int(c) for i, c in enumerate(pres_counts.strip().split(' '))})

    return [parse_ts(ts) for ts in treespaces.splitlines()]


def can_raw_area_fit(wl: WidLen, shapes: Counter[PresShape]):
    w, l = wl
    shape_areas = (sum(map(sum, shape)) * n for shape, n in shapes.items())
    return sum(shape_areas) <= w * l


def three(itr: Iterable[T]):
    a, b, c = itr
    return a, b, c


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
""")

def test_example():
    assert num_fit(example) == 2
