import itertools as it
import functools as ft
from typing import Any, Generator

import aoc

type Grid = list[list[str]]
type Y = int
type X = int
type Coord = tuple[Y, X]


def main():
    print(num_available_fresh(*parsed()))

    print(num_fresh(*parsed()))


def merge_ranges(ranges: list[range]):
    merged: list[range] = []

    for r in sorted(ranges, key=lambda r: (r.start, r.stop)):
        if not merged or merged[-1].stop < r.start:
            merged.append(r)
        elif merged[-1].stop < r.stop:
            merged[-1] = range(merged[-1].start, r.stop)

    return merged


def num_fresh(ranges: list[range], _: list[int]):
    return sum(rng.stop - rng.start + 1 for rng in merge_ranges(ranges))


def num_available_fresh(ranges: list[range], ingredients: list[int]):
    return sum(1 for i in ingredients if any(covers(r, i) for r in ranges))


def covers(rng: range, n: int):
    return rng.start <= n <= rng.stop


def parsed(inp: str | None = None):
    inp = inp or aoc.input_str()
    fresh_id_ranges, available_ingredients = inp.split('\n\n')
    return (
        [
            range(int(lo), int(hi))
            for lo, hi in (p.split('-') for p in fresh_id_ranges.splitlines())
        ],
        [int(i) for i in available_ingredients.splitlines()],
    )


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    3-5
    10-14
    16-20
    12-18

    1
    5
    8
    11
    17
    32
""")


def test_num_available_fresh():
    ranges, ingredients = parsed(example)
    assert num_available_fresh(ranges, ingredients) == 3


def test_num_fresh():
    ranges, ingredients = parsed(example)
    assert num_fresh(ranges, ingredients) == 14


def test_merge_ranges():
    assert merge_ranges([
        range(3, 5),
        range(5, 8),
    ]) == [range(3, 8)]

    assert merge_ranges([
        range(3, 5),
        range(6, 8),
    ]) == [
        range(3, 5),
        range(6, 8),
    ]

    assert merge_ranges([
        range(3, 8),
        range(4, 8),
    ]) == [
        range(3, 8),
    ]
