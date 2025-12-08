import itertools as it
from collections import Counter
import operator as op
import re
import functools as ft
from typing import Any, Generator, Iterable

import aoc

type Coord = tuple[int, int, int]


def main():
    print(connect_junctions(parsed(), 1_000))
    print(connect_junctions(parsed()))


def connect_junctions(boxes: list[Coord], n: int = -1):
    dists = sorted(it.combinations(boxes, 2), key=lambda p: disty(*p))
    circuits = {c: {c} for c in boxes}

    def count():
        return Counter({id(c): len(c) for c in circuits.values()}.values())

    for i in it.count():
        a, b = dists[i]
        if not circuits[a] is circuits[b]:
            for c in (connected := circuits[a] | circuits[b]):
                circuits[c] = connected

        if i + 1 == n:
            break
        elif len(count()) == 1:
            return a[0] * b[0]

    return int(ft.reduce(op.mul, sorted(count())[-3:]))


def disty(a: Coord, b: Coord):
    (i, j, k), (x, y, z) = a, b
    return (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2


def parsed(inp: str | None = None) -> list[Coord]:
    lines = inp.splitlines() if inp else aoc.input_lines()
    return [
        (x, y, z) for line in lines for x, y, z in [map(int, line.split(','))]
    ]


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    162,817,812
    57,618,57
    906,360,560
    592,479,940
    352,342,300
    466,668,158
    542,29,236
    431,825,988
    739,650,466
    52,470,668
    216,146,977
    819,987,18
    117,168,530
    805,96,715
    346,949,466
    970,615,88
    941,993,340
    862,61,35
    984,92,344
    425,690,689
""")


def test_example():
    assert connect_junctions(parsed(example), 10) == 40
    assert connect_junctions(parsed(example)) == 25272
