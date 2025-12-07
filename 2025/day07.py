import itertools as it
from collections import Counter
import operator as op
import re
import functools as ft
from typing import Any, Generator, Iterable

import aoc

def main():
    print(num_splits())
    print(num_timelines())


def num_splits(inp: str | None = None):
    lines = inp.splitlines() if inp else aoc.input_lines()
    incoming = [i for i, c in enumerate(lines[0]) if c == 'S']
    manifold = [
        {i for i, c in enumerate(line) if c == '^'}
        for line in lines[1:]
    ]

    num = 0
    inc = incoming[:]
    for row in manifold:
        num += sum(1 for i in inc if i in row)
        inc = set(split(inc, row))

    return num


def num_timelines(inp: str | None = None):
    lines = inp.splitlines() if inp else aoc.input_lines()
    first, *rest = lines[::2]
    incoming = Counter(i for i, c in enumerate(first) if c == 'S')
    manifold = [
        {i for i, c in enumerate(line) if c == '^'}
        for line in rest
    ]

    for splitters in manifold:
        for i in splitters:
            if num := incoming[i]:
                del incoming[i]
                incoming[i - 1] += num
                incoming[i + 1] += num

    return sum(incoming.values())


def split(incoming: Iterable[int], splitters: set[int]):
    return it.chain.from_iterable(([i - 1, i + 1] if i in splitters else [i] for i in incoming))


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    .......S.......
    ...............
    .......^.......
    ...............
    ......^.^......
    ...............
    .....^.^.^.....
    ...............
    ....^.^...^....
    ...............
    ...^.^...^.^...
    ...............
    ..^...^.....^..
    ...............
    .^.^.^.^.^...^.
    ...............
""")


def test_split():
    assert set(split([5], {5})) == set([4, 6])
    assert set(split([5], {4})) == set([5])
    assert set(split([5], set())) == set([5])
    assert set(split([5, 7], {5, 7})) == set([4, 6, 8])
    assert set(split([5, 7], {5})) == set([4, 6, 7])


def test_num_splits():
    assert num_splits(example) == 21


def test_num_timelines():
    assert num_timelines(example) == 40
