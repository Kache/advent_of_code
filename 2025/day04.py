import itertools as it
from typing import Any, Generator

import aoc

type Grid = list[list[str]]
type Y = int
type X = int
type Coord = tuple[Y, X]


def main():
    print(num_accessible(parsed()))

    print(num_removed(parsed()))


def at(grid: Grid, c: Coord):
    return grid[c[0]][c[1]]

def up(c: Coord) -> Coord:
    return (c[0] - 1, c[1])

def down(c: Coord) -> Coord:
    return (c[0] + 1, c[1])

def left(c: Coord) -> Coord:
    return (c[0], c[1] - 1)

def right(c: Coord) -> Coord:
    return (c[0], c[1] + 1)

def box(c: Coord) -> list[Coord]:
    return [b for b in square(c) if b != c]

def square(c: Coord) -> list[Coord]:
    return [h for v in [up(c), c, down(c)] for h in [left(v), v, right(v)]]

def coords(grid: Grid) -> Generator[Coord, Any, None]:
    for j, row in enumerate(grid):
        for i, col in enumerate(row):
            yield (j, i)


def num_accessible(grid: Grid):
    rolls = {c for c in coords(grid) if at(grid, c) == '@'}

    return sum(
        1 for r in rolls
        if sum(1 for b in box(r) if b in rolls) < 4
    )


def num_removed(grid: Grid):
    rolls = {c for c in coords(grid) if at(grid, c) == '@'}
    orig_num = len(rolls)

    while True:
        to_remove = [
            r for r in rolls
            if sum(1 for b in box(r) if b in rolls) < 4
        ]
        if to_remove:
            for r in to_remove:
                rolls.remove(r)
        else:
            break

    return orig_num - len(rolls)


def parsed(inp: str | None = None):
    inp = inp or aoc.input_str()
    return [list(line) for line in inp.splitlines()]


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    ..@@.@@@@.
    @@@.@.@.@@
    @@@@@.@.@@
    @.@@@@..@.
    @@.@@@@.@@
    .@@@@@@@.@
    .@.@.@.@@@
    @.@@@.@@@@
    .@@@@@@@@.
    @.@.@@@.@.
""")


def test_accessible():
    assert num_accessible(parsed(example)) == 13


def test_removed():
    assert num_removed(parsed()) == 43
