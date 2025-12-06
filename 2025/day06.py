import itertools as it
import operator as op
import re
import functools as ft
from typing import Any, Generator, Iterable

import aoc

ops = {
    '+': op.add,
    '*': op.mul,
}


def main():
    print(solve_problems(parsed()))
    print(solve_rtl_col(parsed()))


def solve_problems(lines: list[str]):
    split_lines = (re.sub(' +', ' ', line).strip().split(' ') for line in lines)

    def parse(strs: tuple[str]):
        return [int(n) for n in strs[:-1]], str(strs[-1])

    problems = [parse(line) for line in zip(*split_lines)]

    return sum(ft.reduce(ops[o], nums) for nums, o in problems)


def solve_rtl_col(lines: list[str]):
    rtl_col_scan = [''.join(split).strip() for split in zip(*(reversed(line) for line in lines))]

    def parse(strs: list[str]):
        nums = it.chain(strs[:-1], [strs[-1][:-1]])
        return list(map(int, nums)), strs[-1][-1]

    problems = [parse(list(grp)) for k, grp in it.groupby(rtl_col_scan, key=bool) if k]

    return sum(ft.reduce(ops[o], nums) for nums, o in problems)


def parsed(inp: str | None = None):
    return inp.splitlines() if inp else aoc.input_lines()


if __name__ == "__main__":
    main()


import pytest

example = (
    '123 328  51 64 \n'
    ' 45 64  387 23 \n'
    '  6 98  215 314\n'
    '*   +   *   +  \n'
)


def test_solve():
    assert solve_problems(parsed(example)) == 4277556


def test_solve_rtl_col():
    assert solve_rtl_col(parsed(example)) == 3263827
