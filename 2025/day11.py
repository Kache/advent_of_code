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


def main():
    print(num_paths('you'))
    print(num_paths('svr', ('dac', 'fft')))


@ft.cache
def num_paths(curr: str, req_visits: tuple[str, ...] = tuple(), inp=None):
    if curr == "out":
        return int(not req_visits)
    req_visits = tuple(n for n in req_visits if n != curr)
    return sum(num_paths(n, req_visits, inp) for n in parsed(inp)[curr])


def parsed(inp: str | None = None):
    lines = inp.splitlines() if inp else aoc.input_lines()

    def parse(line: str):
        device, *outputs = line.split(' ')
        return (device[:-1], outputs)

    return dict(parse(line) for line in lines)


if __name__ == "__main__":
    main()


import pytest

you_example = aoc.heredoc("""
    aaa: you hhh
    you: bbb ccc
    bbb: ddd eee
    ccc: ddd eee fff
    ddd: ggg
    eee: out
    fff: out
    ggg: out
    hhh: ccc fff iii
    iii: out
""")

svr_example = aoc.heredoc("""
    svr: aaa bbb
    aaa: fft
    fft: ccc
    bbb: tty
    tty: ccc
    ccc: ddd eee
    ddd: hub
    hub: fff
    eee: dac
    dac: fff
    fff: ggg hhh
    ggg: out
    hhh: out
""")

def test_num_paths():
    assert num_paths('you', inp=you_example) == 5
    assert num_paths('svr', ('dac', 'fft'), svr_example) == 2
