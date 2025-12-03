import itertools as it
import operator as op
import functools as ft
from typing import Callable

import aoc


def main():
    print(sum_invalid_ids(parsed()))
    print(sum_invalid_ids(parsed(), pred=valid2))


def sum_invalid_ids(ranges: list[range], pred: Callable[[int], bool] | None = None):
    pred = pred or valid1
    return sum(n for rng in ranges for n in rng if not pred(n))


def valid1(product_id: int):
    id_str = str(product_id)
    is_odd = bool(len(id_str) % 2)
    return is_odd or (
        id_str[:len(id_str) // 2] != id_str[len(id_str) // 2:]
    )


def all_same(nums: list[int]):
    return ft.reduce(lambda a, b: b if a == b else None, nums)


def valid2(product_id: int):
    return not any(all_same(batch) for batch in batches(product_id))


def batches(product_id: int):
    id_str = str(product_id)

    for sublen in range(len(id_str) - 1, 0, -1):
        if len(id_str) % sublen == 0:
            yield [int(''.join(b)) for b in it.batched(id_str, sublen)]


def parsed(inp: str | None = None):
    inp = inp or aoc.input_str(2)
    def parse_range(s: str):
        lo, hi = s.split('-')
        return range(int(lo), int(hi) + 1)

    return [parse_range(rng) for rng in inp.split(',')]


if __name__ == "__main__":
    main()


import pytest

example = (
    '11-22,95-115,998-1012,1188511880-1188511890,222220-222224,'
    '1698522-1698528,446443-446449,38593856-38593862,565653-565659,'
    '824824821-824824827,2121212118-2121212124'
)

def test_all_same():
    assert all_same([12, 12])
    assert all_same([12, 12, 12])
    assert all_same([123, 123, 123])
    assert all_same([1, 1, 1])

    assert not all_same([12, 123])
    assert not all_same([12, 12, 13])
    assert not all_same([123, 123, 12])
    assert not all_same([1, 1, 2])


def test_batches():
    assert list(batches(1234)) == [
        [12, 34], [1, 2, 3, 4]
    ]
    assert list(batches(123456)) == [
        [123, 456], [12, 34, 56], [1, 2, 3, 4, 5, 6]
    ]


def test_valid1():
    assert valid1(101)
    assert not valid1(11)
    assert not valid1(22)
    assert not valid1(1010)
    assert not valid1(1188511885)


def test_valid2():
    assert not valid2(824824824)


def test_example():
    assert sum_invalid_ids(parsed(example)) == 1227775554
    assert sum_invalid_ids(parsed(example), pred=valid2) == 4174379265
