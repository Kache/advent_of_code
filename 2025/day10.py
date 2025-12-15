import functools as ft
import itertools as it
import operator as op
from math import gcd
from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

import aoc
from matrix import Fraction, Matrix, Num

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT


type Lights = list[bool]
type Button = set[int]
type Joltage = tuple[int, ...]
type Machine = tuple[Lights, list[Button], Joltage]
T = TypeVar('T')


def main():
    print(sum_toggles(parsed()))
    print(sum_jolts(parsed()))


def sum_toggles(machines: list[Machine]):
    return sum(num_presses(lights, btns) or 0 for lights, btns, _ in machines)


def sum_jolts(machines: list[Machine]):
    return sum(joltage_cost(btns, joltage) for _, btns, joltage in machines)


def parsed(inp: str | None = None):
    lines = inp.splitlines() if inp else aoc.input_lines()
    return [parse(line) for line in lines]


def parse(line: str) -> Machine:
    light_str, *btns_str, jolt_str = line.split(' ')
    return (
        [s == '#' for s in light_str[1:-1]],
        [set(map(int, s[1:-1].split(','))) for s in btns_str],
        tuple(map(int, jolt_str[1:-1].split(','))),
    )


def num_presses(lights: Lights, buttons: list[Button]):
    def press(btns: tuple[Button, ...]) -> Joltage:
        return tuple(sum(i in b for b in btns) for i in range(len(lights)))

    def pattern(jolts: Joltage) -> Lights:
        return [bool(n % 2) for n in jolts]

    for n in range(len(buttons) + 1):
        for combo in it.combinations(buttons, n):
            if pattern(press(combo)) == lights:
                return n


def joltage_cost(buttons: list[Button], joltage: Joltage):
    def groupby(itr: Iterable[T], key: Callable[[T], 'SupportsRichComparisonT']):
        return {k: list(v) for k, v in it.groupby(sorted(itr, key=key), key=key)}

    def sub_halve(j_a: Joltage, j_b: Joltage) -> Joltage:
        return tuple((a - b) // 2 for a, b, in zip(j_a, j_b))

    def press(btns: tuple[Button, ...]) -> Joltage:
        return tuple(sum(i in b for b in btns) for i in range(len(joltage)))

    def pattern(jolts: Joltage) -> Joltage:
        return tuple(n % 2 for n in jolts)

    all_btn_combos = (combo for n in range(len(buttons) + 1) for combo in it.combinations(buttons, n))
    press_patterns = groupby(all_btn_combos, lambda btns: pattern(press(btns)))

    @ft.cache
    def cost(jolts: Joltage) -> int:
        if not any(jolts):
            return 0
        elif any(j < 0 for j in jolts) or pattern(jolts) not in press_patterns:
            return sum(joltage)
        else:
            btn_combos = press_patterns[pattern(jolts)]
            return min(len(btns) + 2 * cost(sub_halve(jolts, press(btns))) for btns in btn_combos)

    return cost(joltage)


def as_introw(row: list[Num]) -> list[int]:
    denoms = [Fraction(v).denominator for v in row]
    mul = ft.reduce(op.mul, denoms)
    res = [int(v * mul) for v in row]
    div = gcd(*res)
    return [v // div for v in res]


def joltage_mtx(buttons: list[Button], joltage: Joltage):
    m = Matrix([
        [int(i in btn) for btn in buttons]
        for i in range(len(joltage))
    ])
    jolt = Matrix([[j] for j in joltage])
    bounds = [min(int(jolt[i][0]) for i in btn) for btn in buttons]

    sys_eq = m.chain(jolt)
    # reduced = sys_eq.reduce()
    # reduced = Matrix([as_introw(row) for row in reduced])  # v4+
    reduced = sys_eq.reduce_int()

    # move free variables to the end (reorder buttons)
    for i in range(reduced.h):
        if reduced[i, i] != 1:
            n = next(j for j in range(i, m.w) if reduced[i, j])
            reduced = reduced.swap_col(i, n)
            buttons[i], buttons[n] = buttons[n], buttons[i]
            bounds[i], bounds[n] = bounds[n], bounds[i]
            m = m.swap_col(i, n)

    free_vars = bounds[reduced.h:m.w]
    free_cols = reduced.subm((0, reduced.h), (reduced.h, m.w))

    def btn_presses(cand_fvars: tuple[int, ...]):
        for i in range(free_cols.h):
            matmul = sum(s * o for s, o in zip(free_cols._mat[i], cand_fvars))
            diff = reduced._mat[i][-1] - matmul
            presses, rem = divmod(diff, int(reduced._mat[i][i]))
            if diff < 0 or rem:
                yield -2**63
                return
            yield int(presses)

        for presses in cand_fvars:
            yield presses

    def solutions():
        if free_vars:
            free_var_candidates = it.product(*(range(v + 1) for v in free_vars))
            # Hot loop
            for cand_fvars in free_var_candidates:

                # v5
                num_presses = sum(btn_presses(cand_fvars))
                if num_presses >= 0:
                    yield num_presses

                # # v4
                # num_presses = sum(cand_fvars)
                # for i in range(free_cols.h):
                #     matmul = sum(s * o for s, o in zip(free_cols._mat[i], cand_fvars))
                #     diff = reduced._mat[i][-1] - matmul
                #     presses, rem = divmod(diff, int(reduced._mat[i][i]))
                #     if diff < 0 or rem:
                #         num_presses = -1
                #         break
                #     num_presses += presses
                # if num_presses >= 0:
                #     yield num_presses

                # # v3
                # num_presses = sum(cand_fvars)
                # for i in range(free_cols.h):
                #     matmul = sum(s * o for s, o in zip(free_cols._mat[i], cand_fvars))
                #     presses = reduced._mat[i][-1] - matmul
                #     if presses < 0 or presses % 1:
                #         num_presses = -1
                #         break
                #     num_presses += presses
                # if num_presses >= 0:
                #     yield num_presses

                # # v2
                # presses = [
                #     reduced._mat[i][-1] - sum(s * o for s, o in zip(free_cols._mat[i], cand_fvars))
                #     for i in range(free_cols.h)
                # ]
                # if not any(presses < 0 or presses % 1 for presses in presses):
                #     yield sum(presses + list(cand_fvars))

                # # v1
                # free_var_contrib = free_cols @ Matrix[cand_fvars].trans()
                # presses = Matrix[*(reduced.tail(1) - free_var_contrib)]
                # if any(presses < 0 or presses % 1 for presses in presses.col(0)):
                #     continue
                # yield sum(presses.reduce_nums().col(0) + list(cand_fvars))

        else:
            num_presses, rem = divmod(sum(reduced.col(-1)), 1)
            assert not rem
            yield int(num_presses)

    return min(solutions())


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    [.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
    [...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
    [.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
""")


@pytest.mark.parametrize(['machine', 'num'], [
    (parsed(example)[0], 2),
    (parsed(example)[1], 3),
    (parsed(example)[2], 2),
])
def test_light_cfg_btns(machine, num):
    assert num_presses(machine[0], machine[1]) == num


def test_as_introw():
    row = [1, 0, 0, -1, Fraction(-1, 2), -10]
    assert as_introw(row) == [2, 0, 0, -2, -1, -20]

    row = [0, Fraction(1, 3), Fraction(-1, 2), -10]
    assert as_introw(row) == [0, 2, -3, -60]


@pytest.mark.parametrize(['machine', 'num'], [
    (parsed(example)[0], 10),
    (parsed(example)[1], 12),
    (parsed(example)[2], 11),
    (parse('[#.##] (1,3) (0,2,3) {0,13,0,13}'), 13),
    (parse('[##..] (0,2) (0,1) (0,3) {22,14,4,4}'), 22),
    (parse('[###.] (1,2,3) (0,1,2) {15,145,145,130}'), 145),
    (parse('[##..] (0,3) (2) (0,2) (1,2) {20,106,123,13}'), 136),
    (parse('[##..] (2,3) (1,2) (1,2,3) (0) {7,147,167,27}'), 174),
    (parse('[..#.###.#.] (0,3,4,5) (1,4,6) (2,9) (0,4) (2,4,7,8) (0,2,3,4,5,6,7,8,9) (1,6) (1,2,5,6,7) (0,4,7,8) (0,1,2,3,5,8,9) (0,3,4,5,6,7,8,9) (4,6,9) {56,51,67,27,82,44,70,56,49,58}'), 132),
    (parse('[.#..#...##] (2,8) (1,2,5,7,8) (0,3,4,6,8) (1,3,4,5,6,7,8,9) (0,5,6,8,9) (1,2,3,4,7,8) (2,7,9) (1,4,5,7) (2,5,6,9) (0,2,3,6) (1,3,6) (0,1,3,4,6,8) {53,39,56,61,47,41,75,47,59,45}'), 113),
    (parse('[.##..] (0,1) (0,1,2,3) (4) (0,1,4) (0,1,2) (2,4) (0,3) {65,57,43,22,34}'), 75),
    (parse('[#.###] (2,3,4) (0,3) (1,4) (2,4) (0,1,3) (0,2,4) (0,4) {57,17,20,43,39}'), 70),
    (parse('[..#...] (1,2,4,5) (0,2) (0,3,4) (0,1,2,3,5) (2,4,5) (1,3) (0,1) {57,55,62,49,48,49}'), 90),
])
def test_joltage_quick(machine, num):
    assert joltage_mtx(machine[1], machine[2]) == joltage_cost(machine[1], machine[2]) == num


@pytest.mark.parametrize(['machine', 'num'], [
    (parse('[.####....#] (0,1,2,4,6,7,9) (0,1,3,4,5,7,8,9) (3,4,6,8) (2,5,6,8) (0,2,3,5,7,8,9) (0,1,5,7,9) (1,2,6) (1,2,3,4,5,9) (0,1,2,4,5,6,8,9) (1,2,4,5,6,9) (1,3) (0,1,2,6,7,8,9) (0,1,2,3,4) {82,113,122,47,76,67,90,53,59,91}'), 135),
    (parse('[..#....###] (1,4,5,7,8,9) (2,3,4,5,7) (0,4) (0,1,6,8) (3,5) (0,2,3,4,5,6,7,8) (0,3,5,6,7,9) (0,1,2,3,5,6,7,8) (1,4,5,6) (0,1,2,3,5,6,8,9) (0,3,5,9) (0,2,3,4,5,7,9) (1,8,9) {78,30,51,90,71,104,54,67,29,51}'), 126),
    (parse('[..#.#.#..#] (1,3,4,5,6,7,8,9) (0,2,3,4,7) (0,1,2,3,7,8,9) (0,1,2,3,4,9) (1,2,3,4,6,7,8) (0,1,2,3,5,6,7,8,9) (2,3,7) (1,2,5) (0,1,2,3,4,5,6,7,8) (1,4,5,8,9) (0,1,2,3,5,6,8,9) (0,1,3,7,8) (0,2,4,6,8,9) {87,134,107,105,80,83,69,86,116,84}'), 136),
    (parse('[.#.#.###.#] (0,5) (0,1,3,4,6,7,8) (0,1,2,5,6,7,8) (0,1,2,6,7,9) (3,4,6,8) (2,3,4,5,6) (0,1,2,3,4,6,8,9) (3,5,7) (1,3,6) (0,1,5,7,8,9) (2,3,5,6,7,8,9) (1,4) (4,8,9) {61,79,46,93,91,61,86,56,71,37}'), 129),
])
def test_joltage_slow(machine, num):
    assert joltage_mtx(machine[1], machine[2]) == joltage_cost(machine[1], machine[2]) == num
