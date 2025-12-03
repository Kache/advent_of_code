import aoc
from aoc import heredoc


def main():
    lines = aoc.input_lines()
    print(password1(lines))
    print(password2(lines))


def password1(rotations: list[str], start: int = 50):
    num = 0
    curr = start
    for rot in rotations:
        curr = turn(curr, rot)
        if curr == 0:
            num += 1

    return  num


def password2(rotations: list[str], start: int = 50):
    num = 0
    curr = start
    for rot in rotations:
        curr, n = turn2(curr, rot)
        num += n

    return  num


def turn(at: int, n: int | str):
    if isinstance(n, str):
        sign = -1 if n[0] == 'L' else 1
        n = int(n[1:]) * sign

    return (at + n) % 100


def turn2(at: int, dist: int | str):
    if isinstance(dist, str):
        signs = {'L': -1, 'R': 1}
        dist = int(dist[1:]) * signs[dist[0]]

    if dist < 0:  # left
        mirror = (100 - at) % 100
        mirror_at, crosses0 = turn2(mirror, -dist)
        ending_num = (100 - mirror_at) % 100
    else:  # right
        div, mod = divmod(at + dist, 100)
        ending_num = mod
        crosses0 = div

    return ending_num, crosses0


if __name__ == "__main__":
    main()


import pytest

example = heredoc("""
    L68
    L30
    R48
    L5
    R60
    L55
    L1
    L99
    R14
    L82
""").splitlines()


def test_example1():
    assert password1(example) == 3


def test_example2():
    assert password2(example) == 6


def test_basic():
    assert turn(11, 8) == 19
    assert turn(19, -19) == 0
    assert turn(0, -1) == 99
    assert turn(99, 1) == 0
    assert turn(5, -10) == 95
    assert turn(95, 5) == 0


@pytest.mark.parametrize(['curr', 'rot', 'expected'], [
    (11, 8,       (19, 0)),
    (50, 'R1000', (50, 10)),
    (99, 1,       (0,  1)),
    (95, 5,       (0,  1)),
    (95, 10,      (5,  1)),
    (95, 110,     (5,  2)),
])
def test_turn_right(curr, rot, expected):
    assert turn2(curr, rot) == expected


@pytest.mark.parametrize(['curr', 'rot', 'expected'], [
    (10, -5,     (5, 0)),
    (0,  -1,      (99, 0)),
    (19, -19,     (0,  1)),
    (5,  -10,     (95, 1)),
])
def test_turn_left(curr, rot, expected):
    assert turn2(curr, rot) == expected
