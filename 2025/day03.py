import itertools as it
import aoc


def main():
    print(sum(map(max_jolt2, aoc.input_lines(3))))
    print(sum(map(max_jolt12, aoc.input_lines(3))))


def max_jolt2(batt_bank: str):
    idx, n = max(enumerate(batt_bank[:-1]), key=lambda p: p[1])
    _, m = max(enumerate(batt_bank[idx + 1:]), key=lambda p: p[1])
    return int(f"{n}{m}")


def max_jolt12(batt_bank: str):
    digits = list(enumerate(batt_bank, start=1))
    offset = len(digits) + 1 - 12

    def max_digits():
        idx = 0
        for end in range(12):
            candidates = digits[idx:offset + end]
            idx, d = max(candidates, key=lambda p: p[1])
            yield d

    return int(''.join(max_digits()))


if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    987654321111111
    811111111111119
    234234234234278
    818181911112111
""").splitlines()


@pytest.mark.parametrize(['batt_bank', 'exp'], [
    ('987654321111111', 98),
    ('811111111111119', 89),
    ('234234234234278', 78),
    ('818181911112111', 92),
])
def test_joltage(batt_bank, exp):
    assert max_jolt2(batt_bank) == exp


@pytest.mark.parametrize(['batt_bank', 'exp'], [
    ('987654321111111', 987654321111),
    ('811111111111119', 811111111119),
    ('234234234234278', 434234234278),
    ('818181911112111', 888911112111),
])
def test_jolt12(batt_bank, exp):
    assert max_jolt12(batt_bank) == exp


def test_example():
    assert sum(map(max_jolt2, example)) == 357
