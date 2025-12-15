from enum import Enum, StrEnum
from math import gcd
from fractions import Fraction
from collections import deque
import dataclasses as dc
import itertools as it
from collections import Counter
import operator as op
import re
import functools as ft
from typing import Any, Callable, Generator, Iterable, Iterator, Literal, NewType, Self, Sequence, SupportsIndex, TypeVar, overload
from numbers import Number


type Num = int | float | Fraction


def all_equal(itr: Iterable):
    gb = it.groupby(itr)
    return next(gb, True) and not next(gb, False)


class MatrixMeta(type):
    def __getitem__(self, rows: tuple[Sequence[Num], ...] | Sequence[Num]) -> 'Matrix':
        if rows and not isinstance(rows[0], Sequence):  # arg is single row
            return self([rows])
        return self(rows)


class Matrix(metaclass=MatrixMeta):
    def __init__(self, rows: Iterable[Iterable[Num]]):
        self._mat = [list(row) for row in rows]
        if not all_equal(map(len, self._mat)):
            raise ValueError(f"Unequal row lengths: {list(map(len, self._mat))}")

    def __len__(self):
        return len(self._mat)

    def __iter__(self):
        return (r[:] for r in self._mat)

    def __eq__(self, other: object):
        match other:
            case Matrix():
                return self._mat == other._mat
            case list():
                return self._mat == other
            case _:
                return False

    def __str__(self):
        return 'Matrix' + str(self._mat)

    def __repr__(self):
        return '\n'.join([
            'Matrix[',
            *(f"    {str(row)}," for row in self),
            ']',
        ])

    @property
    def height(self):
        return len(self)

    @property
    def width(self):
        return len(self[0]) if self else 0

    @property
    def square(self):
        return self.height == self.width

    def row(self, i: int):
        return self[i]

    def col(self, j: int):
        return list(self._col(j))

    def _col(self, j: int):
        return (r[j] for r in self)

    def rows(self):
        return iter(self)

    def cols(self):
        for j in range(self.width):
            yield self.col(j)

    @classmethod
    def identity(cls, n: int):
        return cls([
            [int(j == i) for j in range(n)]
            for i in range(n)
        ])

    @overload
    def __getitem__(self, i: SupportsIndex) -> list[Num]: ...
    @overload
    def __getitem__(self, i: slice) -> list[list[Num]]: ...
    @overload
    def __getitem__(self, i: tuple[int, int]) -> Num: ...

    def __getitem__(self, i: SupportsIndex | slice | tuple[int, int]):
        if isinstance(i, tuple):
            return self._mat[i[0]][i[1]]
        elif isinstance(i, slice):
            return [list(row) for row in self._mat[i]]
        else:
            return list(self._mat[i])

    def transpose(self):
        return type(self)(list(map(list, zip(*self))))

    def map(self, func: Callable[[Num], Num]):
        return type(self)([
            [func(v) for v in row]
            for row in self
        ])

    def _op(self, func: Callable[[Num, Num], Num], other: Self):
        if (self.w != other.w or self.h != other.h):
            raise ValueError(f"mismatching dimensions {self.w=} {self.h=} {other.w=} {other.h=}")

        return Matrix([
            [func(self[i, j], other[i, j]) for j in range(self.w)]
            for i in range(self.h)
        ])

    def __sub__(self, other: Self):
        return self._op(op.sub, other)

    def __mul__(self, n: Num):
        return self.map(lambda v: n * v)

    def __rmul__(self, other: Num):
        return self * other

    def __matmul__(self, other: Self) -> Self:
        if self.width != other.height:
            raise ValueError(f"Can't matmul {self.width=} and {other.height=}")

        rows = [self._mulrow(j, other) for j in range(self.h)]
        return type(self)(rows)

    def _mulrow(self, j: int, other: Self):
        return [
            sum(s * o for s, o in zip(self.row(j), other._col(i)))
            for i in range(other.w)
        ]

    def minor(self, i: int, j: int):
        return Matrix([
            [v for jj, v in enumerate(row) if jj != j]
            for ii, row in enumerate(self) if ii != i
        ]).determinant()

    def cofactor(self, i: int, j: int):
        return (-1)**(i + j) * self.minor(i, j)

    def determinant(self):
        if not (m := self).square:
            raise ValueError(f"Matrix not square {m.h=} {m.w=}")
        elif m.h < 2:
            return m[0, 0] if m.h else 1
        elif m.h == 2:
            (a, b), (c, d) = m
            return a * d - b * c
        elif m.h == 3:
            (a, b, c), (d, e, f), (g, h, i) = m
            return a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

        return sum(m[0, j] * m.cofactor(0, j) for j in range(m.h))

    def inverse(self):
        if not (m := self).square:
            raise ValueError(f"Matrix not square {m.h=} {m.w=}")

        return m.chain(type(m).identity(m.h)).reduce().tail(m.h).reduce_nums()

        comatrix_transpose = type(m)([
            [m.cofactor(j, i) for j in range(m.w)]
            for i in range(m.h)
        ])
        det = m.det()
        coeff = 1 / det if isinstance(det, float) else Fraction(1, det)
        return coeff * comatrix_transpose

    def reduce_nums(self):
        return self.map(reduce_num)

    def chain(self, m: Self):
        if self.h != m.h:
            raise ValueError(f"Cannot chain mismatching heights {self.h=} {m.h=}")

        return type(self)(
            it.chain(row_a, row_b)
            for row_a, row_b in zip(self, m)
        )

    @overload
    def reduce_row(self, i: int, scale: Num, /) -> Self: ...
    @overload
    def reduce_row(self, a_i: int, scale: None, b_i: int, /) -> Self: ...
    @overload
    def reduce_row(self, mul_i: int, scale: Num, add_i: int, /) -> Self: ...

    def reduce_row(self, a: int, scale: Num | None, b: int | None = None):
        def row_a(row: list[Num]):
            if b is None:  # scale row_a
                if not scale:
                    raise ValueError(f"Scalar must be non-zero {scale=}")
                return [reduce_num(scale * v) for v in row]
            elif scale is None:  # swap
                return self[b]
            else:  # add scaled(row_a) to row_b
                return row

        def row_b(row: list[Num]):
            if scale is None:  # swap
                return self[a]
            else:  # add scaled(row_a) to row_b
                return [scale * _a + _b for _a, _b in zip(self[a], row)]

        return Matrix([
            (
                row_a(row) if i == a else
                row_b(row) if i == b else
                row
            )
            for i, row in enumerate(self)
        ])

    def swap_col(self, a_j: int, b_j: int):
        mapping = {a_j: b_j, b_j: a_j}
        return Matrix([
            [row[mapping.get(j, j)] for j in range(self.w)]
            for row in self
        ])

    def reduce_int(self):
        mat = Matrix(self)
        i = 0
        for j in range(mat.w):
            m = next((m for m in range(i, mat.h) if mat[m, j]), None)
            if m is None:
                continue
            elif m != i:
                mat = mat.reduce_row(m, None, i)

            if mat[i, j] < 0:
                mat = mat.reduce_row(i, -1)

            # denom = gcd(*map(int, mat[i])) * (-1 if mat[i, j] < 0 else 1)
            # if denom != 1:
            #     mat = mat.reduce_row(i, 1 / Fraction(denom)).reduce_nums()

            for m in range(mat.h):
                if m != i and mat[m, j] != 0:
                    d = gcd(int(mat[i, j]), int(mat[m, j]))
                    mat = mat.reduce_row(m, mat[i, j] // d).reduce_row(i, -mat[m, j] // d, m)

            # if mat[i, j] != 1:
            #     mat = mat.reduce_row(i, 1 / gcd(*mat[i])).reduce_nums()

            i += 1

        mat = Matrix([r for r in mat if any(v for v in r)])

        for i in range(mat.h):
            if (d := gcd(*map(int, mat[i]))) != 1:
                mat = mat.reduce_row(i, 1 / Fraction(d))

        return mat.reduce_nums()

    def reduce(self):
        mat = Matrix(self)
        i = 0
        for j in range(mat.w):
            m = next((m for m in range(i, mat.h) if mat[m, j]), None)
            if m is None:
                continue
            elif m != i:
                mat = mat.reduce_row(m, None, i)

            if mat[i, j] != 1:
                mat = mat.reduce_row(i, 1 / Fraction(mat[i, j])).reduce_nums()

            for m in range(mat.h):
                if m != i and mat[m, j] != 0:
                    mat = mat.reduce_row(i, -mat[m, j], m)

            i += 1

        mat = Matrix([r for r in mat if any(v for v in r)])
        return mat.reduce_nums()

    @overload
    def submatrix(self, stop: int, /) -> Self: ...
    @overload
    def submatrix(self, stop: tuple[int, int], /) -> Self: ...
    @overload
    def submatrix(self, start: tuple[int, int], stop: tuple[int, int]) -> Self: ...

    def submatrix(self, start: tuple[int, int] | int, stop: tuple[int, int] | None = None):
        if isinstance(start, int):
            start = (start, start)

        if stop is None:
            start, stop = (0, 0), start

        return Matrix([
            [v for j, v in enumerate(row) if start[1] <= j < stop[1]]
            for i, row in enumerate(self) if start[0] <= i < stop[0]
        ])

    def head(self, n: int):
        return self.submatrix((0, 0), (self.h, n))

    def tail(self, n: int):
        return self.submatrix((0, self.w - n), (self.h, self.w))

    h = height
    w = width
    iden = identity
    trans = transpose
    det = determinant
    rr = reduce_row
    subm = submatrix


def reduce_num(v: Num) -> Num:
    if isinstance(v, Fraction):
        return v.numerator if v.is_integer() else v
    elif isinstance(v, float):
        return i if (i := int(v)) == v else v
    else:
        return v


import pytest


def test_init():
    assert Matrix([[1]])[0] == [1]
    assert Matrix([[0, 1], [1, 2]])[1] == [1, 2]
    assert Matrix([[2, 3], [4, 5]])[1, 0] == 4

    (a, b), (c, d) = Matrix([[2, 3], [4, 5]])
    assert (a, b, c, d) == (2, 3, 4, 5)

    assert Matrix.identity(0) == []
    assert Matrix.identity(1) == [[1]]
    assert Matrix.identity(2) == [[1, 0], [0, 1]]

    assert Matrix.identity(2) == Matrix([[1, 0], [0, 1]])
    assert Matrix.identity(2) != Matrix([[1, 1], [0, 1]])
    assert Matrix.identity(2) != Matrix([[0, 1], [1, 2]])

    with pytest.raises(ValueError):
        Matrix([[1], [2, 3]])

    assert Matrix[[1, 2], [3, 4]] == Matrix([[1, 2], [3, 4]])

    row = [1, 2]
    m = Matrix[row, row]
    row[1] = 3
    assert m == Matrix[[1, 2], [1, 2]]


def test_getitem():
    m = Matrix[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    assert m[1:] == Matrix[
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_transpose():
    m = Matrix[[1, 2], [3, 4]]
    assert m.transpose() == Matrix[[1, 3], [2, 4]]


def test_mul():
    a = Matrix[
        [3, 7],
        [1, -4],
    ]
    assert 3 * a == a * 3 == Matrix[
        [9, 21],
        [3, -12],
    ]


def test_matmul():
    i2 = Matrix.identity(2)
    assert i2 @ i2 == Matrix.identity(2)

    a = Matrix[
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]
    assert a @ Matrix.identity(4) == a
    with pytest.raises(ValueError):
        _ = a @ i2

    b = Matrix[[7], [140], [7], [20]]
    c = Matrix[[7], [147], [167], [27]]
    assert a @ b == c


def test_det():
    assert Matrix[*[]].determinant() == 1
    assert Matrix[[4]].determinant() == 4

    a = Matrix[
        [3, 7],
        [1, -4],
    ]
    assert a.determinant() == -19

    a = Matrix[
        [1, -2, 3],
        [2, 0,  3],
        [1, 5,  4],
    ]
    assert a.det() == 25

    a = Matrix[
        [-2, -1, 2],
        [2,  1,  4],
        [-3, 3,  -1],
    ]
    assert a.det() == 54

    a = Matrix[
        [2,  1,  3, 4],
        [0,  -1, 2, 1],
        [3,  2,  0, 5],
        [-1, 3,  2, 1],
    ]
    assert a.det() == 35


def test_inverse():
    a = Matrix[
        [3, 7],
        [1, -4],
    ]
    assert a.inverse() @ a == Matrix.identity(2)
    assert a @ a.inverse() == Matrix.identity(2)


def test_reduce_row():
    a = Matrix[
        [0, -1, -1, 0],
        [2, -1, 0,  -1],
        [1, -1, 1,  1],
    ]

    assert a.reduce_row(1, 3) == Matrix[
        [0, -1, -1, 0],
        [6, -3, 0,  -3],
        [1, -1, 1,  1],
    ], "Should scale a row"

    assert a.reduce_row(0, None, 2) == Matrix[
        [1, -1, 1,  1],
        [2, -1, 0,  -1],
        [0, -1, -1, 0],
    ], "Should swap rows"

    assert a.reduce_row(0, -1, 2) == Matrix[
        [0, -1, -1, 0],
        [2, -1, 0,  -1],
        [1,  0, 2,  1],
    ], "Should add scaled row to another row"

    assert a.reduce_row(0, 0, 2) == Matrix[
        [0, -1, -1, 0],
        [2, -1, 0,  -1],
        [1, -1, 1,  1],
    ], "Should no-op"

    with pytest.raises(ValueError):
        a.reduce_row(1, 0)


def test_reduce_full():
    a = Matrix[
        [1, 2,  3,  6],
        [2, -3, 2,  14],
        [3, 1,  -1, -2],
    ]
    assert a.reduce() == Matrix[
        [1, 0, 0, 1],
        [0, 1, 0, -2],
        [0, 0, 1, 3],
    ]

    a = Matrix[
        [1, 0],
        [1, 0],
    ]
    assert a.reduce() == Matrix[
        [1, 0],
    ]

    a = Matrix[
        [0, 1],
        [0, 1],
    ]
    assert a.reduce() == Matrix[
        [0, 1],
    ]

    a = Matrix[
        [1, 3,  1,  9],
        [1, 1,  -1, 1],
        [3, 11, 5,  35],
    ]
    assert a.reduce() == Matrix[
        [1, 0, -2, -3],
        [0, 1, 1,  4],
    ]

    a = Matrix[
        [2, 1, 12, 1],
        [1, 2, 9, -1],
    ]
    assert a.reduce() == Matrix[
        [1, 0, 5, 1],
        [0, 1, 2, -1],
    ]

    a = Matrix[
        [1, 1, 0, 1, 1, 0, 1, 65],
        [1, 1, 0, 1, 1, 0, 0, 57],
        [0, 1, 0, 0, 1, 1, 0, 43],
        [0, 1, 0, 0, 0, 0, 1, 22],
        [0, 0, 1, 1, 0, 1, 0, 34],
    ]
    assert a.reduce() == Matrix[
        [1, 0, 0, 1, 0, -1, 0, 14],
        [0, 1, 0, 0, 0, 0, 0, 14],
        [0, 0, 1, 1, 0, 1, 0, 34],
        [0, 0, 0, 0, 1, 1, 0, 29],
        [0, 0, 0, 0, 0, 0, 1, 8],
    ]

def test_reduce_fraction():
    m = Matrix[
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 82],
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 113],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 122],
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 47],
        [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 76],
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 67],
        [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 90],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 53],
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 59],
        [1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 91],
    ]
    assert m.reduce() == Matrix[
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, Fraction(-1, 2), -1, 2, 27],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 7],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, Fraction(1, 2), -1, 1, 4],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, Fraction(-1, 2), -2, 2, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, -2, 16],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, Fraction(1, 2), -2, 2, 3],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, -1, 38],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, Fraction(1, 2), -3, 4, 20],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 29],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, Fraction(-1, 2), 3, -5, -11],
    ]
    assert m.reduce_int() is None

    m = Matrix[
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 78],
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 30],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 51],
        [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 90],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 71],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 104],
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 54],
        [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 67],
        [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 29],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 51],
    ]
    assert m.reduce() == Matrix[
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, Fraction(1, 6), Fraction(1, 6), Fraction(1, 2), Fraction(15, 2)],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, Fraction(-2, 3), Fraction(1, 3), 1, 21],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, Fraction(5, 6), Fraction(5, 6), Fraction(-1, 2), Fraction(61, 2)],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, Fraction(-5, 6), Fraction(-5, 6), Fraction(3, 2), Fraction(-17, 2)],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, Fraction(2, 3), Fraction(-1, 3), 0, 13],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, Fraction(-1, 6), Fraction(-1, 6), Fraction(-1, 2), Fraction(11, 2)],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, Fraction(1, 3), Fraction(1, 3), 0, 26],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, Fraction(1, 3), Fraction(1, 3), -1, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, Fraction(-1, 6), Fraction(-1, 6), Fraction(-1, 2), Fraction(13, 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(35, 2)],
    ]
