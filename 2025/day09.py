from enum import Enum, StrEnum
from collections import deque
import dataclasses as dc
import itertools as it
from collections import Counter
import operator as op
import re
import functools as ft
from typing import Any, Callable, Generator, Iterable, Iterator, Literal, NewType, Sequence, TypeVar, overload

import aoc


@dc.dataclass(frozen=True, eq=True, order=True)
class Coord:
    i: int
    j: int

    def __iter__(self):
        yield self.i
        yield self.j


    def __getitem__(self, idx: int):
        return self.j if idx else self.i

    def __eq__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            return (self.i, self.j) == other
        else:
            return isinstance(other, Coord) and tuple(self) == tuple(other)


type CoordTup = Coord | tuple[int, int]
type Red = Literal['#']
type Green = Literal['X']
type Color = Red | Green
type Rect = tuple[Coord, Coord]


RED: Red = '#'
GRN: Green = 'X'


@dc.dataclass(frozen=True, eq=True)
class Tile:
    c: Coord
    n: Coord  # reds point to next red, greens point to next red or green
    v: Color

    def cardir(self):
        return cardir(self.c, self.n)



T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

class Dir(StrEnum):
    N = 'N'
    S = 'S'
    W = 'W'
    E = 'E'


def north(c: Coord):
    return Coord(c.i, c.j - 1)


def south(c: Coord):
    return Coord(c.i, c.j + 1)


def east(c: Coord):
    return Coord(c.i + 1, c.j)


def west(c: Coord):
    return Coord(c.i - 1, c.j)


def box(c: Coord):
    return [n for p in [north(c), c, south(c)] for n in [west(p), p, east(p)]]


def neighbors(c: Coord):
    return [b for b in box(c) if b != c]


@overload
def coordtup(func: Callable[[Coord, Coord], V]) -> Callable[[CoordTup, CoordTup], V]: ...
@overload
def coordtup(func: Callable[[Coord, Coord, Coord], V]) -> Callable[[CoordTup, CoordTup, CoordTup], V]: ...

def coordtup(func: Callable[..., V]) -> Callable[..., V]:
    def wrapper(*args: CoordTup):
        coords = (c if isinstance(c, Coord) else Coord(*c) for c in args)
        return func(*coords)

    return wrapper


def main():
    print(biggest_rect(parsed()))

    # looks like a diamond with a cut into it from the left:
    #                       #
    #                     #   #
    #                   #       #
    #                 #           #
    #               #############   #    <- (94927, 48406)
    #                           #     #  <- start
    #               #############   #    <- (94927, 50365)
    #                 #           #
    #                   #       #
    #                     #   #
    #                       #
    # print(parsed()[0])
    # for d, n in cardinal(parsed()):
    #     print(f"{d} {n}")

    # rngs = [0, 12_500, 25_000, 37_500, 50_000, 62_500, 75_000, 87_500, 100_000]

    # lo_rng = range(12_500, 37_500)
    # hi_rng = range(62_500, 87_500)

    # def in_rng(v: int):
    #     return lo_rng.start <= v <= lo_rng.stop or hi_rng.start <= v <= hi_rng.stop

    # top = [c for c in parsed() if in_rng(c.i) and lo_rng.start <= c.j <= 48406]
    # btm = [c for c in parsed() if in_rng(c.i) and 50365 <= c.j <= hi_rng.stop]

    # print(max(biggest_rect(top), biggest_rect(btm)))

    # print(p2(aoc.input_str()))
    print(p2(parsed()))

    # filled = fill(polygon(parsed()), west(parsed()[0]))

    too_high = [
        3012543072,
        3037295676,
    ]
    ans = 1543501936
    too_low = [
        548394154,
    ]


def groupby(key: Callable[[T], U], items: Iterable[T]):
    grouped: dict[U, list[T]] = {}
    for item in items:
        k = key(item)
        grouped[k] = grouped.get(k, [])
        grouped[k].append(item)

    return grouped


def covers(rect: Rect, wall: Rect):
    tl, br = rect
    w1, w2 = wall

    tl.i


def partition(pred: Callable[[T], bool], itr: Iterable[T]) -> tuple[list[T], list[T]]:
    res = {True: [], False: []}
    for i in itr:
        res[pred(i)].append(i)
    return res[True], res[False]



def crosses(rect: Rect, poly_wall: Rect):
    tl, br = rect
    w1, w2 = poly_wall

    is_vwall = w1.i == w2.i

    return (
        (w1.j < tl.j < w2.j) or # intersects_north_wall
        (w1.j < br.j < w2.j) or # intersects_south_wall
        (w1.i < br.i < w2.i) or # intersects_east_wall
        (w1.i < tl.i < w2.i) or # intersects_west_wall
        ((tl.i <= w1.i <= br.i) and (tl.j < w1.j < br.j)) or  # w1 inside rect
        ((tl.i < w2.i < br.i) and (tl.j < w2.j < br.j))     # w2 inside rect
    )


def p2(tiles: list[Coord]):
    def norm_rect(rect: Rect) -> Rect:
        a, b = sorted(rect)
        return (a, b)

    pairs = (norm_rect((a, b)) for a, b in zip(tiles, rot(tiles)))
    vert_walls, hori_walls = partition(lambda p: p[0].i == p[1].i, pairs)

    vert_walls_by_i = groupby(lambda w: w[0].i, vert_walls)
    hori_walls_by_j = groupby(lambda w: w[0].j, vert_walls)

    for a, b in zip(tiles, rot(tiles)):
        if a.i == b.i:
            (a, b) if a.j < b.j else (a, b)

    vert_walls = groupby(lambda p: p[0].i, vert_walls)
    hori_walls = groupby(lambda p: p[0].j, hori_walls)

    def valid_combos():
        for a, b in it.combinations(tiles, 2):
            rect = bounds([a, b])
            tl, br = rect

            vwalls = it.chain.from_iterable(vert_walls.get(i, []) for i in range(tl.i + 1, br.i))
            hwalls = it.chain.from_iterable(hori_walls.get(j, []) for j in range(tl.j + 1, br.j))

            if any(crosses(rect, vw) for vw in vwalls) or any(crosses(rect, hw) for hw in hwalls):
                continue

            yield a, b


    return max(map(lambda p: area(p[0], p[1]), valid_combos()))


def fill(poly: dict[Coord, Tile], start: Coord):
    frontier = {start}
    while frontier:
        curr = frontier.pop()

        poly[curr] = Tile(curr, Coord(-1, -1), GRN)

        for n in neighbors(curr):
            if n not in frontier and n not in poly:
                frontier.add(n)

    return poly


@coordtup
def sub(a: Coord, b: Coord):
    return Coord(b.i - a.i, b.j - a.j)


@coordtup
def area(a: Coord, b: Coord):
    return ft.reduce(lambda a, b: a * b, (abs(u - v) + 1 for u, v in zip(a, b)))


def biggest_rect(coords: list[Coord]):
    return max(map(lambda p: area(p[0], p[1]), it.combinations(coords, 2)))


def parsed(inp: str | None = None) -> list[Coord]:
    lines = inp.splitlines() if inp else aoc.input_lines()
    return [
        Coord(i, j) for line in lines for i, j in [map(int, line.split(','))]
    ]


def rot(seq: Sequence[T], n: int = 1):
    r = n % len(seq)
    return it.chain(it.islice(seq, r, None), it.islice(seq, r))


def downjust(tiles: list[Coord]) -> list[Coord]:
    i, j = min(i for i, _ in tiles), min(j for _, j in tiles)
    return [Coord(x - i, y - j) for x, y in tiles]


@coordtup
def tiles_between(a: Coord, b: Coord):
    if a == b:
        raise ValueError("Same coords")
    elif a.i == b.i:
        sign = 1 if a.j < b.j else -1
        return (Coord(a.i, j) for j in range(a.j + sign, b.j, sign))
    elif a.j == b.j:
        sign = 1 if a.i < b.i else -1
        return (Coord(i, a.j) for i in range(a.i + sign, b.i, sign))

    raise ValueError("Not on same row or column")


def bounds(tiles: Iterable[Coord]):
    tl: Coord = Coord(min(i for i, _ in tiles), min(j for _, j in tiles))
    br: Coord = Coord(max(i for i, _ in tiles), max(j for _, j in tiles))
    return tl, br


def green_tiles(red_tiles: list[Coord]):
    return (t for a, b in zip(red_tiles, rot(red_tiles)) for t in tiles_between(a, b))


def is_inside(polygon: set[Coord], pt: Coord):
    if pt in polygon:
        return True

    tl, br = bounds(polygon)
    in_bounds = tl.i <= pt.i <= br.i and tl.j <= pt.j <= br.j
    if not in_bounds:
        return False


def polygon(tiles: list[Coord]):
    poly = {a: Tile(a, b, RED) for a, b in zip(tiles, rot(tiles))}
    for red in list(poly.values()):
        for a, b in it.pairwise([*tiles_between(red.c, red.n), red.n]):
            poly[a] = Tile(a, b, GRN)

    return poly


def smooth(poly: dict[Coord, Tile]):
    pass


@coordtup
def cardir(a: Coord, b: Coord):
    if a == b:
        raise ValueError("Same coords")
    elif a.i == b.i:
        return Dir.S if a.j < b.j else Dir.N
    elif a.j == b.j:
        return Dir.E if a.i < b.i else Dir.W

    raise ValueError("Not on same row or column")


def cardinal(red_tiles: list[Coord]):
    return [
        (cardir(a, b), manhat_dist(a, b))
        for a, b in zip(red_tiles, rot(red_tiles))
    ]


def manhat_dist(a: Coord, b: Coord):
    return abs(b.i - a.i) + abs(b.j - a.j)


def render(tiles: list[Coord], fill_at: Coord):
    tl, br = bounds(tiles)
    poly = fill(polygon(tiles), fill_at)

    for j in range(tl.j, br.j + 1):
        for i in range(tl.i, br.i + 1):
            t = poly.get(Coord(i, j))
            print(t.v if t else '.', end='')

        print()


import bisect


def part2(coords: list[tuple[int, int]]):
    """Solve part 2 by checking for the largest rectangle inside the shape."""

    # Build edges from consecutive coordinates, normalized by min/max
    hedges: list[tuple[int, int, int]] = []
    vedges: list[tuple[int, int, int]] = []
    for i, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(i + 1) % len(coords)]
        if x1 == x2:
            # Vertical edge: store as (min_y, max_y, x)
            vedges.append((min(y1, y2), max(y1, y2), x1))
        else:
            # Horizontal edge: store as (min_x, max_x, y)
            hedges.append((min(x1, x2), max(x1, x2), y1))

    # Find longest edges to check first
    # longest_hedge_idx = max(range(len(hedges)), key=lambda i: hedges[i][1] - hedges[i][0])
    # longest_vedge_idx = max(range(len(vedges)), key=lambda i: vedges[i][1] - vedges[i][0])

    # longest_hedge = hedges[longest_hedge_idx]
    # longest_vedge = vedges[longest_vedge_idx]

    # Sort edges by their singular coordinate for binary search
    # hedges sorted by y value (third element)
    # hedges_sorted = sorted(enumerate(hedges), key=lambda item: item[1][2])
    # hedges = [h for _, h in hedges_sorted]
    hedges.sort(key=lambda item: item[2])
    hedge_py_values = [h[2] for h in hedges]

    # vedges sorted by x value (third element)
    # vedges_sorted = sorted(enumerate(vedges), key=lambda item: item[1][2])
    # vedges = [v for _, v in vedges_sorted]
    vedges.sort(key=lambda item: item[2])
    vedge_px_values = [v[2] for v in vedges]

    def is_valid_rectangle(rect: tuple[int, int, int, int]):
        """Check if rectangle is valid (no edges cross through it).

        Uses binary search to only check relevant edges based on their singular coordinate.
        hedge_py_values and vedge_px_values are sorted lists for binary search.
        Checks longest edges first for early exit.
        """
        x1, y1, x2, y2 = rect

        # # Check longest horizontal edge first
        # px1, px2, py = longest_hedge
        # if y1 < py < y2 and px1 < x2 <= px2:
        #     return False
        # if y1 < py < y2 and px1 <= x1 < px2:
        #     return False

        # # Check longest vertical edge first
        # py1, py2, px = longest_vedge
        # if x1 < px < x2 and py1 < y2 <= py2:
        #     return False
        # if x1 < px < x2 and py1 <= y1 < py2:
        #     return False

        # For horizontal edges, find those with py in range (y1, y2)
        # Binary search for first edge with py > y1, check while py < y2
        hedge_start = bisect.bisect_right(hedge_py_values, y1)
        hedge_end = bisect.bisect_left(hedge_py_values, y2)

        for i in range(hedge_start, hedge_end):
            px1, px2, py = hedges[i]
            if px1 < x2 <= px2:
                return False
            if px1 <= x1 < px2:
                return False

        # For vertical edges, find those with px in range (x1, x2)
        # Binary search for first edge with px > x1, check while px < x2
        vedge_start = bisect.bisect_right(vedge_px_values, x1)
        vedge_end = bisect.bisect_left(vedge_px_values, x2)

        for i in range(vedge_start, vedge_end):
            py1, py2, px = vedges[i]
            if py1 < y2 <= py2:
                return False
            if py1 <= y1 < py2:
                return False

        return True

    largest = 0
    for a, b in it.combinations(coords, 2):
        x1, y1 = a
        x2, y2 = b

        # Check area first (cheap) before expensive intersection checks
        area = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
        if area <= largest:
            continue

        # Build normalized rectangle
        rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        if is_valid_rectangle(rect):
            largest = area

    return largest



if __name__ == "__main__":
    main()


import pytest

example = aoc.heredoc("""
    7,1
    11,1
    11,7
    9,7
    9,5
    2,5
    2,3
    7,3
""")


def test_coord():
    c = Coord(3, 5)
    assert c.i == 3
    assert c.j == 5
    assert Coord(3, 5) == Coord(3, 5)
    assert {Coord(3, 5), Coord(3, 5)} == {Coord(3, 5)}


def test_area():
    assert area((2, 5), (9, 7)) == 24
    assert area((7, 1), (11, 7)) == 35


def test_rot():
    assert list(rot([1, 2, 3])) == [2, 3, 1]
    assert list(rot([1, 2, 3], 2)) == [3, 1, 2]
    assert list(rot([1, 2, 3], 3)) == [1, 2, 3]
    assert list(rot([1, 2, 3], 4)) == [2, 3, 1]
    assert list(rot([1, 2, 3], -1)) == [3, 1, 2]


def test_tiles_between():
    assert list(tiles_between((2, 2), (2, 5))) == [(2, 3), (2, 4)]
    assert list(tiles_between((2, 5), (2, 2))) == [(2, 4), (2, 3)]

    assert list(tiles_between((2, 2), (5, 2))) == [(3, 2), (4, 2)]
    assert list(tiles_between((5, 2), (2, 2))) == [(4, 2), (3, 2)]


# def test_crosses():
#     rect = (Coord(5, 5), Coord(15, 15))

    # assert     crosses(rect, (Coord(10, 0),  Coord(10, 10)))
    # assert     crosses(rect, (Coord(10, 10), Coord(10, 20)))
    # assert not crosses(rect, (Coord(10, 5),  Coord(10, 15)))
    # assert not crosses(rect, (Coord(10, 0),  Coord(10, 5)))
    # assert not crosses(rect, (Coord(10, 15), Coord(10, 20)))

    # assert     crosses(rect, (Coord(0,  10), Coord(10, 10)))
    # assert     crosses(rect, (Coord(10, 10), Coord(20, 10)))
    # assert not crosses(rect, (Coord(5, 10),  Coord(15, 10)))
    # assert not crosses(rect, (Coord(0, 10),  Coord(5, 10)))
    # assert not crosses(rect, (Coord(15, 10), Coord(20, 10)))


def test_part2():
    assert part2(list(map(lambda c: (c.i, c.j), parsed(example)))) == 24
    assert p2(parsed(example)) == 24

    # assert part2(parsed(aoc.input_str(9))) == 1543501936


def test_vecdir():
    assert cardir((2, 2), (2, 5)) == Dir.S
    assert cardir((2, 5), (2, 2)) == Dir.N

    assert cardir((2, 2), (5, 2)) == Dir.E
    assert cardir((5, 2), (2, 2)) == Dir.W


def test_example():
    assert biggest_rect(parsed(example)) == 50


def demo():
    parse = parsed(example)
    render(parse, east(south(parse[0])))
    for d, n in cardinal(parsed(example)):
        print(f"{d} {n}")


"""
    97803,50388 S 1204
    97803,51592 W 243
    97560,51592 S 1237
    97560,52829 E 505
    98065,52829 S 1214
    98065,54043 W 119
    97946,54043 S 1242
    97946,55285 E 143
    98089,55285 S 1121
    98089,56406 W 854
    97235,56406 S 1317
    97235,57723 E 580
    97815,57723 S 1147
    97815,58870 W 542
    97273,58870 S 1072
    97273,59942 W 812
    96461,59942 S 1219
    96461,61161 W 79
    96382,61161 S 1333
    96382,62494 E 304
    96686,62494 S 1224
    96686,63718 W 176
    96510,63718 S 1048
    96510,64766 W 753
    95757,64766 S 1141
    95757,65907 W 426
    95331,65907 S 1258
    95331,67165 W 116
    95215,67165 S 1138
    95215,68303 W 453
    94762,68303 S 1152
    94762,69455 W 423
    94339,69455 S 1137
    94339,70592 W 466
    93873,70592 S 946
    93873,71538 W 853
    93020,71538 S 1246
    93020,72784 W 252
    92768,72784 S 967
    92768,73751 W 784
    91984,73751 S 1107
    91984,74858 W 532
    91452,74858 S 607
    91452,75465 W 1328
    90124,75465 S 1082
    90124,76547 W 548
    89576,76547 S 1175
    89576,77722 W 429
    89147,77722 S 811
    89147,78533 W 946
    88201,78533 S 952
    88201,79485 W 743
    87458,79485 S 952
    87458,80437 W 743
    86715,80437 S 835
    86715,81272 W 881
    85834,81272 S 914
    85834,82186 W 786
    85048,82186 S 819
    85048,83005 W 888
    84160,83005 S 1322
    84160,84327 W 387
    83773,84327 S 1059
    83773,85386 W 680
    83093,85386 S 156
    83093,85542 W 1503
    81590,85542 S 776
    81590,86318 W 921
    80669,86318 S 1011
    80669,87329 W 734
    79935,87329 S 711
    79935,88040 W 982
    78953,88040 S 826
    78953,88866 W 897
    78056,88866 S 832
    78056,89698 W 902
    77154,89698 S 288
    77154,89986 W 1265
    75889,89986 S 1370
    75889,91356 W 577
    75312,91356 S 97
    75312,91453 W 1362
    73950,91453 S 346
    73950,91799 W 1193
    72757,91799 S 618
    72757,92417 W 1038
    71719,92417 S 1162
    71719,93579 W 781
    70938,93579 S 248
    70938,93827 W 1231
    69707,93827 S 604
    69707,94431 W 1067
    68640,94431 S 588
    68640,95019 W 1081
    67559,95019 S 497
    67559,95516 W 1122
    66437,95516 N 237
    66437,95279 W 1371
    65066,95279 S 664
    65066,95943 W 1061
    64005,95943 S 183
    64005,96126 W 1212
    62793,96126 S 749
    62793,96875 W 1063
    61730,96875 S 397
    61730,97272 W 1165
    60565,97272 N 154
    60565,97118 W 1281
    59284,97118 N 120
    59284,96998 W 1255
    58029,96998 S 836
    58029,97834 W 1099
    56930,97834 S 171
    56930,98005 W 1213
    55717,98005 N 79
    55717,97926 W 1237
    54480,97926 S 372
    54480,98298 W 1197
    53283,98298 N 330
    53283,97968 W 1242
    52041,97968 S 51
    52041,98019 W 1215
    50826,98019 S 118
    50826,98137 W 1218
    49608,98137 S 66
    49608,98203 W 1222
    48386,98203 N 253
    48386,97950 W 1209
    47177,97950 N 344
    47177,97606 W 1193
    45984,97606 N 289
    45984,97317 W 1185
    44799,97317 S 214
    44799,97531 W 1246
    43553,97531 N 607
    43553,96924 W 1133
    42420,96924 S 700
    42420,97624 W 1357
    41063,97624 N 1091
    41063,96533 W 1021
    40042,96533 S 102
    40042,96635 W 1265
    38777,96635 N 546
    38777,96089 W 1112
    37665,96089 N 227
    37665,95862 W 1193
    36472,95862 S 80
    36472,95942 W 1299
    35173,95942 N 285
    35173,95657 W 1196
    33977,95657 N 997
    33977,94660 W 932
    33045,94660 S 47
    33045,94707 W 1327
    31718,94707 N 329
    31718,94378 W 1192
    30526,94378 N 1201
    30526,93177 W 793
    29733,93177 N 110
    29733,93067 W 1295
    28438,93067 N 734
    28438,92333 W 992
    27446,92333 N 419
    27446,91914 W 1159
    26287,91914 N 363
    26287,91551 W 1205
    25082,91551 N 1206
    25082,90345 W 688
    24394,90345 N 232
    24394,90113 W 1302
    23092,90113 N 984
    23092,89129 W 803
    22289,89129 N 331
    22289,88798 W 1269
    21020,88798 N 1092
    21020,87706 W 701
    20319,87706 N 484
    20319,87222 W 1178
    19141,87222 N 1069
    19141,86153 W 693
    18448,86153 N 986
    18448,85167 W 745
    17703,85167 N 805
    17703,84362 W 904
    16799,84362 N 673
    16799,83689 W 1042
    15757,83689 N 1138
    15757,82551 W 564
    15193,82551 N 711
    15193,81840 W 1017
    14176,81840 N 1150
    14176,80690 W 520
    13656,80690 N 818
    13656,79872 W 907
    12749,79872 N 616
    12749,79256 W 1188
    11561,79256 N 1019
    11561,78237 W 679
    10882,78237 N 895
    10882,77342 W 856
    10026,77342 N 1188
    10026,76154 W 422
    9604,76154  N 1110
    9604,75044  W 523
    9081,75044  N 1040
    9081,74004  W 629
    8452,74004  N 1272
    8452,72732  W 205
    8247,72732  N 863
    8247,71869  W 957
    7290,71869  N 1034
    7290,70835  W 656
    6634,70835  N 1067
    6634,69768  W 597
    6037,69768  N 1083
    6037,68685  W 577
    5460,68685  N 1069
    5460,67616  W 627
    4833,67616  N 1334
    4833,66282  E 80
    4913,66282  N 1004
    4913,65278  W 830
    4083,65278  N 1321
    4083,63957  E 129
    4212,63957  N 1027
    4212,62930  W 835
    3377,62930  N 1261
    3377,61669  W 6
    3371,61669  N 1193
    3371,60476  W 248
    3123,60476  N 1229
    3123,59247  W 53
    3070,59247  N 1182
    3070,58065  W 280
    2790,58065  N 1240
    2790,56825  E 96
    2886,56825  N 1100
    2886,55725  W 961
    1925,55725  N 1262
    1925,54463  E 330
    2255,54463  N 1177
    2255,53286  W 608
    1647,53286  N 1232
    1647,52054  E 92
    1739,52054  N 1236
    1739,50818  E 660
    2399,50818  N 453
    2399,50365  E 92528
    94927,50365 N 1959
    94927,48406 W 92515
    2412,48406  N 1228
    2412,47178  W 334
    2078,47178  N 1224
    2078,45954  W 41
    2037,45954  N 1173
    2037,44781  E 484
    2521,44781  N 1187
    2521,43594  E 255
    2776,43594  N 1171
    2776,42423  E 319
    3095,42423  N 1337
    3095,41086  W 601
    2494,41086  N 1169
    2494,39917  E 389
    2883,39917  N 1067
    2883,38850  E 785
    3668,38850  N 1353
    3668,37497  W 388
    3280,37497  N 1201
    3280,36296  E 260
    3540,36296  N 969
    3540,35327  E 992
    4532,35327  N 1319
    4532,34008  W 103
    4429,34008  N 922
    4429,33086  E 1019
    5448,33086  N 1359
    5448,31727  W 135
    5313,31727  N 967
    5313,30760  E 842
    6155,30760  N 977
    6155,29783  E 772
    6927,29783  N 1438
    6927,28345  W 180
    6747,28345  N 749
    6747,27596  E 1200
    7947,27596  N 1236
    7947,26360  E 266
    8213,26360  N 1038
    8213,25322  E 634
    8847,25322  N 874
    8847,24448  E 893
    9740,24448  N 1283
    9740,23165  E 255
    9995,23165  N 848
    9995,22317  E 914
    10909,22317 N 1204
    10909,21113 E 415
    11324,21113 N 942
    11324,20171 E 781
    12105,20171 N 545
    12105,19626 E 1257
    13362,19626 N 993
    13362,18633 E 696
    14058,18633 N 1189
    14058,17444 E 491
    14549,17444 N 483
    14549,16961 E 1256
    15805,16961 N 1247
    15805,15714 E 463
    16268,15714 N 515
    16268,15199 E 1186
    17454,15199 N 1286
    17454,13913 E 472
    17926,13913 N 414
    17926,13499 E 1251
    19177,13499 N 1058
    19177,12441 E 703
    19880,12441 N 509
    19880,11932 E 1145
    21025,11932 N 742
    21025,11190 E 959
    21984,11190 N 900
    21984,10290 E 854
    22838,10290 N 974
    22838,9316  E 821
    23659,9316  N 491
    23659,8825  E 1139
    24798,8825  N 61
    24798,8764  E 1377
    26175,8764  N 1240
    26175,7524  E 698
    26873,7524  S 128
    26873,7652  E 1442
    28315,7652  N 659
    28315,6993  E 1021
    29336,6993  N 973
    29336,6020  E 888
    30224,6020  N 445
    30224,5575  E 1138
    31362,5575  S 45
    31362,5620  E 1327
    32689,5620  N 874
    32689,4746  E 968
    33657,4746  N 69
    33657,4677  E 1262
    34919,4677  N 376
    34919,4301  E 1150
    36069,4301  N 284
    36069,4017  E 1177
    37246,4017  N 1014
    37246,3003  E 992
    38238,3003  N 133
    38238,2870  E 1228
    39466,2870  N 417
    39466,2453  E 1165
    40631,2453  S 179
    40631,2632  E 1276
    41907,2632  N 268
    41907,2364  E 1191
    43098,2364  S 30
    43098,2394  E 1232
    44330,2394  S 125
    44330,2519  E 1231
    45561,2519  N 358
    45561,2161  E 1187
    46748,2161  S 322
    46748,2483  E 1229
    47977,2483  N 389
    47977,2094  E 1198
    49175,2094  N 99
    49175,1995  E 1215
    50390,1995  S 186
    50390,2181  E 1211
    51601,2181  S 42
    51601,2223  E 1211
    52812,2223  S 419
    52812,2642  E 1182
    53994,2642  N 771
    53994,1871  E 1296
    55290,1871  S 439
    55290,2310  E 1178
    56468,2310  S 426
    56468,2736  E 1166
    57634,2736  S 51
    57634,2787  E 1224
    58858,2787  N 7
    58858,2780  E 1246
    60104,2780  S 458
    60104,3238  E 1148
    61252,3238  S 818
    61252,4056  E 1043
    62295,4056  N 129
    62295,3927  E 1294
    63589,3927  S 249
    63589,4176  E 1198
    64787,4176  S 139
    64787,4315  E 1244
    66031,4315  S 1000
    66031,5315  E 932
    66963,5315  N 177
    66963,5138  E 1381
    68344,5138  S 1078
    68344,6216  E 868
    69212,6216  S 654
    69212,6870  E 1031
    70243,6870  S 338
    70243,7208  E 1180
    71423,7208  S 660
    71423,7868  E 1022
    72445,7868  S 617
    72445,8485  E 1041
    73486,8485  N 11
    73486,8474  E 1415
    74901,8474  S 761
    74901,9235  E 970
    75871,9235  S 878
    75871,10113 E 884
    76755,10113 S 480
    76755,10593 E 1151
    77906,10593 S 1110
    77906,11703 E 698
    78604,11703 S 310
    78604,12013 E 1296
    79900,12013 S 1400
    79900,13413 E 430
    80330,13413 S 487
    80330,13900 E 1173
    81503,13900 S 637
    81503,14537 E 1064
    82567,14537 S 1206
    82567,15743 E 530
    83097,15743 S 908
    83097,16651 E 798
    83895,16651 S 855
    83895,17506 E 850
    84745,17506 S 406
    84745,17912 E 1356
    86101,17912 S 939
    86101,18851 E 785
    86886,18851 S 1191
    86886,20042 E 470
    87356,20042 S 767
    87356,20809 E 995
    88351,20809 S 1017
    88351,21826 E 677
    89028,21826 S 1320
    89028,23146 E 230
    89258,23146 S 606
    89258,23752 E 1282
    90540,23752 S 940
    90540,24692 E 807
    91347,24692 S 1425
    91347,26117 W 11
    91336,26117 S 1013
    91336,27130 E 667
    92003,27130 S 821
    92003,27951 E 1057
    93060,27951 S 1153
    93060,29104 E 429
    93489,29104 S 1024
    93489,30128 E 704
    94193,30128 S 1363
    94193,31491 W 77
    94116,31491 S 1215
    94116,32706 E 219
    94335,32706 S 946
    94335,33652 E 931
    95266,33652 S 1229
    95266,34881 E 168
    95434,34881 S 1157
    95434,36038 E 364
    95798,36038 S 1158
    95798,37196 E 364
    96162,37196 S 1135
    96162,38331 E 464
    96626,38331 S 1095
    96626,39426 E 684
    97310,39426 S 1242
    97310,40668 E 49
    97359,40668 S 1181
    97359,41849 E 348
    97707,41849 S 1259
    97707,43108 W 138
    97569,43108 S 1268
    97569,44376 W 351
    97218,44376 S 1151
    97218,45527 E 622
    97840,45527 S 1249
    97840,46776 W 415
    97425,46776 S 1163
    97425,47939 E 975
    98400,47939 S 1236
    98400,49175 W 486
    97914,49175 S 1213
    97914,50388 W 111
"""
