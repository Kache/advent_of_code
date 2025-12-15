import datetime as dt
import functools as ft
import inspect
import os
import sys
from http.client import HTTPResponse
from pathlib import Path
from urllib.request import Request, urlopen

now = dt.datetime.now()

def input_lines(day: int = 0, year: int = now.year):
    return input_str(day, year).splitlines()


def input_str(day: int = 0, year: int = now.year):
    return input_bytes(day, year).decode()


def input_bytes(day: int = 0, year: int = now.year):
    return _input_bytes(day or infer_day(), year)


def infer_day():
    main = sys.modules['__main__']
    path = main.__file__ and Path(main.__file__)
    if path and path.stem.startswith('day'):
        return int(path.stem[3:])

    raise RuntimeError("Unable to infer day")


@ft.cache
def _input_bytes(day: int, year: int):
    path = Path(f"inputs/input{day:02}")
    if path.exists():
        return path.read_bytes()

    resp: HTTPResponse = urlopen(Request(
        f'https://adventofcode.com/{year}/day/{day}/input',
        headers = {'Cookie': f"session={os.getenv('AOC_SESSION')}"}
    ))
    data = resp.read()
    path.write_bytes(data)
    return data


def heredoc(s: str):
    return inspect.cleandoc(s) + '\n'
