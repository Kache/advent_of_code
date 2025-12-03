import datetime as dt
import functools as ft
import inspect
import os
from http.client import HTTPResponse
from urllib.request import Request, urlopen

now = dt.datetime.now()

def input_lines(day: int = 1, year: int = now.year):
    return input_str(day, year).splitlines()


def input_str(day: int = 1, year: int = now.year):
    return input_bytes(day, year).decode()


@ft.cache
def input_bytes(day: int = 1, year: int = now.year):
    resp: HTTPResponse = urlopen(Request(
        f'https://adventofcode.com/{year}/day/{day}/input',
        headers = {'Cookie': f"session={os.getenv('AOC_SESSION')}"}
    ))
    return resp.read()


def heredoc(s: str):
    return inspect.cleandoc(s) + '\n'
