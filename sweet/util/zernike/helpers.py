import math


def _order(j):
    if j == 0:
        return 0
    r = int(math.sqrt(1 + 2*j) + 0.5) - 1
    return r


def _freq(j, n):
    if j == 0:
        return 0
    return 2*j - n**2 - 2*n


def double_index(j):
    n = _order(j)
    m = _freq(j, n)
    return n, m


def single_index(n, m):
    return int((n**2 + 2*n + m) / 2)
