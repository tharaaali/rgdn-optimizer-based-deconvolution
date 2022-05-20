import math
import numpy as np


def zernike_R(n, m, r):
    assert n >= m >= 0, f"Incorrect n,m passed. Expected n >= m >= 0, got n={n}, m={m}"

    def Rk(n, m, r, k):
        num = (r ** (n - 2 * k)) * math.factorial(n - k) * (-1) ** k
        denum = math.factorial(k) * math.factorial((n + m) / 2 - k) * math.factorial((n - m) / 2 - k)
        return num / denum

    if (n - m) & 1:
        return 0.0
    if m == n:
        return r ** m

    val = 0.0
    for k in range(int((n - m) / 2 + 1)):
        val += Rk(n, m, r, k)
    return val


# A recursive implementation of Zernike polynomials calculation in polar coordinates
def zernike(n, m, r, phi):
    assert n >= m and not ((n - m) & 1), \
        f"Incorrect n, m passed. Expected n>=m and (n-m) is even, got n={n}, m={m}"

    if m > 0:
        return math.sqrt(2) * math.sqrt(n + 1) * np.cos(abs(m) * phi) * zernike_R(n, m, r)
    elif m == 0:
        return math.sqrt(n + 1) * zernike_R(n, m, r)
    else:  # m < 0
        return math.sqrt(2) * math.sqrt(n + 1) * np.sin(abs(m) * phi) * zernike_R(n, abs(m), r)
