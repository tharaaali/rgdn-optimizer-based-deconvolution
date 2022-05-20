import math


def _positive_Z(n, m, x, y):
    assert n >= m >= 0
    z = 0.0
    for s in range(int((n - m) / 2) + 1):
        for j in range(int((n - m) / 2) - s + 1):
            for k in range(int(m / 2) + 1):
                num = math.factorial(n - s) * math.factorial(m) * (-1) ** (s + k)
                denum = math.factorial(s) * math.factorial((n + m) / 2 - s) \
                        * math.factorial(j) * math.factorial((n - m) / 2 - s - j) \
                        * math.factorial(2 * k) * math.factorial(m - 2 * k)
                val = (num / denum) * (x ** (n - 2 * (s + j + k))) * (y ** (2 * (j + k)))
                z += val
    return z


def _negative_Z(n, m, x, y):
    assert n >= m >= 0
    z = 0.0
    for s in range(int((n - m) / 2) + 1):
        for j in range(int((n - m) / 2) - s + 1):
            for k in range(int((m - 1) / 2 + 1)):
                num = math.factorial(n - s) * math.factorial(m) * (-1) ** (s + k)
                denum = math.factorial(s) * math.factorial((n + m) / 2 - s) \
                        * math.factorial(j) * math.factorial((n - m) / 2 - s - j) \
                        * math.factorial(2 * k + 1) * math.factorial(m - 2 * k - 1)
                val = (num / denum) * (x ** (n - 2 * (s + j + k) - 1)) * (y ** (2 * (j + k) + 1))
                z += val
    return z


# A recursive implementation of Cartesian Zernike polynomials calculation
# Source: https://wp.optics.arizona.edu/visualopticslab/wp-content/uploads/sites/52/2016/08/Zernike-Notes-15Jan2016.pdf
def zernike(n, m, x, y):
    assert n >= m and not ((n - m) & 1), \
        f"Incorrect n, m passed. Expected n>=m and (n-m) is even, got n={n}, m={m}"

    if m > 0:
        return math.sqrt(2 * (n + 1)) * _positive_Z(n, m, x, y)
    elif m == 0:
        return math.sqrt(n + 1) * _positive_Z(n, m, x, y)
    else:  # m < 0
        return math.sqrt(2 * (n + 1)) * _negative_Z(n, abs(m), x, y)
