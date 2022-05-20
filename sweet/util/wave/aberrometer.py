import numpy as np

from sweet.util.zernike import single_index, double_index, CartZernike, cartesian_zernike
from .wavefront import Wavefront


class AberrationWavefront(Wavefront):

    def __init__(self, D0, coeffs, fast_calc=True):
        assert D0 > 0, f"Incorrect value of pupil's diameter = {D0}"
        assert isinstance(coeffs, (list, dict, np.ndarray)), \
            f"Expected `list`, `dict` or `np.ndarray`, got {type(coeffs)}"
        if fast_calc:
            assert len(coeffs) <= 28, \
                f"When `fast_calc` is set to `True`, only predefined Zernike" \
                f"from Z0 to Z27 are used, so len(coeffs) is expected to be <= 28, " \
                f"but got {len(coeffs)}\n" \
                f"Hint: If Z28 and higher coeffs are needed, set `fast_calc to `False`"

        self._coeffs = coeffs
        self._fast_calc = fast_calc
        self._pupil_diam = D0
        self._viewing_dist = None

    def __eq__(self, other):
        return other.__class__ == self.__class__ \
            and self._coeffs == other._coeffs \
            and self._fast_calc == other._fast_calc \
            and self._pupil_diam == other._pupil_diam \
            and self._viewing_dist == other._viewing_dist

    def __hash__(self):
        return hash((str(self._coeffs), self._fast_calc, self._pupil_diam, self._viewing_dist))

    def coeffs(self):
        if type(self._coeffs) == dict:
            return [self._coeffs[f'Z{i}'] for i in range(len(self._coeffs))]
        else:
            return [*self._coeffs]

    def coeffs_dict(self):
        if type(self._coeffs) == dict:
            return self._coeffs.copy()
        else:
            return {f"Z{i}": coeff for i, coeff in enumerate(self._coeffs)}

    def calc(self, x, y):
        coeffs = self.coeffs()
        if self._viewing_dist:
            coeff_update = (100 / self._viewing_dist) * (self._pupil_diam ** 2) / (16 * np.sqrt(3))
            coeffs[single_index(2, 0)] -= coeff_update

        w = 0.0

        if self._fast_calc:
            zz = cartesian_zernike(x, y)
            for i, coeff in enumerate(coeffs):
                w += coeff * zz[i]
        else:  # recursive Zernike calculation
            for i, coeff in enumerate(coeffs):
                n, m = double_index(i)
                w += coeff * CartZernike(n, m, x, y)

        return w
