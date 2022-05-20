import numpy as np

from .wavefront import Wavefront


class PrescriptionWavefront(Wavefront):

    def __init__(self, D0, A, C, S, Vd=12):
        assert D0 > 0, f"Incorrect pupil diameter param: {D0}"
        assert -180 <= A <= 180, f"Expected A in [-180; 180] degrees, got {A}"
        assert -3 <= C <= 0, f"Expected C in [-3; 0] D, got {C}"
        assert -20 <= S <= 20, f"Expected S in [-20; 20] D, got {S}"
        assert 0 <= Vd <= 20, f"Expected Vd in [0; 20] mm, got {Vd}"

        self._prescription = {"A": A, "C": C, "S": S, "Vd": Vd}
        self._pupil_diam = D0
        self._viewing_dist = None

    def __eq__(self, other):
        return other.__class__ == self.__class__ \
            and self._prescription == other._prescription \
            and self._pupil_diam == other._pupil_diam \
            and self._viewing_dist == other._viewing_dist

    def __hash__(self):
        return hash((str(self._prescription), self._pupil_diam, self._viewing_dist))

    def coeffs(self):
        R = self._pupil_diam / 2.0
        A = self._prescription["A"] * np.pi / 180
        C = self._prescription["C"]
        S = self._prescription["S"]
        Vd = self._prescription["Vd"]

        S = S / (1 - Vd * S / 1000)

        # if self._viewing_dist:
        #     S = S + 100.0 / self._viewing_dist

        c_22 = (R ** 2) * C * np.sin(2 * A) / (4 * np.sqrt(6))
        c02 = -(R ** 2) * (S + C / 2) / (4 * np.sqrt(3))
        c22 = (R ** 2) * C * np.cos(2 * A) / (4 * np.sqrt(6))

        return [c_22, c02, c22]

    def coeffs_dict(self):
        coeffs = [*self.coeffs()]
        return {"c-2_2": coeffs[0], "c0_2": coeffs[1], "c2_2": coeffs[2]}

    def calc(self, x, y):
        c_22, c02, c22 = self.coeffs()

        if self._viewing_dist:
            coeff_update = (100 / self._viewing_dist) * (self._pupil_diam ** 2) / (16 * np.sqrt(3))
            c02 -= coeff_update

        z_22 = 2 * np.sqrt(6) * x * y
        z02 = np.sqrt(3) * (2 * x ** 2 + 2 * y ** 2 - 1)
        z22 = np.sqrt(6) * (x ** 2 - y ** 2)

        return c_22 * z_22 + c02 * z02 + c22 * z22
