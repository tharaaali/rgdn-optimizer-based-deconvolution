import math
import numpy as np
from numba import jit

"""
This module contains hardcoded implementation of Z0-Z27 (a total of 28)
Zernike polynomials for both polar and cartesian coordinates.
Hardcoded polynomials are optimized with numba's JIT compiler and run faster than recursive ones,
however, if recursive implementation is preferred or Z28 and higher polynomials are needed,
one can refer to corresponding modules (`cartesian.py` & `polar.py`) in the same package.
"""


@jit
def polar_zernike(p, phi):
    """
    Hardcoded implementation of Z0-Z27 Zernike polynomials for polar coords.

    Parameters
    ----------
    p : float or numpy.ndarray
        Radius (rho) value(-s)
    phi : float or numpy.ndarray
        Angle value(-s)
    Returns
    -------
    zz
        Tuple of first 28 Zernike polynomials
    """
    zz = (
        1,  # ---------------------------------------------------------------------------------- Z0
        2*p*np.sin(phi),  # -------------------------------------------------------------------- Z1
        2*p*np.cos(phi),  # -------------------------------------------------------------------- Z2
        math.sqrt(6)  * (p**2)*np.sin(2*phi),  # ----------------------------------------------- Z3
        math.sqrt(3)  * (2*p**2 - 1),  # ------------------------------------------------------- Z4
        math.sqrt(6)  * (p**2)*np.cos(2*phi),  # ----------------------------------------------- Z5
        math.sqrt(8)  * (p**3)*np.sin(3*phi),  # ----------------------------------------------- Z6
        math.sqrt(8)  * (3*p**3 - 2*p)*np.sin(phi),  # ----------------------------------------- Z7
        math.sqrt(8)  * (3*p**3 - 2*p)*np.cos(phi),  # ----------------------------------------- Z8
        math.sqrt(8)  * (p**3)*np.cos(3*phi),  # ----------------------------------------------- Z9
        math.sqrt(10) * (p**4)*np.sin(4*phi),  # ----------------------------------------------- Z10
        math.sqrt(10) * (4*p**4 - 3*p**2)*np.sin(2*phi),  # ------------------------------------ Z11
        math.sqrt(5)  * (6*p**4 - 6*p**2 + 1),  # ---------------------------------------------- Z12
        math.sqrt(10) * (4*p**4 - 3*p**2)*np.cos(2*phi),  # ------------------------------------ Z13
        math.sqrt(10) * (p**4)*np.cos(4*phi),  # ----------------------------------------------- Z14
        math.sqrt(12) * (p**5)*np.sin(5*phi),  # ----------------------------------------------- Z15
        math.sqrt(12) * (5*p**5 - 4*p**3)*np.sin(3*phi),  # ------------------------------------ Z16
        math.sqrt(12) * (10*p**5 - 12*p**3 + 3*p)*np.sin(phi),  # ------------------------------ Z17
        math.sqrt(12) * (10*p**5 - 12*p**3 + 3*p)*np.cos(phi),  # ------------------------------ Z18
        math.sqrt(12) * (5*p**5 - 4*p**3)*np.cos(3*phi),  # ------------------------------------ Z19
        math.sqrt(12) * (p**5)*np.cos(5*phi),  # ----------------------------------------------- Z20
        math.sqrt(14) * (p**6)*np.sin(6*phi),  # ----------------------------------------------- Z21
        math.sqrt(14) * (6*p**6 - 5*p**4)*np.sin(4*phi),  # ------------------------------------ Z22
        math.sqrt(14) * (15*p**6 - 20*p**4 + 6*p**2)*np.sin(2*phi),  # ------------------------- Z23
        math.sqrt(7)  * (20*p**6 - 30*p**4 + 12*p**2 - 1),  # ---------------------------------- Z24
        math.sqrt(14) * (15*p**6 - 20*p**4 + 6*p**2)*np.cos(2*phi),  # ------------------------- Z25
        math.sqrt(14) * (6*p**6 - 5*p**4)*np.cos(4*phi),  # ------------------------------------ Z26
        math.sqrt(14) * (p**6)*np.cos(6*phi),  # ----------------------------------------------- Z27
    )
    return zz


@jit
def cartesian_zernike(x, y):
    """
    Hardcoded implementation of Z0-Z27 Zernike polynomials for cartesian coords.

    Parameters
    ----------
    x : float or numpy.ndarray
        X coord(-s)
    y : float or numpy.ndarray
        Y coord(-s)
    Returns
    -------
    zz
        Tuple of first 28 Zernike polynomials
    """
    zz = (
        1,  # ---------------------------------------------------------------------------------- Z0
        2*y,  # -------------------------------------------------------------------------------- Z1
        2*x,  # -------------------------------------------------------------------------------- Z2
        math.sqrt(6)  * 2*x*y,  # -------------------------------------------------------------- Z3
        math.sqrt(3)  * (2*(x**2 + y**2) - 1),  # ---------------------------------------------- Z4
        math.sqrt(6)  * (x**2 - y**2),  # ------------------------------------------------------ Z5
        math.sqrt(8)  * y*(3*x**2 - y**2),  # -------------------------------------------------- Z6
        math.sqrt(8)  * y*(3*(x**2 + y**2) - 2),  # -------------------------------------------- Z7
        math.sqrt(8)  * x*(3*(x**2 + y**2) - 2),  # -------------------------------------------- Z8
        math.sqrt(8)  * x*(x**2 - 3*y**2),  # -------------------------------------------------- Z9
        math.sqrt(10) * 4*x*y*(x**2 - y**2),  # ------------------------------------------------ Z10
        math.sqrt(10) * 2*x*y*(4*(x**2 + y**2) - 3),  # ---------------------------------------- Z11
        math.sqrt(5)  * (6*(x**2 + y**2)**2 - 6*(x**2 + y**2) + 1),  # ------------------------- Z12
        math.sqrt(10) * (x**2 - y**2)*(4*(x**2 + y**2) - 3),  # -------------------------------- Z13
        math.sqrt(10) * (x**4 - 6*(x**2)*(y**2) + y**4),  # ------------------------------------ Z14
        math.sqrt(12) * y*(5*x**4 - 10*(x**2)*(y**2) + y**4),  # ------------------------------- Z15
        math.sqrt(12) * y*(3*x**2 - y**2)*(5*(x**2 + y**2) - 4),  # ---------------------------- Z16
        math.sqrt(12) * y*(10*(x**2 + y**2)**2 - 12*(x**2 + y**2) + 3),  # --------------------- Z17
        math.sqrt(12) * x*(10*(x**2 + y**2)**2 - 12*(x**2 + y**2) + 3),  # --------------------- Z18
        math.sqrt(12) * x*(x**2 - 3*y**2)*(5*(x**2 + y**2) - 4),  # ---------------------------- Z19
        math.sqrt(12) * x*(x**4 - 10*(x**2)*(y**2) + 5*y**4),  # ------------------------------- Z20
        math.sqrt(14) * 2*x*y*(3*x**4 - 10*(x**2)*(y**2) + 3*y**4),  # ------------------------- Z21
        math.sqrt(14) * 4*x*y*(x**2 - y**2)*(6*(x**2 + y**2) - 5),  # -------------------------- Z22
        math.sqrt(14) * 2*x*y*(15*(x**2 + y**2)**2 - 20*(x**2 + y**2) + 6),  # ----------------- Z23
        math.sqrt(7)  * (20*(x**2 + y**2)**3 - 30*(x**2 + y**2)**2 + 12*(x**2 + y**2) - 1),  # - Z24
        math.sqrt(14) * (x**2 - y**2)*(15*(x**2 + y**2)**2 - 20*(x**2 + y**2) + 6),  # --------- Z25
        math.sqrt(14) * (x**4 - 6*(x**2)*(y**2) + y**4)*(6*(x**2 + y**2) - 5),  # -------------- Z26
        math.sqrt(14) * (x**6 - 15*(x**4)*(y**2) + 15*(x**2)*(y**4) - y**6),  # ---------------- Z27
    )
    return zz