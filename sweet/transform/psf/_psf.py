import numpy as np
from sweet.util.fft import fft, shift
from sweet.util.wave import Wavefront
from functools import lru_cache
from skimage.transform import downscale_local_mean

import logging
logger = logging.getLogger(__name__)

def PSF_params(h, k, d):
    """
    Helper func to obtain angle params for PSF

    :param h: [cm] optotype height & width
    :param k: `canvas / optotype` ratio
    :param d: [cm] viewing distance
    :return:
        alpha: [deg] optotype angular size
        betta: [deg] canvas angular size
    """

    assert h > 0, f"Incorrect optotype height param = {h}"
    assert d > 0, f"Incorrect viewing distance param = {d}"
    assert k >= 1, f"Incorrect ratio param = {k}"

    alpha = np.arctan2(h, 2 * d) * 360 / np.pi
    betta = np.arctan2(k * h, 2 * d) * 360 / np.pi
    return alpha, betta


def _downscale_to_center(img, scale):
    """downscale to center"""
    res = np.zeros_like(img)
    s = len(img)//2 - (len(img)//scale)//2
    e = s + len(img)//scale

    res[s:e, s:e] = img[::scale, ::scale]
    return res


@lru_cache(50)
def cached_get_values(W, lam, N, R0, w_multiplicator=1.0):
    """many unused parameters for correct caching"""
    logger.debug("Wavefront values not found in cache, calculating...")

    def get_value(x, y):
        return np.exp(1j * 2 * np.pi * W.calc(x,y) / lam * w_multiplicator)

    ii, jj = np.meshgrid(np.arange(-N / 2, N / 2), np.flip(np.arange(-N / 2, N / 2)))
    return get_value(ii/R0, jj/R0)


@lru_cache(256)
def PSF(
    D,
    D0,
    betta,
    d,
    W: Wavefront,
    N=1024,
    lam=0.5876,
    norm=True,
    calc_scale=1,
    w_multiplicator=1.0
):
    """
    PSF generation

    :param D: [mm] pupil's diameter from eye tracker
    :param D0: [mm] pupil's diameter from aberrometer
    :param betta: [deg] canvas angular size
    :param d: [cm] viewing distance
    :param W: wavefront func (`AberrationWavefront` or `PrescriptionWavefront`)
    :param N: [points] canvas size
    :param lam: [μm] wavelength (default = 0.5876 μm)
    :param norm: [bool] flag to normalize output
    :param calc_scale: [int] calculation downscale coefficient useful for smaller N
    """

    N *= calc_scale

    assert D <= D0, f"Expected D <= D0, got D={D}, D0={D0}"
    assert d > 0, f"Incorrect viewing distance param = {d}"
    assert betta > 0, f"Incorrect canvas angular size value = {betta}"
    assert isinstance(W, Wavefront), f"Expected W to be instance of Wavefront subclasses"

    W.update(view_dist=d)

    R = 25 * np.pi * betta * D / (9 * lam)
    R0 = 25 * np.pi * betta * D0 / (9 * lam)

    dup_mode = False
    if 2*R >= N:
        dup_mode = True
        logger.warning(f'Big 2*R={2*R}, falling to dup_mode')
        R /= 2
        R0 /= 2
    assert 2*R < N, f"Incorrect parameters, the psf can't be calculated correctly, R={R} < 0.5N, consider increasing N={N} or calc_scale={calc_scale} or dup_mode={dup_mode}"

    values = cached_get_values(W, lam, N, R0, w_multiplicator) #, R0, betta, d, N, norm, calc_scale, D)

    ii, jj = np.meshgrid(np.arange(-N / 2, N / 2), np.flip(np.arange(-N / 2, N / 2)))
    mask = ii ** 2 + jj ** 2 <= R ** 2
    p = np.zeros((N, N)).astype(complex)
    p[mask] = values[mask]

    psf = np.power(np.abs(fft(p)), 2)
    if dup_mode:
        _downscale_to_center(psf, 2)

    if calc_scale != 1:
        psf = downscale_local_mean(psf, (calc_scale, calc_scale))

    if norm:
        psf = psf / np.max(psf)  # Max is the first ([0,0]) element
    return psf
