"""Utils for format LOSSY conversion between std image and others data type.
Available formats:
    std sRGB (reference point)
        gamma-corrected
        3-channel rgb
        0-255 uint8

    psf (PSF, point spread function)
        linear
        monochrome
        autocontracted
        doctor view

    opt (optotype)
        linear
        clipped
        doctor view (flipped to patient view on vizualization)

"""

import numpy as np


def to_psf(img):
    """Convert usual image to psf-like data

    Parameters
    ----------
    img : np.array
        usual image (3-channel gamma-corrected uint8)

    Returns:
    --------
    data : np.array
        2d array with psf-like data (linear float32)
    """
    assert img.dtype == np.uint8
    assert len(img.shape) == 3

    img = _uint_to_float(img)
    img = _gamma_to_linear(img)
    img = _to_1ch(img)
    return img


def from_psf(data, float64_ok=False):
    """Autocontrast + convert psf-like data to usual image

    Parameters
    ----------
    data : np.array
        2d array with psf-like data (linear float32)

    Returns:
    --------
    img : np.array
        usual image (3-channel gamma-corrected uint8)
    """
    if float64_ok:
        assert data.dtype == np.float32 or data.dtype == np.float64
    else:
        assert data.dtype == np.float32

    assert len(data.shape) == 2

    data = _autocontrast(data)
    data = _linear_to_gamma(data)
    data = _to_3ch(data)
    data = _float_to_uint(data)
    return data


def to_opt(img):
    """Convert usual image to optotype-like data

    Parameters
    ----------
    img : np.array
        usual image (3-channel gamma-corrected uint8)

    Returns:
    --------
    data : np.array
        2d array with optotype-like data (linear float32 flipped)
    """
    return _flip(to_psf(img))


def from_opt(data, float64_ok=False):
    """Clip + convert optotype-like data to usual image

    Parameters
    ----------
    data : np.array
        2d array with optotype-like data (linear float32 flipped)

    Returns:
    --------
    img : np.array
        usual image (3-channel gamma-corrected uint8)
    """
    if float64_ok:
        assert data.dtype == np.float32 or data.dtype == np.float64
    else:
        assert data.dtype == np.float32

    assert len(data.shape) == 2

    data = _clip(data)
    data = _linear_to_gamma(data)
    data = _to_3ch(data)
    data = _float_to_uint(data)
    data = _flip(data)
    return data


def _autocontrast(data):
    if data.min() == data.max():
        return np.zeros_like(data)
    return (data - data.min()) / (data.max() - data.min())


def _clip(data):
    return np.clip(data, 0, 1)


def _gamma_to_linear(img):
    return img**2.2


def _linear_to_gamma(data):
    return data**(1/2.2)


def _to_3ch(data):
    return np.tile(data[:, :, np.newaxis], [1, 1, 3])


def _to_1ch(img):
    return np.mean(img, axis=2)


def _uint_to_float(img):
    return img.astype(np.float32) / 255


def _float_to_uint(data):
    return (data*255).astype(np.uint8)


def _flip(img):
    return img[:, ::-1]
