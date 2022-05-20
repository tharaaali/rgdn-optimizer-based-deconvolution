import numpy as np
from sweet.util.fft import fft, ifft


# Implementation of Inverse Blur Filter proposed by (3.29)
def inverse_blur_filtering(image, psf, K=0.01, real_output=True):
    assert image.shape == psf.shape, \
        f"Expected equal shapes, got: image={image.shape}, psf={psf.shape}"

    shape = image.shape
    otf = fft(psf, shape=shape)
    mtf = np.abs(otf)

    # normalize MTF & OTF
    mtf_max = mtf.max()  # change to mtf[0,0] once tested properly
    mtf = mtf / mtf_max
    otf = otf / mtf_max

    # we manually set values for ration in IBF as {num} / {denum}
    # to be able to handle zero division cases
    num = fft(image, shape=shape) * np.power(mtf, 2)
    denum = otf * (np.power(mtf, 2) + K)

    ratio = np.divide(num, denum, out=np.zeros_like(num), where=denum != 0)

    # the result is already shifted to center
    result = ifft(ratio, shape=shape)
    if real_output:
        return result.real
    else:  # complex output if requested
        return result
