from sweet.util.fft import fft_conv
from .scale import scale_intensity_coeffs
from sweet.transform.hist.histogram import scale


# Helper func to simulate retinal image
# Retinal image is obtained from convolution of input image with PSF
def retinal(image, psf):
    return fft_conv(image, psf)


# Scaled version of retinal simulation
# Based on formula (3.34) in baseline paper
# FIXME: doesn't work as of Jul. 2021
def scaled_retinal(image, psf, d=(0, 1)):
    alpha, betta = scale_intensity_coeffs(image, d=d)
    return fft_conv(alpha * image, psf) + betta*psf
