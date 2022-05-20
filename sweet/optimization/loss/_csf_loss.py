import numpy as np

from sweet.util.data import defaults

assert type(defaults['GPU_enabled']) is bool
if defaults['GPU_enabled']:
    from sweet.util.fft._fft_tf import fft, ifft, shift
    from tensorflow.math import reduce_max
    from tensorflow.math import reduce_sum
    import tensorflow as tf

else:
    from sweet.util.fft._fft import fft, ifft, shift
    from numpy import max as reduce_max
    from numpy import sum as reduce_sum

PARAMS = {
    'a': 0.8,
    'b': 1.,
    'c': 0.1,
}


def _coeffs_function(r, a, b, c, norm=True):
    return -a * np.exp(-r**2 / (c**2)) + b * np.exp(-r**2 / (4 * c**2))


def build_coeffs(width=defaults['N'], height=defaults['N'], params=PARAMS, **args):
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    rs = ((xs - (width/2))**2 + (ys - (height/2))**2) ** 0.5
    rs = rs / rs.max()

    scale_coeffs = _coeffs_function(rs, **params, **args)

    scale_coeffs = scale_coeffs / np.max(np.abs(scale_coeffs))  # normalize
    return scale_coeffs


class CSF_loss_max:
    def __init__(self, width=defaults['N'], height=defaults['N'], params=PARAMS, **args):
        sc = build_coeffs(width, height, params, **args).astype(np.complex64)
        self.coeffs_squared = shift(sc)**2

    def __call__(self, img1, img2):
        v1 = fft(img1)
        v2 = fft(img2)
        common = abs(ifft(v1 * v2 * self.coeffs_squared))
        return reduce_max(common)


class CSF_loss_L2:
    def __init__(self, width=defaults['N'], height=defaults['N'], params=PARAMS, **args):
        sc = build_coeffs(width, height, params, **args).astype(np.complex64)
        self.coeffs_squared = shift(sc)**2
        self.width = width
        self.height = height

    def __call__(self, img1, img2):
        assert img1.shape == (self.height, self.width)

        if defaults['GPU_enabled']:
            img1 = tf.complex(img1, tf.zeros_like(img1))
            img2 = tf.complex(img2, tf.zeros_like(img2))
        else:
            pass
            # not tested



        v1 = fft(img1)
        v2 = fft(img2)

        l2 = (reduce_sum(abs((v1 - v2)**2 * self.coeffs_squared)) / self.width / self.height) ** 0.5

        return l2

class CSF_weighted_image:
    def __init__(self, width=defaults['N'], height=defaults['N'], params=PARAMS, **args):
        self.coeffs = shift(build_coeffs(width, height, params, **args).astype(np.complex64))

    def __call__(self, img):
        return ifft(fft(img) * self.coeffs)

    # def __str__(self):
    #     return


class CSF_loss_easy_mse:
    def __init__(self, csf_balanced_image):
        self.csf = csf_balanced_image

    def __call__(self, img1, img2):
        csf1 = self.csf(img1)
        csf2 = self.csf(img2)
        mse = reduce_sum((csf1-csf2)**2) / img1.shape[0] / img1.shape[1]

        return mse


class CSF_loss_easy_correlation:
    def __init__(self, csf_balanced_image):
        self.csf = csf_balanced_image

    def __call__(self, img1, img2, reg=0):
        csf1 = self.csf(img1)
        csf2 = self.csf(img2)

        corr = reduce_sum(csf1*csf2) / (reduce_sum(csf1*csf1)**0.5 * reduce_sum(csf2*csf2)**0.5 + reg)
        return 1-corr
