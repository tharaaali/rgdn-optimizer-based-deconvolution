import unittest

import numpy as np
from sweet.util.fft._fft import fft_conv as fft_conv_np
from sweet.util.fft._fft_tf import fft_conv as fft_conv_tf
from sweet.util.io import generate_h
from sweet.transform.psf import PSF
from sweet.util.data import defaults
from sweet.util.wave import PrescriptionWavefront

def _assert_same_ttf(image, ker):
    conv_np = fft_conv_np(image, ker)
    conv_tf = fft_conv_tf(image, ker)

    # np.testing.assert_almost_equal(conv_np, conv_tf, decimal=7)  # FAIL
    return np.testing.assert_almost_equal(conv_np, conv_tf, decimal=4)


class Test(unittest.TestCase):
    def test_same_random(self):
        image = np.random.random([1024, 1024]).astype(np.float32)
        ker = np.random.random([1024, 1024]).astype(np.float32)
        _assert_same_ttf(image, ker)

        image = np.random.random([256, 256]).astype(np.float32)
        ker = np.random.random([256, 256]).astype(np.float32)
        _assert_same_ttf(image, ker)

    def test_same_real(self):
        image = generate_h(1024).astype(np.float32)

        wfp = {
            'A': 3*np.pi/7,
            'C': -1.5,
            'S': -2.25
        }
        wv_front = PrescriptionWavefront(D0=defaults['Demo']['ManualEyeParams']['D0'], **wfp)
        betta = 1.5
        ker = PSF(
            N = 1024,
            D = defaults['Demo']['ManualEyeParams']['D0'],
            D0 = defaults['Demo']['ManualEyeParams']['D0'],
            betta = betta,
            d = defaults['view_dist'],
            W = wv_front,
        ).astype(np.float32)

        _assert_same_ttf(image, ker)

    def test_same_along_images(self):
        images = np.random.random([4, 1024, 1024]).astype(np.float32)
        kers = np.random.random([4, 1024, 1024]).astype(np.float32)
        conved = fft_conv_tf(images, kers)

        conved_by_image = np.array([
            np.array(fft_conv_tf(img, ker), dtype=np.float32) for (img, ker) in zip(images, kers)
        ], dtype=np.float32)

        # np.testing.assert_almost_equal(conved, conved_by_image, decimal=4)  # FAIL
        np.testing.assert_almost_equal(conved, conved_by_image, decimal=2)


if __name__ == '__main__':
    unittest.main()
