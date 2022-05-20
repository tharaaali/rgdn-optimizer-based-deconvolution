from sweet.util.io import to_psf, from_psf, to_opt, from_opt
import imageio
import numpy as np
from pathlib import Path
import unittest

IMAGE_PATH = Path(__file__).parent / 'test_images' / 'monochrome_3channel.png'


class Test(unittest.TestCase):
    def test_flip(self):
        image = imageio.imread(IMAGE_PATH)

        psf_like = to_psf(image)
        # black sqare on top right
        self.assertEqual   (np.mean(psf_like[:32, -32:]), 0)
        self.assertNotEqual(np.mean(psf_like[:32, :32 ]), 0)

        opt_like = to_opt(image)
        # black sqare on top left
        self.assertNotEqual(np.mean(opt_like[:32, -32:]), 0)
        self.assertEqual   (np.mean(opt_like[:32, :32 ]), 0)

    def test_same_save(self):
        image = imageio.imread(IMAGE_PATH)

        np.testing.assert_allclose(
            image,
            from_psf(to_psf(image)),
        ) # for MONOCHROME only

        np.testing.assert_allclose(
            image,
            from_opt(to_opt(image)),
        ) # for MONOCHROME only

        # data == to_opt(from_opt(data)) - may be false due to clipping
        # data == to_psf(from_psf(data)) - may be false due to autocontrast

    def test_linear(self):
        """Test gamma is correct.
        Check visually on the image (left square brightness = average 'chess')"""
        image = imageio.imread(IMAGE_PATH)
        data = to_psf(image)

        # top left is 0.5
        self.assertTrue(np.all(np.abs(data[:64, :64] - 0.5) < 1/255))
        # 'chess' rectangular is 0.5 on average
        self.assertTrue(abs(data[64:74, :].mean() - 0.5) < 1/255)
