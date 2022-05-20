import os.path as osp
import imageio
import numpy as np


def load_image(path, binary=True):
    # loads image from file
    # TODO: Add all sizes support

    assert osp.exists(path), f"{path} does not exist"
    image = imageio.imread(path, as_gray=True) / 255.
    if binary:
        image[image > 0.5] = 1.0
        image[image < 1.] = 0.0

    assert image.shape[0] == image.shape[1], \
        f"Incorrect image shape, expected to have (N, N), got {image.shape}"
    assert image.shape[0] % 2 == 0, \
        f"Currently only even sizes are supported"
    return image


# TODO: Remove later
def generate_h(size=128, invert=False):
    image = np.zeros(shape=(size, size)) if invert else np.ones(shape=(size, size))
    center = int(size / 2)
    h_size = size / 2
    step = h_size / 3
    xx = np.arange(center - (h_size / 2), center + (h_size / 2), dtype=int)
    for i in range(len(xx)):
        for j in range(len(xx)):
            if step <= i < step * 2:
                if j < step or j >= step * 2:
                    continue
            image[xx[j], xx[i]] = 1. if invert else 0.
    return image
