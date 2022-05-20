import numpy as np


def approx_size(expected):
    # Helper func to estimate the size of character canvas
    # so that the area of character matches the expected size
    if expected <= 128:
        return round(expected * 1.4)
    else:
        return round(expected * 1.39)


def real_size(canvas, inverted=False):
    # Helper func to get the real size of character's area on a given canvas
    # The image (canvas) is expected to be a binary one
    # with black color for char and white for background (and vice versa for inverted)
    assert len(np.unique(canvas)) == 2, f"Expected binary image, got {len(np.unique(canvas))} values"

    if inverted:
        ax0 = np.arange(0, canvas.shape[0])[np.sum(canvas, axis=0) > 0]
        ax1 = np.arange(0, canvas.shape[1])[np.sum(canvas, axis=1) > 0]
    else:
        ax0 = np.arange(0, canvas.shape[0])[np.sum(canvas, axis=0) != canvas.shape[0]]
        ax1 = np.arange(0, canvas.shape[1])[np.sum(canvas, axis=1) != canvas.shape[1]]

    ii = ax0[0]
    jj = ax1[0]
    width = len(ax0)
    height = len(ax1)
    return (width, height), (ii, ii + width), (jj, jj + height)


def pad(img, shape, constant_values=0):
    # Helper func to pad the character image to have a square image
    w, h = shape
    top = np.floor((h - img.shape[0]) / 2).astype(int)
    bottom = np.ceil((h - img.shape[0]) / 2).astype(int)
    right = np.ceil((w - img.shape[1]) / 2).astype(int)
    left = np.floor((w - img.shape[1]) / 2).astype(int)

    # Sometimes we may have img.shape values >= (h,w) so that paddings can be < 0
    # In that case we apply zero paddings
    paddings = (
        (top if top >= 0 else 0, bottom if bottom >= 0 else 0),
        (left if left >= 0 else 0, right if right >= 0 else 0)
    )

    return np.pad(img, paddings, mode='constant', constant_values=constant_values)
