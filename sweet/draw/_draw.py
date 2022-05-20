from .letter import LetterDrawer
from .helper import pad, real_size
from pathlib import Path
import numpy as np

from ..util.data import defaults


DEFAULT_FONT_PATH = (Path(__file__).parents[2] / 'data' / 'fonts' / 'Optician-Sans.otf').resolve()


def get_square_chars(font=str(DEFAULT_FONT_PATH), size=128, tol=1/64):
    chars = "QWERTYUIOPASDFGHJKLZXCVBNM"
    squares = []
    for ch in chars:
        ld = LetterDrawer(ch, size, font, False)
        ld.draw()
        shape, _, _ = real_size(ld._draft)
        w, h = shape
        if abs(w - h) <= int(size * tol):
            squares.append(ch)
    return squares


def draw_char(char, size=defaults['N'], char_ratio=1/2, invert=False, font=str(DEFAULT_FONT_PATH), opt_fmt=True):
    ld = LetterDrawer(char, round(size * char_ratio), font)
    char_img = ld.draw(invert=invert)
    bg = 0 if invert else 1
    res_img = pad(char_img, (size, size), constant_values=bg)

    if opt_fmt:
        # planning to make by default
        # or use img_fmt (?)
        return res_img[:, ::-1, ...]
    return res_img




def draw_char_multiscale(char, size=defaults['N'], char_ratio=0.8, rows=5, invert=False):
    """Draw multiple scale of a same characters on a single image.

    Parameters
    ----------
    char : str
        the character to use
    size : int, default=1024
        size of the output image
    char_ratio : float, default=0.8
        the char ratio for draw_char function
    rows : int, default=5
        rows of scales
    invert : bool, default=False

    Returns
    -------
    np.array
        image of shape (size, size)
    """

    image = draw_char(char, size=size//2, char_ratio=char_ratio)
    imgs = []
    for k in range(rows):
        imgs.append(np.hstack(np.tile(image[np.newaxis, ::2**k, ::2**k], (2**k, 1, 1))))

    ret = pad(np.vstack(imgs), [size, size], constant_values=1)
    if invert:
        ret = 1 - ret
    return ret
