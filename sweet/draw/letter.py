import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .helper import approx_size, pad, real_size


def _maximum_fitting_font(char, size, font_path):
    # Helper func to estimate the maximum font size
    # that fits in a given area of (size x size)

    font_size = 20 if size > 16 else 2
    font = ImageFont.truetype(font_path, font_size)
    w, h = font.getsize(char)

    while w < size and h < size:
        font_size += 2
        font = ImageFont.truetype(font_path, font_size)
        w, h = font.getsize(char)

    return font, w, h


class LetterDrawer():
    def __init__(self, character, letter_size, font_path, estim_size=True):
        assert character, "Got incorrect or empty character"

        self._char = character
        self._size = approx_size(letter_size) if estim_size else letter_size
        self._font = font_path

    def draw(self, invert=False):
        background, char_color = (0, 255) if invert else (255, 0)

        font, font_width, font_height = _maximum_fitting_font(
            self._char,
            self._size,
            self._font
        )

        image_size = 2 + max((font_width, font_height))
        image = Image.new('L', (image_size, image_size), color=background)
        draw = ImageDraw.Draw(image)

        x = int((image_size - font_width) / 2)
        y = int((image_size - font_height) / 2)
        draw.text((x, y), self._char, char_color, font=font)

        temp_image = np.array(image) / 255
        temp_image[temp_image > 0.5] = 1.0
        temp_image[temp_image < 1.0] = 0.0

        self._draft = temp_image.copy()

        shape, ii, jj = real_size(temp_image, inverted=invert)

        letter_color = 0 if invert else 1
        self._drawing = pad(
            temp_image[jj[0]:jj[1],
            ii[0]:ii[1]],
            (max(shape), max(shape)),
            letter_color
        )
        return self._drawing
