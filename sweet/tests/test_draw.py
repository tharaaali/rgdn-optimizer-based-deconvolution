import unittest
from sweet.draw import draw_char


class Test(unittest.TestCase):
    def test_font_available(self):
        draw_char('0', 1024)

    def test_font_size(self):
        for i in range(0, 10):
            letter = chr(ord('a') + i)
            self.assertEqual(
                draw_char(letter, 1024).shape,
                (1024, 1024)
            )

        # todo: FIX
        # self.assertEqual(
        #     draw_char('a', 1024).shape,
        #     (1024, 1024),
        #     char_ratio=0.9
        # )
