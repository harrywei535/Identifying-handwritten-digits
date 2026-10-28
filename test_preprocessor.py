"""
Checks for preprocessor.py using synthetic "photos".
Run with:  python -m unittest test_preprocessor -v
"""
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from preprocessor import deskew_image, load_digits, load_image


def draw(text, size=(400, 300), ink=30, paper=220, thickness=12, scale=6.0):
    """Grayscale image with `text` drawn roughly in the middle."""
    h, w = size[1], size[0]
    img = np.full((h, w), paper, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thickness, cv2.LINE_AA)
    return img


def draw_touching(text, overlap=0.3, height=300, ink=30, paper=220, thickness=12, scale=5.0):
    """Draw each character so it overlaps its neighbour, like digits written too close."""
    sizes = [cv2.getTextSize(c, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for c in text]
    step = [int(w * (1 - overlap)) for w, _ in sizes]
    width = sum(step) + sizes[-1][0] + 80
    img = np.full((height, width), paper, np.uint8)
    x = 40
    for c, (w, h), s in zip(text, sizes, step):
        cv2.putText(img, c, (x, (height + h) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, ink, thickness, cv2.LINE_AA)
        x += s
    return img


def add_shadow(img):
    """Darken one side heavily, like a phone casting a shadow over the page."""
    h, w = img.shape
    ramp = np.linspace(0.35, 1.0, w, dtype=np.float32)[None, :]
    return (img.astype(np.float32) * ramp).astype(np.uint8)


class PreprocessorTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, array, name):
        path = os.path.join(self.tmp.name, name)
        Image.fromarray(array).save(path)
        return path

    def assert_mnist_like(self, t):
        self.assertEqual(tuple(t.shape), (1, 28, 28))
        a = t.squeeze().numpy()
        self.assertAlmostEqual(float(a.max()), 1.0, places=3)
        self.assertGreaterEqual(float(a.min()), 0.0)
        # Background is black and the digit never touches the edge of the canvas
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        self.assertLess(float(border.max()), 0.2)
        # Digit occupies a sensible fraction of the canvas, centred by mass
        frac = float((a > 0.5).mean())
        self.assertGreater(frac, 0.05)
        self.assertLess(frac, 0.45)
        ys, xs = np.nonzero(a > 0.5)
        self.assertLess(abs(ys.mean() - 14), 3)
        self.assertLess(abs(xs.mean() - 14), 3)

    def test_dark_ink_on_paper(self):
        self.assert_mnist_like(load_image(self.save(draw("7"), "plain.png")))

    def test_shadow_does_not_become_ink(self):
        clean = load_image(self.save(draw("3"), "clean.png"))
        shadow = load_image(self.save(add_shadow(draw("3")), "shadow.png"))
        self.assert_mnist_like(shadow)
        # Same digit with and without the shadow should give nearly the same image
        self.assertLess(float((clean - shadow).abs().mean()), 0.05)

    def test_specks_do_not_stretch_the_crop(self):
        clean = load_image(self.save(draw("1"), "one.png")).squeeze().numpy()
        noisy = draw("1")
        rng = np.random.default_rng(0)
        for _ in range(15):
            y, x = rng.integers(5, 295), rng.integers(5, 395)
            cv2.circle(noisy, (int(x), int(y)), 2, 30, -1)
        speckled = load_image(self.save(noisy, "specks.png")).squeeze().numpy()
        self.assertLess(float(np.abs(clean - speckled).mean()), 0.05)

    def test_light_ink_on_dark_background(self):
        a = load_image(self.save(draw("5"), "dark_on_light.png")).squeeze().numpy()
        b = load_image(self.save(draw("5", ink=230, paper=25), "light_on_dark.png")).squeeze().numpy()
        self.assertLess(float(np.abs(a - b).mean()), 0.05)

    def test_transparent_png(self):
        rgba = np.zeros((300, 400, 4), np.uint8)
        rgba[..., 3] = 255 - draw("8", ink=0, paper=255)     # black strokes, transparent elsewhere
        self.assert_mnist_like(load_image(self.save(rgba, "transparent.png")))

    def test_blank_image_raises(self):
        with self.assertRaises(ValueError):
            load_image(self.save(np.full((200, 200), 200, np.uint8), "blank.png"))

    def test_multi_digit_segmentation(self):
        img = add_shadow(draw("2026", size=(900, 300), scale=5.0, thickness=10))
        digits = load_digits(self.save(img, "number.png"))
        self.assertEqual(len(digits), 4)
        for d in digits:
            self.assert_mnist_like(d)
        # The two 2s should look alike, and different from the 0
        self.assertLess(float((digits[0] - digits[2]).abs().mean()), 0.05)
        self.assertGreater(float((digits[0] - digits[1]).abs().mean()), 0.05)

    @unittest.skipUnless(os.path.exists("mnist_cnn.pth"), "needs a trained model")
    def test_touching_digits_are_split(self):
        from predict import read_number

        # Overlap chosen per number so its digits really do run together
        for text, overlap in (("2026", 0.35), ("5890", 0.3), ("347", 0.3), ("1234", 0.3)):
            img = draw_touching(text, overlap=overlap)
            # Sanity check: the digits really are joined into fewer blobs
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            blobs = cv2.connectedComponents(binary, connectivity=8)[0] - 1
            self.assertLess(blobs, len(text), text)

            number, _, digits = read_number(self.save(add_shadow(img), f"touching_{text}.png"))
            self.assertEqual(number, text)
            for d in digits:
                self.assert_mnist_like(d)

    @unittest.skipUnless(os.path.exists("mnist_cnn.pth"), "needs a trained model")
    def test_reads_separate_and_single_digit_numbers(self):
        from predict import read_number

        for text in ("7", "40", "2026", "13579"):
            img = add_shadow(draw(text, size=(260 * len(text) + 140, 300), scale=5.0, thickness=10))
            number, _, _ = read_number(self.save(img, f"number_{text}.png"))
            self.assertEqual(number, text)

    def test_single_digit_is_one_digit(self):
        for text in ("0", "4", "8"):
            digits = load_digits(self.save(draw(text), f"single_{text}.png"))
            self.assertEqual(len(digits), 1, text)

    def test_deskew_straightens_a_slanted_stroke(self):
        a = np.zeros((28, 28), np.float32)
        cv2.line(a, (9, 24), (19, 4), 1.0, 3)                # leans right
        m_before = cv2.moments(a)
        m_after = cv2.moments(deskew_image(a))
        self.assertGreater(abs(m_before["mu11"]), 5 * abs(m_after["mu11"]))


if __name__ == "__main__":
    unittest.main()
