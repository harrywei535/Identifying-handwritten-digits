import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage


def load_image(filename):
    """
    Load an image and convert it into an MNIST-style 1x28x28 tensor
    (white digit on black background, values 0..1).
    """
    # Grayscale + stretch contrast
    img = ImageOps.autocontrast(Image.open(filename).convert("L"))
    a = np.asarray(img, dtype=np.float32) / 255.0

    # Make the digit white on black (only inverts if the background is bright)
    if np.median(a) > 0.5:
        a = 1.0 - a

    # Remove faint noise
    a[a < 0.25] = 0.0

    # Crop tightly around the digit
    ys, xs = np.where(a > 0.3)
    if len(ys) == 0:
        raise ValueError(f"No digit found in {filename}")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Scale so the longer side is 20 px, keeping aspect ratio
    h, w = a.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray((a * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.LANCZOS
    )
    digit = np.asarray(digit, dtype=np.float32) / 255.0
    
    # Stretch brightness so the strokes are fully white
    digit = digit / digit.max()

    # Estimate stroke width at 20px scale and thicken if it's thinner than MNIST
    mask = digit > 0.3
    dist = ndimage.distance_transform_edt(mask)
    stroke_width = 2 * np.percentile(dist[mask], 90)

    if stroke_width < 2.5:
        digit = ndimage.grey_dilation(digit, size=(2, 2))

    # Paste onto a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    # Center by center of mass
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1)

    return torch.from_numpy(canvas).unsqueeze(0)   # shape (1, 28, 28)