import glob
import os
import sys
import torch
import matplotlib.pyplot as plt

from model import CNN
from preprocessor import load_digits

# Load the trained model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()


def confidences(digits):
    """The model's confidence in its best guess for each digit image."""
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack(digits)), dim=1)
    return probs.max(dim=1).values.tolist()


def read_number(path):
    """
    Read every digit in an image, left to right.
    Returns (number as a string, confidence per digit, digit images).
    A single digit is just a one-digit number.
    """
    # The model helps decide where touching digits should be split
    digits = load_digits(path, scorer=confidences)   # list of (1, 28, 28) tensors
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack(digits)), dim=1)
    conf, pred = probs.max(dim=1)
    number = "".join(str(p) for p in pred.tolist())
    return number, conf.tolist(), digits


def read_files(paths):
    """Mode 1: print the number found in each image, e.g. "2026"."""
    for path in paths:
        number, conf, _ = read_number(path)
        print(f"{os.path.basename(path)}: {number}  "
              f"(confidence per digit: {', '.join(f'{c:.0%}' for c in conf)})")


def check_my_digits():
    """
    Mode 2: check every image in my_digits/. The file name gives the true
    number, e.g. "8_0.png" -> 8 and "2026_0.jpg" -> 2026.
    """
    files = sorted(
        f for ext in ("png", "jpg", "jpeg")
        for f in glob.glob(f"my_digits/*.{ext}")
    )

    fig, axes = plt.subplots(1, len(files), figsize=(2.5 * len(files), 2.5))
    if len(files) == 1:
        axes = [axes]

    correct = 0
    for ax, path in zip(axes, files):
        true_number = os.path.basename(path).split("_")[0]

        number, conf, digits = read_number(path)

        correct += (number == true_number)
        print(f"{os.path.basename(path)}: true={true_number} pred={number} "
              f"conf={', '.join(f'{c:.0%}' for c in conf)}")

        # Show the digits the model actually saw, side by side
        ax.imshow(torch.cat([d.squeeze(0) for d in digits], dim=1), cmap="gray")
        ax.set_title(f"{true_number}→{number}\n{min(conf):.0%}",
                     color="green" if number == true_number else "red")
        ax.axis("off")

    print(f"Accuracy: {correct}/{len(files)}")
    plt.show()


# Only run when started directly, so other files can import read_number
if __name__ == "__main__":
    if len(sys.argv) > 1:
        read_files(sys.argv[1:])        # python predict.py photo.jpg [more.jpg ...]
    else:
        check_my_digits()               # python predict.py