# Identifying Handwritten Digits

Reads handwritten numbers from photos. A PyTorch CNN trained on MNIST does the recognition. An OpenCV pipeline turns a real photo (shadows, lined paper, digits written close together) into clean, MNIST-style digit images first.

```
> python predict.py photo.jpg
photo.jpg: 3248  (confidence per digit: 78%, 57%, 65%, 97%)
```

## Results

| | |
|---|---|
| MNIST test accuracy | 99.60% |
| MNIST validation accuracy | 99.58% (best of 15 epochs) |
| Model size | 468,138 parameters, about 1.9 MB |

Multi-digit numbers were tested on 600 numbers (2 to 4 digits) built from MNIST test digits the model never trained on. "Exact" means every digit in the number was right.

| Spacing between digits | Exact |
|---|---|
| Spaced apart | 92% |
| Just touching | 88% |
| Overlapping slightly | 69% |
| Overlapping more | 53% |

## Setup

Requires Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The trained model (`mnist_cnn.pth`) is included, so you can start reading photos straight away.

## Usage

Run every command from the project folder.

**Read a photo**

```powershell
python predict.py photo.jpg            # one photo
python predict.py a.jpg b.png c.jpeg   # several
```

This prints the number found in each image, with the model's confidence for each digit. A low confidence shows which digit is most likely wrong.

**Check your own samples**

Put photos in `my_digits/`, named after the number they show followed by `_` and anything you like, e.g. `2026_0.jpg` or `7_phone.png`. Then run:

```powershell
python predict.py
```

This prints each result and the overall accuracy. It also opens a window showing the cleaned-up digits the model actually read. Titles are green when the whole number is right and red otherwise.

**Retrain the model**

```powershell
python traincnn.py       # 15 epochs (default)
python traincnn.py 25    # choose the number of epochs
```

MNIST downloads into `data/` automatically on the first run. On a laptop CPU an epoch takes 1 to 3 minutes. Only the epoch with the best validation accuracy is saved to `mnist_cnn.pth`, so the result never gets worse because of a late overfitted epoch.

**Run the tests**

```powershell
python -m unittest test_preprocessor -v
```

## How it works

### 1. Cleaning up the photo (`preprocessor.py`)

1. **Load** the image as grayscale. Phone rotation (EXIF) and transparent PNGs are handled.
2. **Ink vs paper.** The image border is mostly paper, so it decides whether the ink is dark-on-light or light-on-dark.
3. **Remove shadows** with a morphological top-hat, then pick the ink threshold automatically with Otsu's method.
4. **Erase ruled lines.** Long, straight, thin lines tilted up to 25° are found with a Hough transform. They're erased only where the ink is no thicker than the line, so pen strokes that cross or lie along a line are kept.
5. **Find the digits** as connected blobs of ink, ignoring specks, dots and anything touching the image edge.

### 2. Splitting the number into digits

Blobs are read left to right. A blob wider than it is tall is probably several digits that touch. The code tries many cut positions, lets the model read each piece, and keeps the split the model is most confident about overall. A slightly wide 0 doesn't get cut in half, because two shaky halves score worse than one confident 0. Pieces of a digit that got broken apart are joined back together.

### 3. Making each digit look like MNIST

Each digit is deskewed, given an MNIST-like stroke thickness, scaled to fit a 20×20 box, and centred by mass in a 28×28 image, the same layout MNIST uses.

### 4. The model (`model.py`)

```
Input 1×28×28
→ [Conv 3×3 → BatchNorm → ReLU] ×2, 32 channels → MaxPool   (32×14×14)
→ [Conv 3×3 → BatchNorm → ReLU] ×2, 64 channels → MaxPool   (64×7×7)
→ Dropout 0.25 → Linear 3136→128 → BatchNorm → ReLU
→ Dropout 0.5  → Linear 128→10
```

### 5. Training (`traincnn.py`)

- Random rotation (±15°), shift (±10%), scale (0.8–1.15×) and shear (±10°), so the model copes with digits that don't look exactly like MNIST.
- Training images are deskewed the same way photos are, so training and real use match.
- 6,000 training images are held out for validation, with a fixed seed so the split is the same every run.
- Adam (learning rate 0.001), halved when validation accuracy stops improving.
- Label smoothing (0.05), so confidence on unusual photos is more honest.

## Project structure

| File | Purpose |
|---|---|
| `predict.py` | Read numbers from photos (the script you run) |
| `preprocessor.py` | Photo → MNIST-style digit images |
| `model.py` | The CNN |
| `traincnn.py` | Training script |
| `test_preprocessor.py` | Tests, using generated "photos" |
| `mnist_cnn.pth` | Trained model weights |
| `my_digits/` | Your sample photos |
| `train.py` | Original simple fully connected network, kept for reference and not used |
| `requirements.txt` | Pinned dependencies |

## Tips for good photos

- Use a dark pen or marker. Plain paper works best, but lined paper is handled.
- Leave a little space between digits.
- Hold the phone straight above the page and crop close to the number.

## Limitations

- Reads one line of digits, left to right. Decimal points, commas and minus signs are ignored.
- Digits that overlap a lot are read correctly about half the time.
- Accuracy figures come from MNIST and generated images, not a large set of real photos.
- One unreadable or empty file in `my_digits/` stops `python predict.py`.
- Grid or squared paper hasn't been tested. The vertical grid lines are not removed.
