"""
Turn a photo, scan or drawing of handwritten digits into MNIST-style
1x28x28 tensors (white digit on black background, values 0..1).

Pipeline:
  1. Load as grayscale (respects phone EXIF rotation and transparent PNGs)
  2. Work out whether the ink is darker or lighter than the paper using the
     image border, and flip it so the ink is bright
  3. Flatten shadows / uneven lighting with a morphological top-hat, then
     pick the ink threshold automatically with Otsu
  4. Erase ruled notebook lines, keeping digit strokes that cross them
  5. Find connected components, drop specks, keep the digit(s)
  6. Deskew, normalise stroke width, fit into a 20x20 box and centre by mass
     in a 28x28 canvas (the same layout MNIST uses)
"""
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage

WORK_SIZE = 512          # longest side used for thresholding (keeps kernel sizes consistent)
DIGIT_BOX = 20           # MNIST digits fit in a 20x20 box...
CANVAS = 28              # ...centred by mass in a 28x28 image
TARGET_STROKE = 0.14     # typical MNIST stroke width / longest side of the digit
MIN_CONTRAST = 20        # top-hat response (0..255) below this is treated as paper
MAX_SKEW = 0.5           # clamp on the deskew shear so odd shapes aren't mangled
SPECK_FRACTION = 0.05    # components smaller than this fraction of the largest are noise
LINE_MIN_LENGTH = 0.4    # ruled lines: straight segments at least this fraction of the image width
LINE_MAX_TILT = 25       # degrees from horizontal; anything steeper is not a ruled line
LINE_VS_DIGIT = 1.5      # a ruled line is at least this many times longer than the tallest digit
LINE_MIN_RATIO = 15      # ...and at least this many times longer than it is thick
LINE_MAX_STROKE = 0.75   # ...and thinner than this x the pen stroke (printed lines are fine)
MERGE_GAP = 0.12         # broken digits: pieces this close (fraction of digit height) can be joined
MERGE_MAX_WIDTH = 0.9    # ...if together they are no wider than this x the digit height
FRAGMENT_HEIGHT = 0.7    # ...and one of them is shorter than this x the tallest digit
MERGE_MIN_AREA = 0.15    # ...and the smaller piece isn't a dot (area vs the larger piece)
TOUCHING_RATIO = 1.3     # without a scorer: a blob wider than this x the digit height is split
DIGIT_ASPECT = 0.6       # typical handwritten digit width / height, used to count touching digits
SPLIT_ASPECT = 1.0       # with a scorer: only blobs wider than this x the digit height are split
MIN_DIGIT_ASPECT = 0.2   # with a scorer: narrowest piece (width / height) that can be one digit
MAX_DIGIT_ASPECT = 1.3   # with a scorer: widest piece that can be one digit
WIDE_ASPECT = 0.85       # pieces wider than this are penalised: few digits are that wide
CUT_STEP = 0.06          # spacing of candidate cut positions, as a fraction of digit height


# --------------------
# 1. Loading
# --------------------

def _read_gray(filename):
    """Load any image file as a uint8 grayscale array."""
    img = ImageOps.exif_transpose(Image.open(filename))

    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        la = np.asarray(img.convert("RGBA").convert("LA"), dtype=np.float32) / 255.0
        lum, alpha = la[..., 0], la[..., 1]
        if alpha.min() < 1.0:
            # Composite onto a background that contrasts with what was drawn,
            # so black-on-transparent and white-on-transparent both work
            opaque = alpha > 0.5
            bg = 1.0 if opaque.any() and lum[opaque].mean() < 0.5 else 0.0
            return ((lum * alpha + bg * (1.0 - alpha)) * 255).astype(np.uint8)

    return np.asarray(img.convert("L"), dtype=np.uint8)


# --------------------
# 2-3. Binarisation that copes with shadows
# --------------------

def _binarize(gray, name="image"):
    """Return a uint8 mask (255 = ink) at WORK_SIZE resolution."""
    h, w = gray.shape
    scale = WORK_SIZE / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=interp)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Polarity: the border is mostly paper. If most of it falls on the bright
    # side of a global Otsu split, the ink is dark, so invert.
    _, split = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = max(2, min(gray.shape) // 30)
    border = np.concatenate([split[:b].ravel(), split[-b:].ravel(),
                             split[:, :b].ravel(), split[:, -b:].ravel()])
    if (border > 0).mean() > 0.5:
        gray = 255 - gray

    # Top-hat = image minus its morphological opening. An opening with a big
    # kernel erases thin strokes and keeps only the slowly varying lighting,
    # so subtracting it removes shadows and gradients.
    k = max(15, min(gray.shape) // 3) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ink = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    if ink.max() < MIN_CONTRAST:
        raise ValueError(f"No digit found in {name}")

    # Otsu picks the ink/paper split from the histogram; the floor stops it
    # from latching onto sensor noise when the ink is faint
    t, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (ink > max(t, MIN_CONTRAST)).astype(np.uint8) * 255

    # Close tiny gaps so a stroke isn't split into several components
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small)

    # Notebook lines come through as ink too: erase them before looking for digits
    return _remove_ruled_lines(mask)


# --------------------
# 4. Ruled-line removal
# --------------------

def _remove_ruled_lines(mask):
    """
    Erase ruled (notebook) lines from an ink mask, keeping digit strokes that
    cross them. Returns the mask unchanged if there are no ruled lines.

    1. A Hough transform finds long, nearly horizontal straight segments.
    2. Each segment is traced column by column (see _trace_line) to measure
       the line's exact edges and erase it, except where a digit crosses it.
    3. A candidate only counts as a ruled line if it is clearly thinner than
       the pen strokes (printed lines are fine, pen strokes are not), and much
       longer than the tallest digit. Those two checks stop the bottoms of
       touching digits, or the top bar of a tightly cropped 7, from being
       mistaken for a line.
    """
    H, W = mask.shape
    gap = max(5, int(0.03 * W))
    segments = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=max(20, int(0.2 * W)),
                               minLineLength=int(LINE_MIN_LENGTH * W), maxLineGap=gap)
    if segments is None:
        return mask

    ink = mask > 0

    # up[y, x] / down[y, x]: length of the unbroken vertical ink run that
    # ends / starts at (y, x). Used to measure how thick the ink is at a line.
    up = np.zeros((H + 1, W), np.int32)
    for y in range(H):
        up[y + 1] = (up[y] + 1) * ink[y]
    up = up[1:]
    down = np.zeros((H + 1, W), np.int32)
    for y in range(H - 1, -1, -1):
        down[y] = (down[y + 1] + 1) * ink[y]
    down = down[:H]

    # Running ink count down each column, for "any ink between rows a and b"
    column_sum = np.vstack([np.zeros((1, W), np.int32),
                            np.cumsum(ink, axis=0, dtype=np.int32)])

    # Longest first, so duplicates of a line already traced can be skipped
    segments = sorted(segments[:, 0].tolist(),
                      key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))[:30]

    candidates = []                              # (length, thickness, pixels to erase)
    covered = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in segments:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if x2 == x1 or abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) > LINE_MAX_TILT:
            continue
        traced = _trace_line(ink, up, down, column_sum, covered, gap, x1, y1, x2, y2)
        if traced is not None and traced[0].any():
            pixels, thickness = traced
            cols = np.flatnonzero(pixels.any(axis=0))
            candidates.append((int(cols[-1] - cols[0] + 1), thickness, pixels))
            covered |= pixels
    if not candidates:
        return mask

    # Pen stroke width, measured on what's left once every candidate is gone.
    # Ruled lines are printed thinner than pen strokes; a "line" as thick as
    # the strokes is really digits (e.g. the bottoms of touching digits).
    rest = (ink & ~covered).astype(np.uint8)
    if not rest.any():
        return mask                              # nothing but lines: leave it alone
    padded = cv2.copyMakeBorder(rest, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    stroke = 2 * float(np.percentile(dist[rest > 0], 90))
    thin = [c for c in candidates if c[1] <= LINE_MAX_STROKE * stroke]
    if not thin:
        return mask

    # How tall are the digits once the thin candidate lines are gone?
    thin_cover = np.zeros((H, W), bool)
    for _, _, pixels in thin:
        thin_cover |= pixels
    _, _, stats, _ = cv2.connectedComponentsWithStats(
        (ink & ~thin_cover).astype(np.uint8), connectivity=8)
    stats = stats[1:]
    if len(stats) == 0:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    tallest = stats[areas >= SPECK_FRACTION * areas.max(), cv2.CC_STAT_HEIGHT].max()

    erase = np.zeros((H, W), bool)
    for length, _, pixels in thin:
        if length >= LINE_VS_DIGIT * tallest:
            erase |= pixels
    if not erase.any():
        return mask
    return np.where(erase, 0, mask).astype(np.uint8)


def _trace_line(ink, up, down, column_sum, covered, gap, x1, y1, x2, y2):
    """
    Follow one Hough segment along the ink and return (pixels to erase,
    line thickness), or None if it doesn't look like a ruled line.

    In each column the vertical ink run through the line is measured. Most
    columns contain only the line, so the typical run length is its
    thickness; columns where a digit touches the line have longer runs and
    are left out when fitting the line's top and bottom edges. The line band
    is then erased only in columns where the ink is no thicker than the line
    itself. Where a stroke crosses the line, or lies along it (the flat
    bottom of a 2 written on a ruled line), the ink is thicker and is kept.
    The line is followed past the segment ends until it has been missing
    for `gap` columns.
    """
    H, W = ink.shape
    xs = np.arange(x1, x2 + 1)
    centre = np.round(y1 + (xs - x1) * (y2 - y1) / (x2 - x1)).astype(int)

    # Nearest ink pixel to the segment in each column (within 2 px)
    hit_y = np.full(len(xs), -1)
    for d in (0, -1, 1, -2, 2):
        y = np.clip(centre + d, 0, H - 1)
        new = (hit_y < 0) & ink[y, xs]
        hit_y[new] = y[new]
    has = hit_y >= 0
    if has.mean() < 0.6:
        return None                              # too broken up to be a ruled line
    hy, hx = hit_y[has], xs[has]
    if covered[hy, hx].mean() > 0.5:
        return None                              # a line that was already traced

    tops = hy - up[hy, hx] + 1
    bottoms = hy + down[hy, hx] - 1
    runs = bottoms - tops + 1
    thickness = float(np.median(runs))
    clean = runs <= 1.5 * thickness + 1          # columns with nothing but the line
    if clean.sum() < max(10, 0.5 * len(xs)) or (x2 - x1) / thickness < LINE_MIN_RATIO:
        return None

    # Fit the line's top and bottom edges, widened by a pixel for the fringe
    all_x = np.arange(W)
    top_edge = np.polyval(np.polyfit(hx[clean], tops[clean], 1), all_x) - 1
    bottom_edge = np.polyval(np.polyfit(hx[clean], bottoms[clean], 1), all_x) + 1
    on_image = (bottom_edge >= 0) & (top_edge <= H - 1)
    band_top = np.clip(np.floor(top_edge), 0, H - 1).astype(int)
    band_bottom = np.clip(np.ceil(bottom_edge), 0, H - 1).astype(int)

    in_band = on_image & (column_sum[band_bottom + 1, all_x] - column_sum[band_top, all_x] > 0)

    # A stroke crosses the line if there is solid ink (two rows, so the
    # line's own fringe doesn't count) both just above and just below it
    above = ((band_top >= 2) & ink[np.maximum(band_top - 1, 0), all_x]
             & ink[np.maximum(band_top - 2, 0), all_x])
    below = ((band_bottom <= H - 3) & ink[np.minimum(band_bottom + 1, H - 1), all_x]
             & ink[np.minimum(band_bottom + 2, H - 1), all_x])
    crossing = above & below

    # Thickest vertical ink run through the band in each column. Ink much
    # thicker than the line means a digit stroke is lying on it: keep it
    band_run = np.zeros(W, np.int32)
    for d in range(int((band_bottom - band_top).max()) + 1):
        y = np.minimum(band_top + d, band_bottom)
        run = np.where(ink[y, all_x], up[y, all_x] + down[y, all_x] - 1, 0)
        band_run = np.maximum(band_run, run)
    line_only = band_run <= thickness + max(2.0, 0.3 * thickness)

    # The segment itself, plus the line beyond its ends while it continues
    active = np.zeros(W, bool)
    active[x1:x2 + 1] = True
    for step, start in ((-1, x1 - 1), (1, x2 + 1)):
        misses, x = 0, start
        while 0 <= x < W and misses <= gap:
            if in_band[x]:
                active[x], misses = True, 0
            else:
                misses += 1
            x += step

    rows = np.arange(H)[:, None]
    pixels = ((rows >= band_top) & (rows <= band_bottom)
              & (active & in_band & line_only & ~crossing) & ink)
    return pixels, thickness


def _components(mask, name="image"):
    """Connected components as (labels, stats); stats row i is label i + 1."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:]                                  # drop the background
    if len(stats) == 0:
        raise ValueError(f"No digit found in {name}")
    return labels, stats


def _boxes(stats):
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    return x, y, x + stats[:, cv2.CC_STAT_WIDTH], y + stats[:, cv2.CC_STAT_HEIGHT]


def _ignore_edges(shape, stats):
    """
    Indices of components to consider. Components touching the image edge
    (table edges, paper edges) are skipped, unless that would throw away the
    digits themselves because the photo is cropped tightly around them.
    """
    H, W = shape
    areas = stats[:, cv2.CC_STAT_AREA]
    x0, y0, x1, y1 = _boxes(stats)
    touches = (x0 == 0) | (y0 == 0) | (x1 >= W) | (y1 >= H)
    candidates = np.flatnonzero(~touches)
    if len(candidates) == 0 or areas[candidates].max() < 0.2 * areas.max():
        candidates = np.arange(len(stats))
    return candidates


# --------------------
# 6. Shape normalisation
# --------------------

def deskew_image(a):
    """
    Shear `a` horizontally so the digit's main axis is vertical, using image
    moments. Keeps the array shape. Also used on MNIST during training so the
    model sees the same kind of input it gets at prediction time.
    """
    a = np.asarray(a, dtype=np.float32)
    m = cv2.moments(a)
    if m["m00"] == 0 or m["mu02"] < 1e-2:
        return a
    skew = float(np.clip(m["mu11"] / m["mu02"], -MAX_SKEW, MAX_SKEW))
    cy = m["m01"] / m["m00"]
    # x' = x - skew * (y - cy): rows above the centre move one way, rows below
    # the other, and the centre of mass stays where it was
    M = np.float32([[1, -skew, skew * cy], [0, 1, 0]])
    h, w = a.shape
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _tight_crop(mask):
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _normalize_stroke(crop):
    """Thicken thin pen strokes / thin fat marker strokes towards MNIST's width."""
    h, w = crop.shape
    # Pad with background first: a crop that is solid ink (e.g. a thin sliver
    # tried while splitting touching digits) has no background pixels, and
    # the distance transform would return infinity
    padded = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    width = 2 * np.percentile(dist[crop > 0], 90)
    target = TARGET_STROKE * max(h, w)

    if width < 0.7 * target:
        r = int(round((target - width) / 2))
        if r > 0:
            crop = cv2.copyMakeBorder(crop, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            crop = cv2.dilate(crop, kernel)
    elif width > 1.5 * target:
        # Erode gently: strokes vary in width and over-eroding breaks them apart
        r = int(round((width - target) / 4))
        if r > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            thinner = cv2.erode(crop, kernel)
            if thinner.any():
                crop = thinner
    return _tight_crop(crop)


def _to_mnist(digit_mask, deskew=True):
    """Binary mask containing one digit -> (1, 28, 28) float tensor."""
    crop = _tight_crop(digit_mask)

    if deskew:
        h, w = crop.shape
        pad = int(np.ceil(MAX_SKEW * h)) + 1          # room for the shear
        crop = cv2.copyMakeBorder(crop, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        crop = (deskew_image(crop) > 127).astype(np.uint8) * 255
        crop = _tight_crop(crop)

    crop = _normalize_stroke(crop)

    # Scale so the longer side is 20 px, keeping aspect ratio. INTER_AREA
    # averages pixels, which gives the soft anti-aliased edges MNIST has.
    h, w = crop.shape
    s = DIGIT_BOX / max(h, w)
    new_w, new_h = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    digit = cv2.resize(crop.astype(np.float32) / 255.0, (new_w, new_h), interpolation=interp)
    digit /= max(digit.max(), 1e-6)

    # Paste onto a 28x28 black canvas, then centre by centre of mass
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    top, left = (CANVAS - new_h) // 2, (CANVAS - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit
    cy, cx = ndimage.center_of_mass(canvas)

    # Centre by mass like MNIST, but never push part of the digit off the
    # canvas (top-heavy digits such as 7 would otherwise lose their tail)
    dy = float(np.clip(CANVAS / 2 - cy, 1 - top, CANVAS - 1 - (top + new_h)))
    dx = float(np.clip(CANVAS / 2 - cx, 1 - left, CANVAS - 1 - (left + new_w)))
    canvas = ndimage.shift(canvas, (dy, dx), order=1)
    canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32)

    return torch.from_numpy(canvas).unsqueeze(0)   # shape (1, 28, 28)


# --------------------
# Public API
# --------------------

def load_image(filename, deskew=True):
    """
    Load an image containing a single digit and convert it into an
    MNIST-style 1x28x28 tensor (white digit on black background, values 0..1).
    """
    mask = _binarize(_read_gray(filename), filename)
    labels, stats = _components(mask, filename)
    areas = stats[:, cv2.CC_STAT_AREA]
    x0, y0, x1, y1 = _boxes(stats)
    candidates = _ignore_edges(mask.shape, stats)

    # The digit is the largest component...
    main = candidates[np.argmax(areas[candidates])]

    # ...plus any sizeable fragments right next to it (a broken stroke, the
    # detached top of a 5), but not specks or anything further away
    mx = 0.25 * (x1[main] - x0[main])
    my = 0.25 * (y1[main] - y0[main])
    keep = [main]
    for i in candidates:
        if i == main or areas[i] < 0.1 * areas[main]:
            continue
        if (x0[i] < x1[main] + mx and x1[i] > x0[main] - mx and
                y0[i] < y1[main] + my and y1[i] > y0[main] - my):
            keep.append(i)

    digit_mask = np.isin(labels, np.array(keep) + 1).astype(np.uint8) * 255
    return _to_mnist(digit_mask, deskew)


def load_digits(filename, deskew=True, scorer=None):
    """
    Segment an image of a number (one or more digits written left to right
    on one line) into MNIST-style 1x28x28 tensors, returned left to right.

    Digits that touch each other come out as one wide blob and are split.
    If `scorer` is given (a function taking a list of digit tensors and
    returning the model's confidence for each), every plausible way of
    splitting a wide blob is tried and the one the model is most confident
    about wins. Without a scorer, a simple width rule is used instead.
    """
    mask = _binarize(_read_gray(filename), filename)
    labels, stats = _components(mask, filename)
    areas = stats[:, cv2.CC_STAT_AREA]
    x0, y0, x1, y1 = _boxes(stats)
    candidates = _ignore_edges(mask.shape, stats)

    # Drop specks, then walk left to right merging pieces that overlap
    # horizontally (fragments of the same digit stacked above each other)
    largest = areas[candidates].max()
    big = [i for i in candidates[np.argsort(x0[candidates])]
           if areas[i] >= SPECK_FRACTION * largest]
    groups = []
    for i in big:
        if groups:
            g = groups[-1]
            overlap = min(g["x1"], x1[i]) - max(g["x0"], x0[i])
            narrower = min(g["x1"] - g["x0"], x1[i] - x0[i])
            if overlap > 0.5 * narrower:
                g["labels"].append(i + 1)
                g["x0"], g["x1"] = min(g["x0"], x0[i]), max(g["x1"], x1[i])
                g["y0"], g["y1"] = min(g["y0"], y0[i]), max(g["y1"], y1[i])
                continue
        groups.append({"labels": [i + 1], "x0": x0[i], "x1": x1[i],
                       "y0": y0[i], "y1": y1[i]})

    # Join pieces of a digit that got broken apart, e.g. a 2 whose flat
    # bottom stroke sat on a ruled line and was erased along with it
    tallest = max(g["y1"] - g["y0"] for g in groups)
    groups = _merge_fragments(groups, areas, tallest)

    # Dots, dashes and underlines are much shorter than real digits
    tallest = max(g["y1"] - g["y0"] for g in groups)
    groups = [g for g in groups if g["y1"] - g["y0"] >= 0.4 * tallest]

    # The tallest blob is the best guess at the digit height (a short blob
    # can be a squat 4 or a 7 whose tail got cut off)
    tallest = float(max(g["y1"] - g["y0"] for g in groups))
    widths = [g["x1"] - g["x0"] for g in groups]
    single = [w for w in widths if w <= TOUCHING_RATIO * tallest]
    digit_width = DIGIT_ASPECT * tallest
    if single:
        digit_width = max(float(np.median(single)), 0.5 * tallest)

    digits = []
    for g, w in zip(groups, widths):
        piece = np.isin(labels, g["labels"]).astype(np.uint8) * 255

        if scorer is None:
            # Width rule: much wider than a digit is tall means touching digits
            n = int(round(w / digit_width)) if w > TOUCHING_RATIO * tallest else 1
            parts = _split_touching(piece, n) if n > 1 else [piece]
            digits.extend(_to_mnist(m, deskew) for m in parts)
            continue

        # A blob no wider than a digit is tall is one digit. Only wider blobs
        # (touching digits) go through the slower model-guided splitter,
        # which would otherwise sometimes cut a clean 0 or 4 in half.
        if w <= SPLIT_ASPECT * tallest:
            digits.append(_to_mnist(piece, deskew))
            continue

        digits.extend(_segment_by_recognition(piece, tallest, scorer, deskew))

    return digits


def _segment_by_recognition(piece, height, scorer, deskew):
    """
    Split one blob (one digit, or several touching digits) the way the model
    finds most convincing.

    Candidate cuts are placed every CUT_STEP * height columns, each snapped to
    the lightest column nearby (usually the thin joint between two digits).
    Every stretch between two cuts that is a plausible digit width is read
    by the model, then dynamic programming picks the chain of stretches
    covering the whole blob with the best total score.

    Score of a piece = log(model confidence) - penalty for being too wide.
    Log confidences add up, so an extra piece has to earn its place: cutting
    a clean 0 in half gives two shaky halves and loses. The width penalty
    stops the model from reading two touching digits as one fat digit.
    """
    xs = np.flatnonzero(piece.any(axis=0))
    left, right = int(xs.min()), int(xs.max()) + 1
    width = right - left
    ink = (piece[:, left:right] > 0).sum(axis=0)

    step = max(2, int(round(CUT_STEP * height)))
    cuts = {0, width}
    for c in range(step, width - step // 2, step):
        lo, hi = max(1, c - step // 2), min(width - 1, c + step // 2 + 1)
        if lo < hi:
            cuts.add(lo + int(np.argmin(ink[lo:hi])))
    cuts = sorted(cuts)

    # Every stretch between two cuts that could be a single digit
    min_w, max_w = MIN_DIGIT_ASPECT * height, MAX_DIGIT_ASPECT * height
    spans, images = [], []
    for i, a in enumerate(cuts):
        for b in cuts[i + 1:]:
            w = b - a
            if w > max_w:
                break
            if w < min_w and not (a == 0 and b == width):
                continue
            part = np.zeros_like(piece)
            part[:, left + a:left + b] = piece[:, left + a:left + b]
            part = _drop_slivers(part)
            if part.any():
                spans.append((a, b))
                images.append(_to_mnist(part, deskew))

    if not images:
        return [_to_mnist(piece, deskew)]

    score = {}
    for (a, b), c in zip(spans, scorer(images)):
        aspect = (b - a) / height
        penalty = ((aspect - WIDE_ASPECT) / 0.15) ** 2 if aspect > WIDE_ASPECT else 0.0
        score[(a, b)] = float(np.log(max(float(c), 1e-6))) - penalty

    # best[x] = (total score, pieces) for the best way to read columns 0..x
    best = {0: (0.0, [])}
    for b in cuts[1:]:
        options = [(best[a][0] + score[(a, b)], best[a][1] + [(a, b)])
                   for a in cuts if a < b and a in best and (a, b) in score]
        if options:
            best[b] = max(options, key=lambda t: t[0])

    if width not in best:
        return [_to_mnist(piece, deskew)]
    index = {s: i for i, s in enumerate(spans)}
    return [images[index[s]] for s in best[width][1]]


def _split_touching(piece, n):
    """
    Cut a blob of `n` touching digits into `n` masks, left to right. Each cut
    goes through the column with the least ink near where the boundary is
    expected, which is usually the thin joint between two digits.
    """
    xs = np.flatnonzero(piece.any(axis=0))
    left, width = xs.min(), xs.max() + 1 - xs.min()
    ink_per_column = (piece[:, left:left + width] > 0).sum(axis=0)

    cuts = [0]
    for k in range(1, n):
        target = k * width / n
        lo = int(max(cuts[-1] + 1, target - 0.25 * width / n))
        hi = int(min(width - 1, target + 0.25 * width / n)) + 1
        if lo < hi:
            cuts.append(lo + int(np.argmin(ink_per_column[lo:hi])))
    cuts.append(width)

    parts = []
    for a, b in zip(cuts, cuts[1:]):
        part = np.zeros_like(piece)
        part[:, left + a:left + b] = piece[:, left + a:left + b]
        part = _drop_slivers(part)
        if part.any():
            parts.append(part)
    return parts


def _drop_slivers(mask):
    """Remove small bits of a neighbouring digit left behind by a cut."""
    _, labels, stats = cv2.connectedComponentsWithStats(mask, connectivity=8)[:3]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) <= 1:
        return mask
    keep = np.flatnonzero(areas >= 0.15 * areas.max()) + 1
    return np.isin(labels, keep).astype(np.uint8) * 255


def _merge_fragments(groups, areas, tallest):
    """
    Join neighbouring pieces (left to right) that are really one broken digit,
    e.g. a 2 whose flat bottom stroke lay on a ruled line and was erased with it.

    Two neighbours are joined only when they are close together, at least
    one is clearly shorter than a full digit, the smaller one isn't a dot,
    and together they are still no wider than a single digit. Two whole
    digits (both full height, or too wide together) are never joined.
    """
    def area(g):
        return int(sum(areas[label - 1] for label in g["labels"]))

    merged = [groups[0]]
    for g in groups[1:]:
        last = merged[-1]
        gap = g["x0"] - last["x1"]                      # negative = overlapping
        width = max(last["x1"], g["x1"]) - min(last["x0"], g["x0"])
        shortest = min(last["y1"] - last["y0"], g["y1"] - g["y0"])
        small, large = sorted((area(last), area(g)))

        if (gap <= MERGE_GAP * tallest
                and width <= MERGE_MAX_WIDTH * tallest
                and shortest < FRAGMENT_HEIGHT * tallest
                and small >= MERGE_MIN_AREA * large):
            last["labels"] = last["labels"] + g["labels"]
            last["x0"], last["x1"] = min(last["x0"], g["x0"]), max(last["x1"], g["x1"])
            last["y0"], last["y1"] = min(last["y0"], g["y0"]), max(last["y1"], g["y1"])
        else:
            merged.append(g)
    return merged