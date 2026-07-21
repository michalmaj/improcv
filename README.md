# improcv

Modern image-processing and computer-vision utilities for Python, NumPy and OpenCV.

`improcv` provides small, well-typed, well-tested helpers for common OpenCV tasks, supporting
both OpenCV 4.x and OpenCV 5.x.

## Installation

`pip install improcv` alone installs NumPy but **not** OpenCV — `import improcv` will fail with a
clear error telling you to install one. Pick exactly one variant:

```bash
pip install "improcv[cv]"                  # opencv-python
pip install "improcv[cv-headless]"         # opencv-python-headless
pip install "improcv[cv-contrib]"          # opencv-contrib-python
pip install "improcv[cv-contrib-headless]" # opencv-contrib-python-headless
```

Already have one of these installed under a different name, or building OpenCV yourself? Just
`pip install improcv` and install/keep your existing OpenCV — improcv only needs `cv2` importable,
it doesn't care how it got there.

## Usage

```python
import cv2
import improcv as im

image = cv2.imread("photo.jpg")
resized = im.resize(image, width=640)
```

Finding and sorting contours:

```python
import cv2
import improcv as im

mask = im.threshold(im.ensure_gray(image), method="otsu")
contours, hierarchy = im.find_contours(mask, retrieval_mode="external")
contours, boxes = im.sort_contours(contours, order="left-to-right")

# `Contour` keeps OpenCV's own (N, 1, 2) int32 shape, so results pass
# straight into any cv2.* function that expects a contour — no conversion.
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

Connected components and flood fill:

```python
import improcv as im

mask = im.threshold(im.ensure_gray(image), method="otsu")
num_labels, labels, stats, centroids = im.connected_components_with_stats(mask)
# stats[0]/centroids[0] describe the background label (0); inspect
# stats[label, 4] (area) before trusting a component's other statistics.

result = im.flood_fill(image, seed_point=(10, 10), new_value=(0, 255, 0))
print(result.filled_count, result.bounding_box)
```

Histogram and template matching:

```python
import improcv as im

gray = im.ensure_gray(image)
hist = im.histogram(gray)  # channel=0, bins=256, value_range=(0.0, 256.0) by default

result = im.match_template(gray, template, method="ccoeff_normed")
match = im.min_max_loc(result)
print(match.max_loc)  # (x, y) of the best match
```

Segmentation and inpainting:

```python
import improcv as im

markers = im.watershed(image, seed_markers)
# Positive values = regions, -1 = boundaries; 0 may remain unassigned.

foreground_mask = im.grabcut_rect(image, im.BoundingBox(x=20, y=20, width=200, height=150))

restored = im.inpaint(image, damage_mask, radius=3.0, method="telea")
```

Feature detection and description:

```python
import cv2
import improcv as im

features = im.detect_and_compute(im.ensure_gray(image), method="orb")
print(len(features.keypoints), features.descriptors.shape, features.norm)

# features.keypoints are real cv2.KeyPoint objects -- pass them straight
# into any cv2.* function that expects them, no conversion needed.
annotated = cv2.drawKeypoints(image, features.keypoints, None)
```

Matching features between two images:

```python
import cv2
import improcv as im

query = im.detect_and_compute(im.ensure_gray(image1), method="orb")
train = im.detect_and_compute(im.ensure_gray(image2), method="orb")

matches = im.match_features(query, train)
# matches is a plain list[cv2.DMatch], sorted by distance (best match
# first) -- pass it straight into cv2.drawMatches, no conversion needed.
annotated = cv2.drawMatches(image1, query.keypoints, image2, train.keypoints, matches, None)

# Or filter with Lowe's ratio test instead of match_features' cross-check:
ratio_matches = im.match_features_ratio(query, train, ratio=0.75)
```

## Status

`improcv` is in early development (pre-`1.0.0`); the public API may still change between minor
releases. See [CHANGELOG.md](https://github.com/michalmaj/improcv/blob/main/CHANGELOG.md) for
what's been added so far, and [ROADMAP.md](https://github.com/michalmaj/improcv/blob/main/ROADMAP.md)
for the planned phases toward `1.0.0`.

## License

MIT — see [LICENSE](https://github.com/michalmaj/improcv/blob/main/LICENSE).
