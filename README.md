# improcv

Modern image-processing and computer-vision utilities for Python, NumPy and OpenCV.

`improcv` provides small, well-typed, well-tested helpers for common OpenCV tasks, supporting
both OpenCV 4.x and OpenCV 5.x.

## Installation

`pip install improcv` alone installs NumPy but **not** OpenCV — `import improcv` will fail with a
clear error telling you to install one. Pick exactly one variant:

```bash
pip install improcv[cv]                  # opencv-python
pip install improcv[cv-headless]         # opencv-python-headless
pip install improcv[cv-contrib]          # opencv-contrib-python
pip install improcv[cv-contrib-headless] # opencv-contrib-python-headless
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

## Status

`improcv` is in early development (pre-`1.0.0`); the public API may still change between minor
releases. See [CHANGELOG.md](https://github.com/michalmaj/improcv/blob/main/CHANGELOG.md) for
what's been added so far.

## License

MIT — see [LICENSE](https://github.com/michalmaj/improcv/blob/main/LICENSE).
