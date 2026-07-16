# improcv

Modern image-processing and computer-vision utilities for Python, NumPy and OpenCV.

`improcv` provides small, well-typed, well-tested helpers for common OpenCV tasks, supporting
both OpenCV 4.x and OpenCV 5.x.

## Installation

```bash
pip install improcv
```

`improcv` does not install an OpenCV distribution for you. Install whichever variant fits your
project — `opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, or
`opencv-contrib-python-headless` — separately:

```bash
pip install opencv-python
```

## Usage

```python
import cv2
import improcv as im

image = cv2.imread("photo.jpg")
resized = im.resize(image, width=640)
```

## Status

`improcv` is in early development (pre-`1.0.0`); the public API may still change between minor
releases. See [IMPROCV_PROJECT_BRIEF.md](IMPROCV_PROJECT_BRIEF.md) for the project's design goals
and roadmap.

## License

MIT — see [LICENSE](LICENSE).
