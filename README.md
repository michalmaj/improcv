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

`improcv.visualization` (matplotlib-based display helpers) needs the separate `viz` extra, on top
of one of the OpenCV extras above:

```bash
pip install "improcv[cv-headless,viz]"
```

`import improcv` never imports matplotlib — only `import improcv.visualization` does, and it
raises a clear error if the `viz` extra isn't installed.

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

# Estimate a RANSAC homography from the matches:
result = im.find_homography(query, train, ratio_matches)
if result.homography is not None:
    print(result.homography, result.inlier_mask.sum(), "inliers")
```

Hough transform shape detection:

```python
import improcv as im

edges = im.auto_canny(im.ensure_gray(image))

lines = im.hough_lines(edges, threshold=100)
segments = im.hough_line_segments(edges, threshold=50, min_line_length=30, max_line_gap=10)

# hough_circles takes a grayscale image directly, not a binary edge mask.
circles = im.hough_circles(im.ensure_gray(image), min_dist=20, param2=30)
for circle in circles:
    print(circle.x, circle.y, circle.radius)
```

QR code decoding:

```python
import improcv as im

code = im.decode_qr_code(image)
if code is not None:
    print(code.data, code.points)  # data is None if detected but undecodable

# For images that may contain more than one QR code:
for code in im.decode_qr_codes(image):
    print(code.data, code.points)
```

Barcode decoding (EAN-8, EAN-13, UPC-A):

```python
import improcv as im

for barcode in im.decode_barcodes(image):
    print(barcode.data, barcode.barcode_type, barcode.points)
    # data/barcode_type are both None if detected but undecodable
```

Annotation drawing:

```python
import improcv as im

mask = im.threshold(im.ensure_gray(image), method="otsu")
contours, _ = im.find_contours(mask)
boxes = im.bounding_boxes(contours)

annotated = im.draw_contours(image, contours, color=(0, 255, 0), thickness=2)
annotated = im.draw_bounding_boxes(annotated, boxes, color=(255, 0, 0), thickness=2)
# Both return a new array; `image` itself is never modified.

# Tiling several images into one grid:
grid = im.montage([image, annotated], tile_width=200, tile_height=200)
```

Point/region detectors:

```python
import cv2
import improcv as im

gray = im.ensure_gray(image)

fast_keypoints = im.detect_fast_keypoints(gray)
blob_keypoints = im.detect_blob_keypoints(gray)
annotated = cv2.drawKeypoints(image, fast_keypoints, None)
annotated = cv2.drawKeypoints(annotated, blob_keypoints, None)

mser_regions = im.detect_mser_regions(gray)
# region.points is every pixel belonging to the region as an unordered
# set -- not an ordered boundary, so don't pass it to draw_contours.
# Use the region's bounding box instead:
annotated = im.draw_bounding_boxes(annotated, [region.bounding_box for region in mser_regions])
```

Visualization (optional, requires `pip install "improcv[viz]"`):

```python
import improcv.visualization as viz

viz.show_image(image, title="input")  # handles BGR->RGB, hides axes by default
viz.plot_histogram(image)              # one line per channel (B/G/R or grayscale)
```

Image quality metrics:

```python
import improcv as im

error = im.mse(original, compressed)
quality_db = im.psnr(original, compressed)      # math.inf if the images are identical
similarity = im.ssim(original, compressed)      # 1.0 for identical images, not clamped otherwise

# float images need an explicit data_range; uint8 and uint16 infer it automatically:
similarity = im.ssim(original_f32, compressed_f32, data_range=1.0)

# gmsd is grayscale-only (2D, or 3D with exactly 1 channel) -- convert first
# with im.ensure_gray. Unlike ssim/psnr, lower is better: 0.0 for identical
# images; higher values generally indicate greater distortion according to
# GMSD.
distortion = im.gmsd(im.ensure_gray(original), im.ensure_gray(compressed))
```

Perceptual hashing:

```python
import improcv as im

# uint8 only: grayscale (H, W)/(H, W, 1), BGR (H, W, 3), or BGRA (H, W, 4).
# For color input, alpha (if present) is ignored, not composited.
a = im.average_hash(original)
b = im.phash(compressed)              # a different algorithm -- see below
c = im.phash(compressed_but_blurred)

distance = c.distance(im.phash(compressed))    # Hamming distance, an int -- lower means more similar
print(c)                                       # fixed-width, lowercase hex

# a.distance(b) would raise ValueError: average_hash and phash are different,
# non-comparable algorithms even though both produce 64-bit hashes by default.

# round-tripping through hex requires the algorithm and hash_size explicitly --
# a hex string alone can't reveal which algorithm produced it:
restored = im.PerceptualHash.from_hex(str(c), algorithm=im.PerceptualHashAlgorithm.PHASH)
assert restored == c
```

A perceptual hash is not a cryptographic one: collisions are expected, and a
smaller Hamming distance usually (not always) means more visually similar
images. `average_hash`/`phash` reproduce `cv2.img_hash`'s own bit decisions
for `hash_size=8` -- but not its packed-byte serialization, and not other
libraries' (e.g. `ImageHash`) genuinely different, incompatible variants of
the same algorithm names.

Photo/creative single-image effects:

```python
import improcv as im

# uint8 only, exactly 3-channel BGR -- no automatic grayscale/BGRA handling.
bgr = im.ensure_bgr(gray)          # convert grayscale first if needed
sketch = im.pencil_sketch(bgr)     # sketch.grayscale: (H, W); sketch.color: (H, W, 3)
stylized = im.stylize(bgr)
enhanced = im.detail_enhance(bgr)

# BGRA must be handled explicitly before calling -- ensure_bgr does not
# accept it, since there's no single correct way to turn alpha into BGR:
bgr_from_bgra = bgra[..., :3]                    # drop alpha, or
# bgr_from_bgra = your_own_alpha_compositing(bgra)  # composite onto a background
```

`sigma_s`/`sigma_r` (and `pencil_sketch`'s `shade_factor`) are restricted to the ranges OpenCV's
own API documents (`0 < sigma_s <= 200`, `0 < sigma_r <= 1`, `0 <= shade_factor <= 0.1`) --
verified directly that `sigma_r=0` leads to division by zero internally, and that values beyond
these ranges are unsupported by OpenCV's own contract (the parameters are stored as a C++ `float`,
so extreme values can silently degrade to a useless result).

Seamless cloning (Poisson image editing):

```python
import improcv as im

result = im.seamless_clone(
    source,
    destination,
    mask,
    center=(x, y),
    mode="normal",
)
```

`source`/`destination` are `uint8` BGR `(H, W, 3)` -- no automatic conversion, and they don't need
to be the same size. `mask` is a `uint8` `(H, W)` array matching `source`'s spatial size, with only
`0`/`255` accepted. OpenCV always ignores `mask`'s outermost 1-pixel border, and the bounding box of
what remains (after that border is zeroed) must be at least `3x3` pixels. `center=(x, y)` is in
`destination`'s coordinate system and is where that bounding box's center is placed -- not the
center of all of `source` or all of `mask`. There is no automatic alpha handling. Seamless cloning
reconstructs the pasted region from *gradients*, not by copying pixels -- so a flat-colored `source`
region can produce little or no visible change, unlike the "cut and paste" effect of alpha blending.
`"mixed"` picks whichever of `source`'s/`destination`'s gradient is stronger at each pixel (useful
for loosely-drawn masks); `"monochrome_transfer"` transfers `source`'s luminance structure rather
than its color. The result always has `destination`'s shape and dtype.

HDR-related operations (`improcv.hdr`) are split into three distinct techniques, not one "HDR"
feature: **exposure fusion** blends a stack of differently-exposed images directly, without
reconstructing any physical light measurement; **radiance HDR merge** reconstructs an actual HDR
radiance map from a stack plus its exposure times; a future **tone mapping** will compress that
radiance map's dynamic range back down for display. Exposure fusion and radiance merge are
implemented; tone mapping is not yet.

Exposure fusion:

```python
import numpy as np
import improcv as im

fused = im.fuse_exposures(images)  # images: list/tuple of uint8, same shape, at least 2

# fused is float32, nominally close to [0, 1] but not clipped to it --
# convert explicitly before saving/displaying as uint8:
display = np.clip(fused, 0.0, 1.0)
display_u8 = np.round(display * 255.0).astype(np.uint8)
```

`fuse_exposures` wraps OpenCV's Mertens exposure fusion: it blends the stack directly (in the domain
of a Laplacian pyramid, weighted by local contrast/saturation/well-exposedness), producing a single
well-exposed image -- it does **not** need exposure times and does **not** produce a physical HDR
radiance map, so its result does not need (and should not go through) tone mapping. `images` must be
a real `Sequence` (list/tuple, or another `collections.abc.Sequence`) of at least 2 `uint8` images,
either all 2D grayscale or all 3D BGR `(H, W, 3)` with identical shape -- a single stacked array,
`(H, W, 1)`, 2-channel, and BGRA are all rejected, with no automatic conversion. `contrast_weight`/
`saturation_weight`/`exposure_weight` must be non-negative and finite; `0` is legal for all three
(it's `exposure_weight`'s own default). Repeated calls with identical input are not guaranteed to be
bit-for-bit identical -- OpenCV's implementation uses internal parallel summation.

Radiance HDR merge -- without calibration (OpenCV's fixed linear response):

```python
import improcv as im

hdr = im.merge_hdr_debevec(images, exposure_times)  # or im.merge_hdr_robertson(...)
```

With a calibrated response curve:

```python
response = im.calibrate_camera_response_debevec(images, exposure_times)
hdr = im.merge_hdr_debevec(images, exposure_times, response_curve=response)

# or the Robertson equivalent:
response = im.calibrate_camera_response_robertson(images, exposure_times)
hdr = im.merge_hdr_robertson(images, exposure_times, response_curve=response)
```

`images[i]` and `exposure_times[i]` are paired by index -- neither is ever reordered. All exposure
times must use one consistent unit (conventionally seconds): uniformly rescaling every time by a
constant factor rescales the entire output radiance map by the reciprocal of that factor. Not passing
`response_curve` does **not** calibrate anything -- calibration is always an explicit, separate step
you call yourself; OpenCV uses a fixed linear response instead if you don't. The output is a raw
radiance map (`float32`, typically ranging far beyond `[0, 1]`, not clipped or normalized) -- it is
**not** display-ready and needs tone mapping (see below) before it can be saved or shown; do not
write it directly as a `uint8` image. **Both merge functions accept BGR only -- grayscale is
not supported by either.** `merge_hdr_robertson` raises a raw `cv2.error` for grayscale regardless of
dtype, verified directly. `merge_hdr_debevec`'s own default (no explicit `response_curve`)
linear-response construction has a confirmed bug in OpenCV's own C++ source that corrupts memory for
a genuinely 1-channel array -- undefined behavior that happens not to crash on some platforms but
crashed the process outright (a non-catchable native abort) in this project's own CI on another, so
grayscale is rejected unconditionally for both merge functions rather than only in the specific
triggering case. `uint8`, `uint16`, and `float32` are all accepted for BGR merge (`float32` values
must be finite and within `[0, 1]`). `uint16`/`float32` additionally require an OpenCV build that
supports them for HDR merge -- verified directly that OpenCV `4.9.0` (this project's documented
minimum) only supports `uint8` here; a clear `ValueError` is raised instead of a raw OpenCV error on
an older build. **Calibration itself is always `uint8`-only** (for both algorithms), so its output is
always a 256-entry curve -- it pairs directly with a `uint8` merge, not automatically with a
`uint16`/`float32` one. `calibrate_camera_response_debevec` accepts grayscale *or* BGR;
`calibrate_camera_response_robertson` is **BGR only** (raises a clear error pointing at the Debevec
calibrator for a grayscale stack). `calibrate_camera_response_debevec`'s `random_sampling=True`
samples pixel locations randomly with no seed control in OpenCV's own API -- its result is not
guaranteed reproducible across calls. `calibrate_camera_response_robertson` needs a reasonably
diverse intensity histogram to produce a finite curve at all: verified directly that an all-black or
all-white image stack (or one with very few distinct intensity values) deterministically raises
`RuntimeError` here, regardless of image size. `calibrate_camera_response_debevec` is generally more
robust to sparse or degenerate intensity histograms because of its smoothness regularization, but a
finite result is not guaranteed on every supported OpenCV build. Non-finite or otherwise
merge-incompatible curves raise `RuntimeError`. Neither calibrator performs exposure alignment or
ghost removal -- the input stack is assumed already aligned, for both calibration and merge.

Tone mapping -- compressing a radiance map down to a display-ready image:

```python
import numpy as np
import improcv as im

response = im.calibrate_camera_response_debevec(images, exposure_times)
hdr = im.merge_hdr_debevec(
    images,
    exposure_times,
    response_curve=response,
)
tone_mapped = im.tone_map_reinhard(hdr)

display_u8 = np.round(
    np.clip(tone_mapped, 0.0, 1.0) * 255.0
).astype(np.uint8)
```

A radiance map from `merge_hdr_debevec`/`merge_hdr_robertson` is not display-ready -- its values
typically extend far beyond `[0, 1]` and are not clipped or normalized. Tone mapping compresses that
dynamic range back down to something a display or `uint8` file can represent. Four operators are
provided, wrapping OpenCV's own `cv2.createTonemap`/`createTonemapDrago`/`createTonemapReinhard`/
`createTonemapMantiuk`: `tone_map` (simple linear normalization with gamma correction), `tone_map_drago`
(adaptive logarithmic compression), `tone_map_reinhard` (photographic, local/global adaptation blend),
and `tone_map_mantiuk` (contrast-domain compression via an iterative solver -- markedly more
expensive than the other three). Each has its own parameters and produces a visibly different look;
none is a drop-in replacement for another. All four return raw `float32` -- **none of them clip,
normalize, or quantize their output**, even though OpenCV's own documentation describes the result as
`[0, 1]`: verified directly that a spatially constant input (e.g. a flat, saturated region) can
produce output well outside that range. Always clip explicitly (`np.clip(..., 0.0, 1.0)`) before
converting to `uint8`, as in the example above. `hdr` must be `float32`, shape `(H, W, 3)` (BGR --
tone mapping never converts to RGB), finite, and non-empty for all four functions; negative values are
allowed, since neither merge function guarantees non-negative radiance. `tone_map_reinhard` and
`tone_map_mantiuk` additionally reject a spatially constant `hdr` (`ValueError`) -- verified directly,
in OpenCV's own C++ source, that both divide by a quantity that is exactly zero for a constant image,
regardless of parameters. `tone_map_drago` and `tone_map_mantiuk` additionally reject an `hdr` that
would produce a true zero-luminance (black) pixel once run through OpenCV's own internal
normalization -- a common case, not just a synthetic one (any `hdr` whose darkest point across all
three channels is a true black pixel triggers it). `tone_map_mantiuk` additionally requires both
`hdr` dimensions to be at least `2`. See each function's docstring for the full parameter contract
and exact value ranges.

Mertens exposure fusion (`fuse_exposures`, above) is a different operation and does **not** produce a
radiance map -- do not run its output through any of the tone-mapping functions; its result is already
close to display-ready (see its own section above).

Non-local means denoising:

```python
import improcv as im

denoised = im.nl_means_denoise(gray)                # uint8 2D grayscale
denoised_bgr = im.nl_means_denoise_colored(bgr)      # uint8 (H, W, 3) BGR
```

Both are `uint8`-only with no automatic conversion (grayscale/BGR/BGRA/2-channel input to the wrong
function is rejected, not silently handled). Higher `h` (and `h_color` for the colored version)
generally smooths more aggressively but can also remove real detail. `nl_means_denoise_colored`
works internally in CIELAB (denoising luminance and color separately, like OpenCV's own
implementation) -- unlike the grayscale version, `h_luminance=0, h_color=0` does **not** guarantee a
result identical to the input, since the BGR/CIELAB round-trip alone can shift values slightly.
Larger `search_window_size` values can substantially increase execution time; `7`/`21`
(`template_window_size`/`search_window_size`) are OpenCV's own recommended defaults, not hard limits.
`template_window_size` and `search_window_size` are independent parameters -- there is no
requirement that `search_window_size` be at least `template_window_size`.

## Status

`improcv` is in early development. `0.1.0a1` is designated as the first public release and covers
the accumulated scope of Phases 0-3 (see [ROADMAP.md](https://github.com/michalmaj/improcv/blob/main/ROADMAP.md)
for what that includes, and why it doesn't match the project's original one-phase-per-minor-version
plan). See [CHANGELOG.md](https://github.com/michalmaj/improcv/blob/main/CHANGELOG.md) for the exact
list of what's been added.

**Compatibility policy before `1.0.0`:**
- `0.1.0a1` is an alpha: functional, tested, but not yet declared stable.
- Before `1.0.0`, the public API may still change, including in backwards-incompatible ways, in any
  `0.MINOR` release. While in alpha, this also applies between consecutive prereleases of the same
  version (e.g. `0.1.0a1` → `0.1.0a2`).
- Any backwards-incompatible change will always be called out explicitly in `CHANGELOG.md`, not
  silently folded into a routine entry.
- Deprecation (a warning period before removal) will be used where practical, but the project does
  not yet guarantee a fixed deprecation window before `1.0.0`.
- A full, stable compatibility and deprecation policy will be established before `1.0.0` ships.

## License

MIT — see [LICENSE](https://github.com/michalmaj/improcv/blob/main/LICENSE).
