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

HDR-related operations (`improcv.hdr`) are split into distinct techniques, not one "HDR" feature:
**Mertens exposure fusion** directly combines aligned LDR images into a display-oriented `float32`
result without reconstructing physical radiance; **radiance HDR merge** reconstructs an actual HDR
radiance map from a stack plus its exposure times; **camera-response calibration** estimates the
response curve that merge can use instead of its default fixed linear one; **tone mapping** then
compresses that radiance map's dynamic range back down for display. All four are implemented below.

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

**A finite result is not unconditionally guaranteed even for well-formed, non-degenerate `hdr`.**
All four operators internally raise a normalized value to some exponent (`gamma`, `tone_map_drago`'s
`bias`, or an internal, non-parametrized exponent for `tone_map_reinhard`/`tone_map_mantiuk`);
OpenCV's own floating-point rounding can occasionally leave that value very slightly negative -- an
ordinary artifact of min/max normalization, not a data problem -- and raising a negative number to a
non-integer power is mathematically undefined, producing `NaN`. Whether this actually happens is
CPU-architecture/SIMD-dispatch-dependent, not just data-dependent: verified directly that the same
seed and parameters that tone-map finitely on Apple Silicon produced a non-finite result (cleanly
caught by `RuntimeError`) on x86_64 CI. Treat `RuntimeError` from any tone-mapping function as a real,
expected possibility for otherwise-ordinary input, not only for the degenerate cases listed above.

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

Panorama and scan stitching:

```python
import improcv as im

panorama = im.stitch_images(
    images,
    mode="panorama",
)

# flat scans / documents captured under a simpler planar transformation:
scan = im.stitch_images(images, mode="scans")
```

`stitch_images` wraps OpenCV's high-level `cv2.Stitcher`. `images` must be a real `Sequence` (list,
tuple, or another `collections.abc.Sequence`) of at least 2 `uint8` BGR `(H, W, 3)` images -- a single
stacked array (including 4D), `str`/`bytes`/`bytearray`, and a generator/iterator are all rejected
explicitly, as are grayscale, `(H, W, 1)`, 2-channel, BGRA, `uint16`, `float32`, and `float64` images,
with no automatic conversion. Images may have different spatial shapes -- there is no requirement that
they match. `mode="panorama"` (the default) uses a homography/perspective model, suited to photos
taken by rotating a camera; `mode="scans"` uses an affine model, suited to flat scans or documents --
**`"scans"` is not limited to pure translational sliding**, it is a broader affine model, just not
projective/perspective. Both modes share the identical input/output contract.

On success, the result is a new, independent `uint8` BGR array -- not cropped, and not guaranteed any
particular shape, aspect ratio, or relationship to the input sizes. Insufficient overlap, too few
usable features, or an algorithmic failure of homography/camera-parameter estimation all raise
`RuntimeError` (never `ValueError` -- the input images are structurally fine, the algorithm simply
could not relate them), naming the specific OpenCV status and its numeric code.

**Neither the outcome nor the exact result is guaranteed deterministic, even within a single process.**
OpenCV's feature matching and geometry estimation use RANSAC internally, drawing from OpenCV's global
RNG -- verified directly that, for a borderline amount of overlap, the same input images can succeed
on one call and fail with `RuntimeError` on the next, in the same process, and that even a reliably
successful stitch is not guaranteed to produce the same output shape or pixel values across repeated
calls. `stitch_images` never calls `cv2.setRNGSeed` itself -- that would silently change OpenCV's
global RNG state for every other OpenCV call in the process, not just this one. You can call
`cv2.setRNGSeed` yourself before calling `stitch_images` if reproducibility matters for your own
experiments, but that is a process-global setting, not a per-call one, and it does not promise
identical results across different OpenCV builds or platforms.

**Stitching can be expensive in both time and memory, in a way this wrapper cannot safely bound.**
OpenCV allocates the output panorama and its internal intermediate buffers *inside* `stitch()`, before
control returns to Python -- verified directly that a poorly-conditioned geometry estimate (e.g. two
images whose relative orientation does not match what `mode` expects) can make OpenCV allocate, and
report as a *successful* result, a panorama and intermediate buffers many times larger than the
inputs. Because the large allocation already happens before this wrapper ever sees a return value, no
check on the returned array could have prevented it, so `stitch_images` does not attempt one. For
untrusted or unpredictable input sets, consider running this function in an isolated process with its
own resource limits, or use OpenCV's lower-level stitching pipeline directly for finer control.

Per-image feature masks supported by the lower-level OpenCV Stitcher API are not exposed by this
wrapper. This first version also does not expose any of `cv2.Stitcher`'s registration/seam/
compositing/confidence settings -- it always uses OpenCV's own defaults.

DNN preprocessing:

```python
import improcv as im

blob = im.create_dnn_blob(
    image,
    size=(224, 224),
    scale=1.0 / 255.0,
    mean=(0.0, 0.0, 0.0),
    swap_rb=True,
)

# a batch of images sharing one set of parameters:
batch = im.create_dnn_batch_blob(
    [image_a, image_b],
    size=(224, 224),
    scale=1.0 / 255.0,
    swap_rb=True,
)
```

`create_dnn_blob`/`create_dnn_batch_blob` wrap `cv2.dnn.blobFromImage`/`blobFromImages` to turn a
`uint8`/`float32` image (or a `Sequence` of them) into a blob ready for `cv2.dnn.Net.setInput` --
they do not load a model, create a `cv2.dnn.Net`, or run inference; that is a separate, later slice.
The output is always `float32` and always 4-D NCHW (`(1, C, H, W)` for a single image, `(N, C, H, W)`
for a batch) -- a single image still gets an explicit batch dimension, there is no 3-D HWC return
path. `size` is `(width, height)`, matching OpenCV's own convention, not NumPy's `(height, width)`
indexing order. `crop=False` (the default) resizes directly to `size`, stretching independently in
each dimension without preserving aspect ratio; `crop=True` uniformly scales the image so it covers
`size` in both dimensions, then takes a centered crop -- both always use OpenCV's `INTER_LINEAR`,
which is not configurable, because the underlying OpenCV function does not expose an `interpolation`
parameter either.

The operation is `(input - mean) * scale`. A scalar `mean` (e.g. `mean=1.0`) is broadcast by improcv
to every channel; passing that same scalar directly to raw `cv2.dnn.blobFromImage` would **not**
broadcast it -- it would be interpreted as only the first channel's mean, leaving the others at zero.
A tuple `mean` must have exactly one element per channel, and its order refers to the *output*
channel order: for BGR/BGRA input, `swap_rb=False` means `(B, G, R[, A])` and `swap_rb=True` means
`(R, G, B[, A])` -- passing BGR-ordered image data does not, by itself, produce RGB-ordered output;
`swap_rb=True` is required for that, and it never touches an alpha channel, which always stays last.

`create_dnn_batch_blob` requires an explicit `size` -- there is no "keep native size" default for a
batch, because OpenCV silently resizes every image after the first to match the *first* image's
native size when no `size` is given, which silently produces wrong results for a batch of
differently-sized images. `create_dnn_blob` (single image) does default `size` to `None`, which
keeps that one image's native size.

Only `uint8` and `float32` input is accepted (grayscale, `(H, W, 1)`, BGR, or BGRA) -- verified
directly that other dtypes accepted by raw OpenCV on some versions (e.g. `int16`, `uint16`,
`float64`) are silently converted on OpenCV >= 4.13 but raise a raw `cv2.error` on OpenCV 4.9, so
allowing them here would make behavior depend on the installed OpenCV version. There is no
`output_dtype` parameter in this first version -- the output is always `float32`; a `uint8`-output
mode may be added later, compatibly, as a new keyword-only parameter if a concrete need for it
appears.

**Extremely large `size` values can exhaust process memory or be killed by the operating system --
this is not, and cannot be, turned into a catchable Python exception.** `size` is validated to be
representable (each dimension and their product must fit a signed 32-bit int, matching an internal
limit in OpenCV's own blob-construction code), but that says nothing about whether the resulting
allocation is a *reasonable* size for the machine running it -- pick `size` based on your actual model
input requirements, not arbitrarily large values.

No DNN inference or `cv2.dnn.Net` configuration (backend/target) is included in this slice, and no
new dependency (or `improcv[ml]` extra) was introduced for it -- both functions run on the same base
OpenCV install as the rest of `improcv`. ONNX model loading is implemented below as a separate
Phase 5 slice.

ONNX model loading:

```python
import improcv as im

net = im.load_onnx_network("model.onnx")

blob = im.create_dnn_blob(
    image,
    size=(224, 224),
    scale=1.0 / 255.0,
    swap_rb=True,
)

net.setInput(blob)
output = net.forward()
```

`load_onnx_network`/`load_onnx_network_from_bytes` load an ONNX file (from a path or from an
in-memory `bytes` buffer, respectively) into a `cv2.dnn.Net` -- they are **ONNX-only**, not a
generic "load any DNN model" function; other formats (TensorFlow, Caffe, Darknet, Torch, OpenVINO)
are out of scope. Preprocessing (`create_dnn_blob`/`create_dnn_batch_blob`, above) and model loading
are both `improcv` API; `Net.setInput`/`Net.forward` (shown above) remain raw OpenCV API, not wrapped
-- this slice does not add an inference function, model-specific postprocessing, or a backend/target
API.

The path and bytes loaders are two separate functions, not one function accepting either, because a
bare `bytes` argument is genuinely ambiguous to raw OpenCV: verified directly that
`cv2.dnn.readNetFromONNX(some_bytes)` (positional) is silently routed to the *path* overload on
OpenCV 4.13/5.0 (interpreting the bytes as a file path and failing to find it) but to the *buffer*
overload on OpenCV 4.9 -- an actual behavioral difference between supported versions, not a
hypothetical one. `load_onnx_network_from_bytes` only accepts `bytes` (not `bytearray`, `memoryview`,
or an `ndarray`) and internally converts to a `uint8` array before calling OpenCV by keyword, which
resolves correctly and identically on every supported version.

`load_onnx_network`'s `path` accepts a `str` or `os.PathLike[str]` (e.g. `pathlib.Path`); its content,
not its extension, is what's parsed -- a valid ONNX file with no `.onnx` extension is accepted.
Filesystem problems that Python can detect directly give native exceptions (`FileNotFoundError`,
`IsADirectoryError`, an empty file raises `ValueError`); a problem OpenCV's own parser hits (a
corrupt/invalid model, or a permission/ACL issue that only surfaces once OpenCV tries to open the
file) raises `RuntimeError` with the original error attached as `__cause__`. A path containing
non-ASCII characters is not guaranteed to work identically on every platform -- verified directly
(via CI) that the same accented path opens fine on Linux/macOS but makes OpenCV's own file-opening
code fail on Windows (surfacing as `RuntimeError`, not a bug in this wrapper's validation); prefer
an ASCII-only path where cross-platform behavior matters.

Every call to either loader parses the model again and returns a new, independent `cv2.dnn.Net` --
**nothing is cached**, and repeated loading of a large model is exactly as expensive as it looks. The
returned `Net` is a stateful object: calling `setInput()` mutates it, backend/target configuration is
your responsibility, and this wrapper makes no thread-safety promise about it.

On OpenCV 5, `improcv` requests `ENGINE_CLASSIC` as the common behavior shared with OpenCV 4.x, since
4.x has no other engine. OpenCV process configuration, including `OPENCV_FORCE_DNN_ENGINE`, may
override that request -- this is a best-effort request, not a guarantee.

**ONNX models are parsed by OpenCV's native (C++) code, with no sandboxing from this wrapper.** There
is no file-size limit, and no attempt is made to validate the ONNX format beyond what OpenCV's own
parser does. A malicious or merely corrupted model can exploit a parser bug, and a very large model
can consume a large amount of memory or end the process outright -- the same risk as parsing any
untrusted binary format. Do not load models from untrusted sources in-process; isolate them in a
separate process with its own resource limits instead. This slice does not download models from
anywhere -- you provide the path or bytes.

Classification evaluation:

```python
import improcv as im

cm = im.confusion_matrix(
    y_true=[0, 0, 1, 1],
    y_pred=[0, 1, 1, 1],
    labels=[0, 1],
)

metrics = im.classification_metrics(
    y_true=[0, 0, 1, 1],
    y_pred=[0, 1, 1, 1],
    labels=[0, 1],
    average=None,
)
```

`confusion_matrix`/`classification_metrics` cover single-label multiclass classification with
integer labels only -- one true class and one predicted class per sample, not multilabel, and not
string/hashable labels. `cm.matrix[i, j]` counts samples whose true class is `cm.labels[i]` and
predicted class is `cm.labels[j]`: **rows are true labels, columns are predicted labels**.
`labels=None` infers the class universe as the sorted union of every value observed in `y_true`/
`y_pred`; an explicit `labels` fixes the exact row/column order instead. `improcv` raises
`ValueError` in two cases where `sklearn.metrics.confusion_matrix` does not: a duplicate value
within an explicit `labels` (`sklearn` silently accepts it, but with confusing index semantics --
the repeated label's row/column gets written to by whichever occurrence's index NumPy resolves
last, not merged or rejected), and an observed `y_true`/`y_pred` value outside an explicit `labels`
(`sklearn` silently drops that sample from the matrix instead of erroring).

`classification_metrics`'s result type depends on `average`: `average=None` (the default) gives
`precision`/`recall`/`f1` as per-class, read-only `float64` arrays; `average="micro"`/`"macro"`/
`"weighted"` gives them as plain Python `float`s instead -- never both forms from the same call.
`support` is always a per-class, read-only `int64` array regardless of `average`, and `accuracy`
is always a plain `float`. `zero_division` (`0.0`, `1.0`, or `"nan"`) controls what a class reports
when its own division is genuinely undefined -- **precision**, **recall**, and **F1** each have
their *own* zero-division condition, checked independently from true positive/false positive/false
negative counts (`TP`/`FP`/`FN`), never from each other: precision uses `zero_division` only when
`TP + FP == 0` (the class was never predicted); recall uses it only when `TP + FN == 0` (the class
never occurs in the true labels); F1 uses it only when `2*TP + FP + FN == 0` (the class is
completely absent from both). A class with `TP = 0` but real, nonzero `FP`/`FN` has a correctly
defined `F1 = 0` -- `zero_division` does not turn that `0` into `1.0` or `NaN`. With `"nan"`, an
actually-undefined value's `NaN` propagates into `"macro"`/`"weighted"` aggregates by plain
averaging, not by skipping the affected class, including when that class's own support (and
therefore its weight) is zero.

A confusion matrix already computed (including one you've aggregated yourself from several
batches, e.g. `ConfusionMatrixResult(matrix=sum(batch_matrices), labels=original_labels)`) can be
passed to `classification_metrics_from_confusion_matrix` directly, without recomputing it from raw
labels -- since such a matrix is never re-derived from `y_true`/`y_pred`, its total count is
verified to fit in `int64` before `support`/precision/recall/F1 are computed, raising `ValueError`
instead of silently wrapping around to a negative count if it doesn't. An empty confusion matrix
is possible only with explicit `labels` (`confusion_matrix([], [], labels=[0, 1])` returns a
well-defined all-zero matrix); `classification_metrics`, `classification_metrics_from_confusion_matrix`,
and `confusion_matrix` with `labels=None` all require at least one observation. Building the dense
matrix costs `O(len(labels) ** 2)` memory, so a very large explicit `labels` can exhaust process
memory even though improcv checks the allocation is at least representable first.

This is a numeric core only: no plotting, no multilabel classification, no sample weights, and no
`scikit-learn` dependency.

Binary one-vs-rest ranking curves -- ROC, precision-recall, ROC AUC, and average precision:

```python
import improcv as im

y_true = [0, 0, 1, 1]
y_score = [0.1, 0.4, 0.35, 0.8]

roc = im.roc_curve(y_true, y_score, positive_label=1)
pr = im.precision_recall_curve(y_true, y_score, positive_label=1)
roc_auc = im.roc_auc_score(y_true, y_score, positive_label=1)
ap = im.average_precision_score(y_true, y_score, positive_label=1)
pr_auc = im.auc(pr.recall, pr.precision)  # trapezoidal PR-curve area -- distinct from `ap`
```

`y_score` is a ranking score, not a predicted label or a probability -- it does not need to lie in
`[0, 1]`, and a larger score means greater confidence in the positive class. `roc_curve`/
`precision_recall_curve`/`roc_auc_score`/`average_precision_score` are binary, one-vs-rest:
`positive_label` is always required and explicit -- a sample is positive iff its label equals
`positive_label`; every other observed label is negative, regardless of how many distinct negative
labels occur, and there is no automatic inference of which label is positive. A threshold
classifies a sample positive iff `score >= threshold`; every sample sharing the same score is
aggregated into one threshold before FPR/TPR/precision/recall is computed there, so permuting the
order of tied samples (or of the whole input) never changes the result.

`y_score` accepts a `Sequence` of Python `int`/`float` or NumPy `float16`/`float32`/`float64`
scalars, or a 1-D `ndarray` with an integer or `float16`/`float32`/`float64` dtype -- not "any
NumPy floating dtype": a NumPy floating scalar or `ndarray` wider than `float64` (e.g.
`np.longdouble` where it is a genuine extended-precision type on the current platform) is rejected
with `TypeError` either way, since narrowing it could silently collapse two distinct scores into
the same tied threshold. An integer score (Python or NumPy, in a `Sequence` or an `ndarray`) is
legal only when it converts to `float64` exactly -- both an out-of-range magnitude (e.g. `10**400`,
which would otherwise raise a raw `OverflowError`) and an in-range value that would lose precision
(e.g. `2**53 + 1`) raise the same documented `ValueError` instead. Every zero in the normalized
score array is canonicalized to positive zero, so a threshold derived from a tied `+0.0`/`-0.0`
group is always `+0.0` regardless of which sign happened to appear first in the input.

Both curves start at the same sentinel threshold `+inf` (predicting nothing positive): ROC starts
at `(FPR, TPR) = (0.0, 0.0)`, precision-recall starts at `(precision, recall) = (1.0, 0.0)` with no
corresponding real threshold. `thresholds[1:]` holds every distinct observed score in strictly
decreasing order, so `thresholds`/the two curve-value arrays always share one length (`K + 1` for
`K` distinct scores) -- `recall` increases alongside decreasing `thresholds`, matching `roc_curve`'s
convention (a deliberate departure from `sklearn.metrics.precision_recall_curve`'s descending
`recall`). `roc_curve`/`roc_auc_score` require at least one positive and one negative sample,
raising `ValueError` instead of scikit-learn's `UndefinedMetricWarning` for a degenerate input;
`precision_recall_curve` only requires at least one positive sample -- a `y_true` with no negative
sample is legal, giving `precision == 1.0` at every real threshold.

`roc_auc_score` integrates the ROC curve with the trapezoidal rule (equivalent to the probability
that a random positive sample outranks a random negative one, with a tied pair counted as
one-half) -- it never calls `np.trapz`/`np.trapezoid`, since neither name exists across this
project's full supported NumPy range: `np.trapezoid` was only introduced in NumPy 2.0, and
`np.trapz` was removed in NumPy 2.4, so no single name works across this project's full
`numpy>=1.24` support range. `roc_curve`/`precision_recall_curve` return new, independent,
read-only `float64` arrays -- never a view of `y_true`/`y_score`; `roc_auc_score`,
`average_precision_score`, and `auc` (below) return a plain Python `float`, not an array.

`average_precision_score` is binary classification ranking average precision -- **not**
object-detection AP or mAP (which additionally require matching predictions to ground truth by
IoU; this function has no notion of bounding boxes, IoU, or per-class averaging). It is a
non-interpolated weighted mean of precision, using each recall increment as its weight:
`sum((recall[i] - recall[i - 1]) * precision[i] for i in 1..K)` over the same grouped-threshold
points `precision_recall_curve` returns, with `precision[i]` always taken from the *right* end of
each recall increment. This is not linear interpolation and not the trapezoidal area under the PR
curve, which is a distinct quantity `average_precision_score` does not compute -- depending on the
shape of the curve and its ties, that trapezoidal area can be larger or smaller than average
precision, never consistently one or the other, so no fixed relationship between the two should be
assumed. A perfectly reversed ranking does not give `0.0` (average precision has no complement
relation the way ROC AUC does); a `y_true` with no negative sample gives exactly `1.0`; constant
scores (no discriminative power) give exactly the positive prevalence `P / len(y_true)` -- these
are the only two inputs with a closed-form result documented here, not a general property of every
weak or random ranking. `average_precision_score` shares `roc_curve`'s input and error contract,
except a `y_true` with no negative sample is accepted rather than rejected (same relaxation as
`precision_recall_curve`).

`auc(x, y)` is a general-purpose trapezoidal area-under-curve helper with no ranking semantics of
its own -- no `positive_label`, no tie-aggregation, no notion of positive/negative samples. `x`
must be non-decreasing or non-increasing throughout (duplicate `x` values are legal either way and
contribute a zero-width segment; constant `x` gives exactly `0.0`); a non-increasing `x` gives the
same positive geometric area a non-decreasing order of the same points would, not a signed integral
-- the result never depends on which direction a monotonic `x` happens to be given in. Unlike
`roc_auc_score`/`average_precision_score`, whose inputs and outputs are always in `[0, 1]` by
construction, `auc`'s `y` may be negative and so may its result. Ordinary calls use a fast,
canonical `float64` summation; if an intermediate segment width, height sum, or product would
overflow `float64`, `auc` falls back to computing the exact trapezoidal sum as a rational number
(`fractions.Fraction`, standard library, used only on this rare path) over the already-normalized
`float64` values, converting only the final total back to `float` -- this correctly preserves
cancellation between huge intermediate contributions and a subnormal residual that a naive
rescale-based fallback could otherwise underflow away to `0.0`. Only an input whose true, exact area
is not representable as a finite `float64` raises `ValueError` -- this never silently returns
`inf`/`-inf`/`NaN`, and never emits a NumPy warning for an accepted input. `auc` computes the
trapezoidal area under the precision-recall curve when called as `auc(curve.recall, curve.precision)`
-- this is a distinct quantity from `average_precision_score` (a non-interpolated, non-trapezoidal
definition, see above), and this is the complete, supported way to compute it: there is no separate
score-level function for it, to avoid one more symbol that could be confused with
`average_precision_score`.

`roc_curve`/`precision_recall_curve`/`roc_auc_score`/`average_precision_score` run in `O(N log N)`
time, `O(N)` memory, dominated by sorting scores by rank. `auc` needs no sorting: its ordinary,
`float64` fast path is `O(N)` time, `O(N)` memory. Its rare exact fallback -- triggered only when an
intermediate computation would overflow `float64` -- uses standard-library arbitrary-precision
rational arithmetic instead, which is not promised to run in strict `O(N)` time independent of how
large the underlying integer representations grow; this slice covers binary ROC/PR/ROC-AUC/
average-precision plus a generic trapezoidal `auc` helper: no multiclass score matrix, no averaging,
no sample weights, and no plotting.

Augmentation sampling and replay -- flip:

```python
import numpy as np

rng = np.random.default_rng(42)

flip_params = im.sample_flip(
    rng,
    horizontal_probability=0.5,
    vertical_probability=0.1,
)

flipped = im.apply_flip(image, flip_params)
```

With a segmentation mask, apply the same sampled flip to both:

```python
pair = im.apply_flip(image, flip_params, mask=mask)
flipped_image = pair.image
flipped_mask = pair.mask
```

Augmentation sampling and replay -- crop:

```python
crop_params = im.sample_crop(
    rng,
    source_size=(image.shape[1], image.shape[0]),
    crop_size=(256, 256),
)

pair = im.apply_crop(image, crop_params, mask=mask)
```

Augmentation sampling and replay -- affine (shear, rotation, translation, isotropic scale):

```python
affine_params = im.sample_affine(
    rng,
    source_size=(image.shape[1], image.shape[0]),
    angle_range=(-10.0, 10.0),
    translation_x_range=(-8.0, 8.0),
    translation_y_range=(-8.0, 8.0),
    scale_range=(0.9, 1.1),
    shear_x_range=(-0.15, 0.15),
    shear_y_range=(-0.10, 0.10),
)

pair = im.apply_affine(
    image,
    affine_params,
    mask=mask,
    mask_border_value=255,
)
```

`sample_flip`/`sample_crop`/`sample_affine` take an explicit `rng: np.random.Generator` and return
a small, independent result (`FlipParameters`/`CropParameters`/`AffineParameters`) -- there is no
global/implicit RNG anywhere in `improcv`. That result can be stored and replayed any number of
times through `apply_flip`/`apply_crop`/`apply_affine`, always producing the same output for the
same input. `CropParameters`/`AffineParameters` carry the `source_size` they were sampled for;
`apply_crop`/`apply_affine` require the image (and mask, if given) to match that size exactly, so
parameters sampled for one image can't be silently misapplied to a differently-sized one. Passing
`mask=` applies the identical transform to the mask, returning both as an `AugmentedImageMask`
instead of a bare array; a segmentation mask is restricted to `uint8`/`uint16`/`int16` in this
slice (not `bool`/`int32`/`int64`/floating-point, and not one-hot/multi-channel encodings).

For `sample_affine`/`apply_affine` specifically: `angle_range` is in degrees (positive =
counter-clockwise, matching `im.rotate`); `translation_x_range`/`translation_y_range` are in pixels
(positive `x` moves content right, positive `y` moves it down, matching `im.translate`);
`scale_range` is a positive, dimensionless, isotropic multiplier. `shear_x_range`/`shear_y_range`
are raw, dimensionless shear *coefficients*, not degrees: `shear_x` maps `x' = x + shear_x * y`,
and `shear_y` (applied after `shear_x`) then maps `y' = y + shear_y * x'`, using the already-sheared
`x'` -- documented as "shear x, then shear y", never as simultaneous, since that's exactly what the
sequential composition means. A positive `shear_x` moves the bottom of the image right relative to
the top; a positive `shear_y` moves the right side down relative to the left. There is no forbidden
angle and no `abs(shear)` limit -- this is a deliberate choice of parameterization:
`[[1, shear_x], [shear_y, 1 + shear_x*shear_y]]` has determinant `1` mathematically, for any finite
coefficients (area- and orientation-preserving), unlike the naive-looking `[[1, shear_x], [shear_y,
1]]`, which becomes singular whenever `shear_x * shear_y == 1` and flips orientation beyond that --
`improcv` never uses the naive form. That determinant-`1` guarantee is about the exact real-number
parameterization, not a promise of infinite `float64` precision: `improcv` rejects a `shear_x`/
`shear_y` pair whose product is large enough (roughly `2**52` in magnitude) that `float64` can no
longer tell `1.0 + shear_x*shear_y` apart from `shear_x*shear_y` itself, since that would silently
store a matrix that has lost the unit term making it invertible. Short of that, a large shear
coefficient is still accepted even though the resulting matrix can be very poorly conditioned,
strongly deform the image, or push its content outside the canvas entirely -- there is no automatic
protection against that beyond the float64-representability check. Each `*_range` is a
`(low, high)` tuple sampled independently via `Generator.uniform` -- `low` is always reachable,
equal endpoints sample that exact constant, but hitting `high` itself is not guaranteed for a
non-degenerate range (an ordinary property of continuous floating-point sampling). The transform is
always shear x, then shear y, then rotation + isotropic scale (all around the image center), then
translated -- this composition order is fixed and documented, not an implementation detail, since
shear does not commute with rotation, and translation does not commute with the rest in general.
`AffineParameters.matrix` (the `(2, 3)` matrix actually applied) is the sole source of truth for
replay; `angle`/`translation`/`scale`/`shear` are sampling metadata kept for debugging/logging/
`repr` only and are never used to reconstruct or cross-check the matrix. When `shear_x_range`/
`shear_y_range` are left at their `(0.0, 0.0)` default, no extra `rng` draw happens and the matrix
is bit-for-bit identical to what `sample_affine` produced before shear existed -- code written
before this feature keeps sampling `angle`/`translation`/`scale` from the exact same `rng` sequence,
call after call. Output spatial size always equals the source size -- this slice does not expand
the canvas the way `im.rotate_bound` does. The mask is always warped with nearest-neighbor
interpolation and a constant border (`mask_border_value`, default `0`); the caller cannot change
the mask's interpolation or border mode, only the fill value.

This slice covers flip, crop, and a shear+rotation+translation+isotropic-scale affine transform: no
perspective warp, no anisotropic scale, no canvas expansion, no photometric augmentation
(brightness/contrast/blur/noise), no bounding box/keypoint/polygon support, and no `Compose`-style
augmentation pipeline.

Dataset image discovery:

```python
files = im.discover_images(
    "dataset/images",
    recursive=True,
)
```

With custom extensions:

```python
files = im.discover_images(
    "dataset/images",
    extensions={".png", ".tif"},
)
```

`discover_images` finds candidate image files under a directory by filename extension only --
files are never opened or decoded, so an empty, corrupted, or non-image file with a matching
extension is still discovered. The result is a materialized, globally-sorted `tuple[Path, ...]`
(sorted by each path's POSIX-style form relative to `root`, independent of the underlying
filesystem's traversal order or the platform's path separator) -- this costs `O(N)` memory, so this
is not a streaming indexer for an arbitrarily large tree. Returned paths keep `root`'s own
relative/absolute form (e.g. `discover_images("data")` returns paths like `Path("data/cat.jpg")`,
never resolved or made absolute). `root` itself may be a symlink to a directory, but any symlink or
Windows reparse point (including a junction) found *while traversing* `root`'s contents is always
skipped, along with everything under it -- there is no `follow_symlinks` option. A descendant file
or directory whose name starts with `.` is skipped by default (this never applies to `root` itself,
so an explicitly given `root` like `".dataset"` is still searched); pass `include_hidden=True` to
include them. Filesystem errors are fail-fast: a missing/inaccessible `root`, or a permission error
encountered anywhere during traversal, raises immediately rather than silently skipping data --
every descendant is classified from a fresh filesystem check at inspection time, not a stale result
from listing the directory, so a file deleted or replaced right before being inspected still raises
rather than passing through unnoticed (a file changed *after* that check is an ordinary, undetected
race, same as with any filesystem operation). An
empty directory, or a directory with no matching files, returns `()`, not an error. This slice does
not pair images with masks, infer classes from directory names, produce dataset splits, or load/
decode any image, and adds no new dependency.

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
