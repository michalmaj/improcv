# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing has been published yet (no PyPI/TestPyPI release, no git tag) — everything so far stays
under `Unreleased` rather than a dated version section, even though `pyproject.toml` already
carries a working `0.1.0a1` version number for local development.

### Added
- Initial project skeleton: `pyproject.toml` (Hatchling, `uv`), Ruff/Pyright/pytest configuration,
  MIT license, README, GitHub Actions CI.
- `resize`.
- Core geometric transforms: `translate`, `rotate`, `rotate_bound`, `flip`, `crop`,
  `center_crop`, `pad`, `warp_affine`, `warp_perspective`.
- Color space conversions: `bgr_to_rgb`, `rgb_to_bgr`, `ensure_gray`, `to_hsv`, `to_lab`,
  `to_ycrcb`.
- Filters: `gaussian_blur`, `median_blur`, `bilateral_filter`, `clahe`, `gamma_correction`,
  `histogram_equalization`.
- Morphology: `threshold` (binary/Otsu/adaptive), `dilate`, `erode`, `morph_open`, `morph_close`,
  `morph_gradient`, `tophat`, `blackhat`.
- Edge and corner detection: `auto_canny`, `sobel_edge`, `laplacian_edge`, `harris_corner`.
- Pixel-level operations: `in_range`, `invert`, `adjust_brightness`, `adjust_contrast`,
  `alpha_blend`, `bitwise_and`, `bitwise_or`, `apply_lut`.
- `improcv.types`: `Image`, `ImageU8`, `Mask`, `TransformMatrix`, `ImageFloat32`, `BoundingBox`
  type aliases.
- Optional extras `cv`, `cv-headless`, `cv-contrib`, `cv-contrib-headless` for installing an
  OpenCV distribution alongside improcv.
- Contours: `find_contours`, `sort_contours`, `bounding_boxes`, `convex_hull`, `approx_poly_dp`,
  `min_area_rect`.
- `improcv.contours`: `Contour`, `Hierarchy`, `RotatedRect` types (`BoundingBox` lives in
  `improcv.types`, re-exported here — see below).
- Region analysis: `connected_components`, `connected_components_with_stats`,
  `distance_transform`, `flood_fill`.
- `improcv.regions`: `Connectivity`, `Labels`, `ComponentStats`, `Centroids`, `DistanceType`,
  `DistanceMaskSize`, `FloodFillResult` types.
- Image analysis: `histogram`, `moments`, `match_template`, `min_max_loc`, `mean_stddev`.
- `improcv.analysis`: `Moments`, `TemplateMatchMethod`, `MinMaxResult`, `MeanStdDevResult` types.
- `improcv._compat.opencv`: the project's first compat-layer helper, `_normalize_calc_hist_output`,
  isolating a genuine `cv2.calcHist` shape difference between OpenCV 4.x and 5.x.
- Segmentation and restoration: `watershed`, `grabcut_rect`, `inpaint`.
- `improcv.restoration`: `InpaintMethod` type.
- Feature detection and description: `detect_and_compute` (ORB, SIFT).
- `improcv.features`: `FeatureMethod`, `DescriptorNorm`, `Features` types.
- Feature matching: `match_features` (brute-force nearest-neighbor, with or without cross-check),
  accepting two `Features` values so a caller can never pass a norm mismatched to the descriptor
  type. Raw `list[cv2.DMatch]`, sorted by distance ascending; no ratio test, KNN, FLANN, RANSAC, or
  match drawing in this slice.
- `match_features_ratio`: KNN (`k=2`) matching filtered by Lowe's ratio test, sharing
  `match_features`'s `Features`-contract validation and L2-magnitude guard. Same raw
  `list[cv2.DMatch]`, sorted-by-distance return contract. Still no FLANN, RANSAC, homography, or
  match drawing.
- `find_homography`: RANSAC homography estimation from two `Features` values and a
  `list[cv2.DMatch]`. Rejects non-finite matched keypoint coordinates explicitly (verified OpenCV
  does not safely handle these itself at the 4-correspondence minimum) and independently recomputes
  `inlier_mask` from the final homography and reprojection threshold rather than trusting OpenCV's
  own raw mask (which has a documented historical correctness bug in versions near this project's
  `4.9` floor). `homography` is `None` for legitimately degenerate (but finite) geometry, not an
  error. `improcv.features`: `HomographyResult` type. Still RANSAC-only, no FLANN, perspective-warp
  helper, or match drawing.
- New `improcv.hough` module: `hough_lines` (standard Hough transform), `hough_line_segments`
  (probabilistic Hough transform), `hough_circles` (`HOUGH_GRADIENT`/`HOUGH_GRADIENT_ALT`). `rho`/
  `theta`/`dp`/`param1` defaults are `improcv`'s own choices matching OpenCV's own C++ defaults, not
  something OpenCV itself defaults to -- all are required parameters in OpenCV's own signatures;
  scale-dependent parameters (`threshold`, `min_dist`) have no default at all. `rho`/`theta` are
  validated as strictly positive before ever calling OpenCV, since a non-positive value crashes
  uncontrolled on both supported OpenCV versions rather than raising cleanly. `hough_circles`'s
  `param2` resolves method-dependently when omitted, since OpenCV's own omitted-parameter default
  violates `HOUGH_GRADIENT_ALT`'s own required range; `max_radius`'s "centers only" negative-value
  semantics are `HOUGH_GRADIENT`-only, and an explicit `0 < max_radius <= min_radius` range is
  rejected rather than silently widened or reordered by OpenCV. `improcv.hough`: `Line`,
  `LineSegment`, `Circle`, `HoughCircleMethod` types.
- `improcv._compat.opencv`: `_normalize_hough_lines_p_output`, isolating a genuine `cv2.HoughLinesP`
  shape difference between OpenCV 4.x and 5.x.
- New `improcv.qrcode` module: `decode_qr_code` (single QR code) and `decode_qr_codes` (multiple),
  built on `cv2.QRCodeDetector` only (not `QRCodeDetectorAruco`). `decode_qr_codes` detects all
  quadrangles with `detectMulti` and decodes each one individually with its own `decode` call,
  rather than trusting `detectAndDecodeMulti`'s `straight_codes` output -- verified that OpenCV's
  Python binding silently drops the `straight_codes` entry for any quadrangle that fails to decode,
  making it unaligned with `decoded_info`/`points` whenever a batch has a mixed success/failure
  result; `decode_qr_code` shares the same per-quadrangle decode path via `detect`+`decode` for
  consistency. `QRCode.data` is `None` when a quadrangle was detected but its content could not be
  decoded, `""` when it was decoded and genuinely encodes empty content (these are distinguished via
  `straight_code`, not `retval`, since `retval == ""` is identical in both cases), or the decoded
  UTF-8 string otherwise -- a non-UTF-8 payload raises `ValueError` rather than a raw
  `UnicodeDecodeError`. `decode_qr_code` attempts to detect and decode one QR code; if `image`
  contains multiple QR codes, OpenCV may select one of them or fail to detect any -- which code (if
  any) is selected is not guaranteed -- use `decode_qr_codes` for images that may contain multiple
  codes. Each result represents one physical QR symbol; Structured Append sequences are not
  reassembled. `improcv.qrcode`: `QRCode` type.
- This completes Phase 2's functional scope (contours, region analysis, image analysis, segmentation and
  restoration) — remaining pre-1.0.0 work moves to Phase 3.

### Changed
- `BoundingBox` moved from `improcv.contours` to `improcv.types` (still importable from both
  locations — no existing import breaks).

### Fixed
- `adjust_brightness`/`adjust_contrast`: no longer reflect negative results back to positive via
  `convertScaleAbs`'s implicit `abs()`; `adjust_contrast` now scales around mid-gray (128), not 0.
- `threshold`: an unrecognized `method` no longer silently runs `adaptive_gaussian`.
- `clahe`: a non-positive `clip_limit`/`tile_grid_size` no longer reaches OpenCV (previously a
  low-level crash on some builds).
- `resize`: a computed dimension can no longer round down to 0 pixels; `width`/`height` must be
  actual ints; empty images are now rejected globally.
- `rotate_bound`: the expanded canvas is no longer truncated by up to 1px (was cropping corners on
  small images despite the "never crop" contract).
- `warp_affine`/`warp_perspective`: a non-positive `output_size` is now rejected instead of being
  silently ignored (OpenCV itself silently returns the *input* size for an invalid `dsize`).
- Unified the mask-returning functions (`in_range`, `harris_corner`, `auto_canny`) on a single
  `uint8` `{0, 255}` convention (previously `in_range`/`harris_corner` returned `bool`). `threshold`
  is intentionally not one of these — it accepts any `max_value` and tolerates non-`uint8` dtypes
  in `"binary"` mode (e.g. `float32` in, `float32` out), so it does not always produce a `{0, 255}`
  `uint8` mask; it stays a flexible `Image -> Image` function instead.
- Several functions (`auto_canny`, `clahe`, `gamma_correction`, `histogram_equalization`,
  `apply_lut`, `threshold`'s `otsu`/`adaptive_*` methods) now raise a clear `TypeError` for an
  unsupported dtype instead of a raw `cv2.error`.
- `pip install improcv` alone no longer fails on `import improcv` with a bare
  `ModuleNotFoundError` when no OpenCV distribution is installed.
- `require_image_ndim` (used across the whole library) now rejects a zero-channel `(H, W, 0)`
  array, not only a zero-height/zero-width one — verified directly that at least one `cv2.*` call
  returned uninitialized-memory garbage for that shape instead of raising.
- `mean_stddev`/`histogram` now reject an image with more than 128 channels: `cv2.meanStdDev`/
  `cv2.calcHist` silently misinterpret channel counts above 128 on OpenCV 5.x (correct up to 512 on
  OpenCV 4.x), so 128 is enforced as the common, cross-version-safe limit instead of silently
  returning wrong statistics on one OpenCV line.
