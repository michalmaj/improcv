# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version strings (e.g. `0.1.0a1`) follow [PEP 440](https://peps.python.org/pep-0440/), Python's
packaging version standard — not SemVer's pre-release hyphen syntax (`0.1.0-alpha.1`) — though the
project's *compatibility policy* is still SemVer-style: pre-`1.0.0`, any `0.MINOR` bump may include
breaking changes; post-`1.0.0`, only a `MAJOR` bump may.

## [Unreleased]

### Added
- New `improcv.quality` module, Phase 4 slice 1 (quality metrics — core): `mse`, `psnr`, `ssim`.
  `mse` is the mean squared error over every element (including channels), always finite and
  non-negative. `psnr` is computed by this project's own formula
  (`20*log10(data_range) - 10*log10(mse)`), not `cv2.PSNR` -- verified directly that `cv2.PSNR`
  returns a large-but-finite sentinel (`~361.2`) for identical images instead of the mathematically
  correct `math.inf`, and silently doesn't scale its default reference value for `uint16` unless the
  caller passes one explicitly; `psnr` returns `math.inf` for identical images and allows a negative
  result when the error exceeds `data_range`, uncapped. `ssim` implements the windowed-Gaussian
  variant from Wang et al. (2004) (`11x11` window, `sigma=1.5`, `K1=0.01`, `K2=0.03`, population
  covariance) via `cv2.GaussianBlur` with an explicit `11x11` kernel size, not NumPy/SciPy -- no new
  runtime dependency. The outermost 5-pixel border (where the window would extend past the image
  edge) is excluded from the final scalar, matching the standard "valid" convolution region rather
  than depending on a border-extension choice; multi-channel images (including BGRA's alpha channel)
  get SSIM computed independently per channel, then averaged with the valid spatial region. Cross-
  checked numerically against `scikit-image` 0.26.0's `structural_similarity(...,
  gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` in an isolated, throwaway
  environment -- agreement at floating-point precision (exact match or low-`1e-9`-and-below,
  depending on dtype) once both the border crop and the explicit `11x11` kernel size were in place;
  `scikit-image` itself was **not** added as a project dependency, runtime or dev. All three
  functions share one validator: both images must be non-empty, 2D or 3D with identical shape and
  dtype (`uint8`/`uint16`/`float32`/`float64`), 1-4 channels, and (for float inputs) free of
  `NaN`/infinity. `data_range` defaults to `255.0`/`65535.0` for `uint8`/`uint16` and must be given
  explicitly (a positive, finite, non-bool number) for `float32`/`float64`, since a float image's
  actual range isn't implied by its dtype; input values are never clipped or rescaled to it. None of
  the three functions mutate their inputs. This is the first of two Phase 4 quality-metrics slices --
  GMSD is a separate, later slice.

### Changed

### Fixed
- `mse`/`psnr`: two distinct, non-identical images could previously be misreported as identical
  (`mse == 0.0`, `psnr == math.inf`) when their squared difference underflowed `float64` (e.g. a
  single-pixel `float64` offset of `1e-162` -- `(1e-162)**2` rounds to exactly `0.0`, below the
  smallest representable subnormal). `mse` now normalizes by the largest absolute difference before
  squaring, so only a genuine, mathematically unavoidable underflow of the true (non-zero) result
  raises a clear `ValueError` instead of silently returning `0.0`; `psnr` computes the error's
  base-10 logarithm directly from that same normalization, so it stays correctly finite (e.g. `~3240`
  dB for the `1e-162` example) even in exactly the case where `mse` itself must raise.
- `ssim`: a `data_range` large enough that `(K2*data_range)**2` alone exceeds `float64`'s range (e.g.
  `1e156`) previously raised a raw `OverflowError`; smaller but still very large values (e.g. `1e100`)
  produced `NaN` from an internal `inf/inf` division, surfacing as a generic non-finite-result error
  with no indication of the cause. Both images and `data_range` are now rescaled together by a common
  factor before computing the windowed statistics whenever either is large enough to risk it (an
  exact invariance of the SSIM formula, verified re-checked against `scikit-image` for the existing,
  non-extreme reference vectors -- no precision change for ordinary inputs, which never reach this
  code path).

## [0.1.0a1] - 2026-07-23

First published release: Phases 0-3 (skeleton, core transforms/color/filters/morphology/edges/pixel
ops, contours/region analysis/image analysis/segmentation/restoration, feature detection/matching/
Hough/QR/drawing/visualization/detectors/barcode).

**On the version number**: the project's original plan (see `ROADMAP.md`) mapped `0.1.x`/`0.2.x`/
`0.3.x` to Phases 1/2/3 as separate releases. No release was ever cut between phases -- Phase 1 even
had its own dedicated pre-alpha hardening pass that was never tagged -- so `0.1.0a1` is designated
as the first public release, covering the accumulated scope of Phases 0-3 together, a deliberate
decision made once a release-readiness audit found nothing had ever actually shipped. See
`ROADMAP.md` for the full explanation.

### Added
- Initial project skeleton: `pyproject.toml` (Hatchling, `uv`), Ruff/Pyright/pytest configuration,
  MIT license, README, GitHub Actions CI.
- `improcv.__version__`, read from installed package metadata
  (`importlib.metadata.version("improcv")`), falling back to `"0.0.0.dev0"` only when run from an
  uninstalled source checkout.
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
- New `improcv.drawing` module: `draw_contours`, `draw_bounding_boxes`, `montage` -- plain `cv2` +
  `numpy` only, no new dependency. `draw_contours`/`draw_bounding_boxes` always draw onto a copy of
  the input image, fixing `cv2.drawContours`/`cv2.rectangle`'s verified in-place-mutation behavior;
  both require a 3-channel BGR `uint8` image (grayscale/BGRA rejected), since OpenCV silently uses
  only a color tuple's first element as a grayscale value rather than raising. `color` and
  `thickness` are validated and normalized (integral, no `bool`, `color` channels in `[0, 255]`,
  `thickness` never `0` -- OpenCV silently treats `thickness=0` as a thin outline rather than "draw
  nothing" -- and, if positive, capped at `32767`, OpenCV's own internal `MAX_THICKNESS` limit
  (`32768` reaches a raw `cv2.error: thickness <= MAX_THICKNESS`); negative `thickness` has no such
  cap but must still fit signed `int32`). `draw_bounding_boxes` uses `cv2.rectangle`'s `Rect` overload
  (`(x, y, width, height)`) rather than its two-point overload, which was verified to draw a filled
  region one pixel wider and taller than intended. `BoundingBox` fields are normalized to plain
  Python `int` before computing `x + width`/`y + height` -- verified that adding two `np.int32`
  scalars near `int32`'s max silently wraps around (only a `RuntimeWarning`, easy to miss) rather
  than raising, which would otherwise let an out-of-range box slip past that very bounds check. A
  wrong-dtype contour now raises `TypeError` (was inconsistently `ValueError`), matching every other
  dtype check in the library. `draw_contours` documents that filling multiple
  contours without hierarchy applies OpenCV's even-odd rule across the whole collection (verified:
  nested-but-unrelated contours filled in one call can produce an unintended "hole"); hierarchy
  support itself is out of scope. `montage` tiles same-`ndim`/channel-count images
  (`(H,W)`/`(H,W,3)`/`(H,W,4)` only) into a grid via a hard (non-aspect-preserving) resize per tile,
  picking `cv2.INTER_AREA` when shrinking and `cv2.INTER_LINEAR` when enlarging or mixed-scaling
  (per OpenCV's own interpolation guidance), and rejects a requested output size above a
  `512 MiB` safety cap with `ValueError` before any allocation or resize call -- the same
  before-allocation-safety-check pattern as `hough_circles`'s accumulator/radius guards, applied
  here to montage's own memory-exhaustion risk. `draw_keypoints`/`draw_matches` wrappers were
  considered and rejected: `cv2.drawKeypoints`/`cv2.drawMatches` with `outImg=None` already return a
  fresh, non-mutated array, so a wrapper would add no value.
- New `improcv.visualization` subpackage: `show_image`, `plot_histogram` -- matplotlib-based, requires
  the new optional `viz` extra (`pip install "improcv[viz]"`). `import improcv` never imports this
  subpackage or matplotlib; importing `improcv.visualization` without the extra installed raises a
  clear `ImportError` naming the missing extra, following the existing `cv2`-missing guard's pattern.
  The subpackage itself only imports `matplotlib`/`matplotlib.axes` at module load -- `matplotlib.
  pyplot` (which resolves a rendering backend as a side effect of import) is imported lazily, only
  when a caller doesn't supply their own `ax`, so importing the subpackage never forces a backend
  choice. `show_image` converts BGR to RGB via `bgr_to_rgb` before display (matplotlib interprets
  channel 0 as red, so an unconverted BGR image displays with red and blue visually swapped) and
  shows grayscale images with `cmap="gray"` and a fixed `vmin=0`/`vmax=255` range (matplotlib's own
  defaults are `cmap="viridis"` and a per-image-normalized range, which would make images of
  different uniform brightness indistinguishable). `plot_histogram` plots one line per channel
  (black for grayscale; blue/green/red for BGR, matching OpenCV's channel order) against each bin's
  *center value* rather than its raw index, so the x-axis reflects `value_range` directly. Both
  functions accept an optional `ax` and return the `Axes` used, never calling `plt.show()`. Neither
  is re-exported from the top-level `improcv` package. `confusion_matrix`/PR-curve/ROC-curve/
  class-bar-chart plotting (classification-evaluation helpers, a different functional area) remain a
  separate, later chunk.
- New `improcv.detectors` module, closing a Phase 3 scope gap found in a completeness audit (FAST,
  blob, and MSER detectors were listed in the original roadmap but never implemented, unlike
  AKAZE/BRISK/KAZE, which are deliberately deferred to a contrib-gated chunk): `detect_fast_keypoints`,
  `detect_blob_keypoints`, `detect_mser_regions`. All three accept grayscale/BGR/BGRA `uint8` images
  (verified all three work correctly with 4 channels too, a wider contract than `drawing.py`/
  `qrcode.py`). `detect_fast_keypoints`'s `threshold` (bounded to `[0, 255]`) and `fast_type` are
  validated explicitly -- OpenCV silently accepts out-of-range/invalid values for both with
  undefined-looking behavior rather than raising. `detect_blob_keypoints` passes a
  `cv2.SimpleBlobDetector.Params` straight through rather than re-exposing its 14 fields, converting a
  structurally-valid-but-internally-invalid configuration (e.g. `thresholdStep <= 0`) from a raw
  `cv2.error` into `ValueError`. `detect_mser_regions` returns a new `MSERRegion` type (`points`,
  `bounding_box`) rather than reusing `Contour` -- verified directly (by rendering a region's points
  into its own bounding box) that MSER's region output is an **unordered set of every pixel in the
  region**, not an ordered boundary walk, so passing it to `draw_contours` would connect points in
  arbitrary order and draw a nonsensical zigzag polygon; `MSERRegion.points` documents this explicitly
  and recommends `find_contours`/`convex_hull` for an actual ordered boundary. `detect_mser_regions`
  also rejects images smaller than 3x3 (OpenCV's own hard floor) and normalizes MSER's `bboxes=()`
  empty-result quirk and a documented pybind11 edge case where a region's points can come back with
  `dtype=object`. Barcode detection (via `cv2.barcode.BarcodeDetector`) remains a separate, later
  chunk -- verified to behave differently from QR's `GraphicalCodeDetector` (a single
  `detectAndDecodeWithType` call already handles multiple codes correctly with no `straight_codes`-style
  misalignment).
- This completes Phase 2's functional scope (contours, region analysis, image analysis, segmentation and
  restoration) — remaining pre-1.0.0 work moves to Phase 3.
- New `improcv.barcode` module, closing the last Phase 3 completeness-audit gap: `decode_barcodes`,
  built on `cv2.barcode.BarcodeDetector.detectAndDecodeWithType`. Unlike QR, OpenCV's barcode detector
  finds all barcodes in one call regardless of count, so only a single function is needed (no
  `decode_barcode`/`decode_barcodes` split). Verified that OpenCV 4.13/5.0 currently instantiate only
  an EAN-13 and an EAN-8 decoder internally -- **Code128 and UPC-E are not supported**; UPC-A is
  produced as a special case of EAN-13 (a decoded payload starting with `'0'` has that leading zero
  stripped and its type changed to `"UPC_A"`). `Barcode.data`/`Barcode.barcode_type` are both `None`
  when a barcode-shaped quadrangle was detected but its content could not be decoded -- verified,
  unlike `QRCode`, that barcode formats have no "successfully decoded but empty" state, so only two
  outcomes exist rather than QR's three. `decode_barcodes` rejects images with either spatial
  dimension of 40 pixels or less: verified directly that OpenCV silently never attempts detection
  below that size, returning results indistinguishable from "nothing found" -- previously this would
  have produced a misleading `[]`. The raw `retval` from `detectAndDecodeWithType` is validated but
  never used to decide whether to return `[]`: verified that `retval` only means "at least one code
  decoded successfully", not "anything was detected" -- an all-corrupted multi-barcode image returns
  `retval=False` with non-empty, all-undecodable results, which are still returned rather than
  dropped. The `*BytesMulti` detector variants are out of scope: verified they reproduce the same
  `decoded_info`/`straight_codes` index-misalignment bug that `detectAndDecodeMulti` has for QR codes
  (`decoded_info` length 2, `straight_codes` length 0 on a 2-barcode image), which
  `detectAndDecodeWithType` does not have since it lacks a `straight_code`-shaped field. Each decoded
  quadrangle is also rejected as degenerate (zero-area, e.g. four identical or collinear corners) via
  the same `float64` shoelace-formula guard used in `improcv.qrcode`.
- **This completes Phase 3's functional scope.** Two items originally listed under Phase 3 remain
  explicitly out of scope for now, by deliberate decision rather than oversight: AKAZE/BRISK/KAZE
  detect+describe (confirmed absent from the non-contrib OpenCV build; would need an
  `opencv-contrib-python` dependency, and this project's OpenCV-distribution policy is still an open
  decision per the project brief) and `confusion_matrix`/PR-ROC-curve/
  class-bar-chart plots (a classification-evaluation concern conceptually closer to Phase 5's ML
  tooling than to this phase's image-display visualization, per the `improcv.visualization` chunk's
  own scoping decision). `draw_keypoints`/`draw_matches` were also considered and rejected outright
  (already-safe raw `cv2` calls; wrapping them would be value-less aliases) rather than deferred.
  Remaining pre-1.0.0 work moves to Phase 4 or a release-hardening pass.

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
