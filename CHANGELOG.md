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
- `improcv.types`: `Image`, `ImageU8`, `Mask`, `TransformMatrix` type aliases.
- Optional extras `cv`, `cv-headless`, `cv-contrib`, `cv-contrib-headless` for installing an
  OpenCV distribution alongside improcv.

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
- Unified all mask-returning functions (`in_range`, `harris_corner`, `threshold`, `auto_canny`) on
  a single `uint8` `{0, 255}` convention (previously `in_range`/`harris_corner` returned `bool`).
- Several functions (`auto_canny`, `clahe`, `gamma_correction`, `histogram_equalization`,
  `apply_lut`, `threshold`'s `otsu`/`adaptive_*` methods) now raise a clear `TypeError` for an
  unsupported dtype instead of a raw `cv2.error`.
- `pip install improcv` alone no longer fails on `import improcv` with a bare
  `ModuleNotFoundError` when no OpenCV distribution is installed.
