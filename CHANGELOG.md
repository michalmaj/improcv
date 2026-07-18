# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

## [0.1.0] - 2026-07-18

### Added
- Initial project skeleton: `pyproject.toml` (Hatchling, `uv`), Ruff/Pyright/pytest configuration,
  MIT license, README, GitHub Actions CI.
- `resize`.
