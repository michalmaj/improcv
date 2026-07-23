# Roadmap

`improcv`'s planned phases toward a stable `1.0.0` API. Each phase maps to a working `0.x` release line
and is built as one or more small, reviewable vertical slices (own branch and PR per slice, TDD'd function
by function) rather than one giant PR per phase. See [CHANGELOG.md](CHANGELOG.md) for the exact list of
what's shipped so far.

- [x] **Phase 0** — Project skeleton, CI, `resize`.
- [x] **Phase 1** (`0.1.x`) — Core geometric transforms (`translate`, `rotate`, `rotate_bound`, `flip`,
  `crop`, `center_crop`, `pad`, `warp_affine`, `warp_perspective`), color space conversions, filters,
  morphology, edge/corner detection, pixel-level operations.
- [x] **Phase 2** (`0.2.x`) — Contours and shape descriptors; region analysis (connected components,
  distance transform, flood fill); image analysis (histograms, moments, template matching, min/max
  location, mean/stddev); seeded segmentation and restoration (watershed, rect-initialized GrabCut,
  inpainting).
- [x] **Phase 3** (`0.3.x`) — Feature detection and descriptors, gated behind an `improcv[viz]` extra for
  the visualization pieces. Built as its own set of small vertical slices (detectors/descriptors,
  matching, Hough, QR, drawing, visualization, FAST/blob/MSER, barcode — one PR per slice). AKAZE/
  BRISK/KAZE (need `opencv-contrib-python`) and classification-evaluation plots (confusion matrix,
  PR/ROC curves, class bar chart — closer to Phase 5's ML tooling) are deliberately out of scope here,
  not oversights; see [CHANGELOG.md](CHANGELOG.md) for the full reasoning.
- [ ] **Phase 4** (`0.4.x`) — Photo/creative operations, quality metrics, perceptual hashing. Quality
  metrics (SSIM/GMSD/MSE) implemented natively in NumPy rather than depending on `opencv-contrib`, since
  `cv2.quality`/`cv2.img_hash` are contrib-only.
- [ ] **Phase 5** (`0.5.x`) — Light ML: augmentation, dataset loading, `cv2.dnn` wrappers, gated behind an
  `improcv[ml]` extra.
- [ ] **Phase 6** (`0.6.x`) — Camera calibration and 3D geometry.
- [ ] **Phase 7** (`0.7.x`) — Video/camera capture and tracking (a lightweight IoU tracker only; SORT/
  ByteTrack-style tracking is left to integration with external libraries, not reimplemented here).
- [ ] `0.8.x` — Buffer/catch-up release.
- [ ] `0.9.x` — Pre-1.0 stabilization.
- [ ] `1.0.0` — First stable API.

Explicitly out of scope for this project: ONNX Runtime inference and dedicated real-time multi-threading
pipeline infrastructure.
