# Roadmap

`improcv`'s planned phases toward a stable `1.0.0` API, built as one or more small, reviewable
vertical slices (own branch and PR per slice, TDD'd function by function) rather than one giant PR
per phase. See [CHANGELOG.md](CHANGELOG.md) for the exact list of what's shipped so far.

**Phase-to-release-version mapping — superseded 2026-07-23.** The original plan (agreed
2026-07-16/18) was a 1:1 mapping: `0.1.x` ships Phase 1 alone, `0.2.x` ships Phase 2 alone, `0.3.x`
ships Phase 3 alone, each as its own public release. In practice, no public release was ever cut
between phases — Phase 1 even had a dedicated "release hardening before first alpha" pass
(2026-07-19) that was never followed by an actual tag, and development continued straight into
Phase 2 and Phase 3 instead. By the time the project did a release-readiness audit (2026-07-23),
Phases 0–3 were all complete but nothing had ever been published. Rather than retroactively
splitting that work into three separate releases after the fact, **`0.1.0a1` is the first published
release and covers the accumulated scope of Phases 0–3 together** — a deliberate decision (confirmed
2026-07-23), not an oversight. The phase-per-minor-version labels below (`0.1.x`/`0.2.x`/`0.3.x`) are
kept as a record of the *original* plan, not a claim about what actually shipped under those version
numbers — see [CHANGELOG.md](CHANGELOG.md)'s `[0.1.0a1]` entry for what the first release actually
contains. Future phases (4 onward) will have their release version decided at the time each is
actually published, rather than mechanically pre-assigned now.

- [x] **Phase 0** — Project skeleton, CI, `resize`.
- [x] **Phase 1** (originally planned as `0.1.x`, see note above) — Core geometric transforms
  (`translate`, `rotate`, `rotate_bound`, `flip`, `crop`, `center_crop`, `pad`, `warp_affine`,
  `warp_perspective`), color space conversions, filters, morphology, edge/corner detection,
  pixel-level operations.
- [x] **Phase 2** (originally planned as `0.2.x`, see note above) — Contours and shape descriptors;
  region analysis (connected components, distance transform, flood fill); image analysis
  (histograms, moments, template matching, min/max location, mean/stddev); seeded segmentation and
  restoration (watershed, rect-initialized GrabCut, inpainting).
- [x] **Phase 3** (originally planned as `0.3.x`, see note above) — Feature detection and
  descriptors, gated behind an `improcv[viz]` extra for the visualization pieces. Built as its own
  set of small vertical slices (detectors/descriptors, matching, Hough, QR, drawing, visualization,
  FAST/blob/MSER, barcode — one PR per slice). AKAZE/BRISK/KAZE (need `opencv-contrib-python`) and
  classification-evaluation plots (confusion matrix, PR/ROC curves, class bar chart — closer to
  Phase 5's ML tooling) are deliberately out of scope here, not oversights; see
  [CHANGELOG.md](CHANGELOG.md) for the full reasoning.
- [ ] **Phase 4** — Photo/creative operations, quality metrics, perceptual hashing. Quality metrics
  (SSIM/GMSD/MSE) implemented natively in NumPy rather than depending on `opencv-contrib`, since
  `cv2.quality`/`cv2.img_hash` are contrib-only. Release version to be decided when this phase ships.
- [ ] **Phase 5** — Light ML: augmentation, dataset loading, `cv2.dnn` wrappers, gated behind an
  `improcv[ml]` extra. Release version to be decided when this phase ships.
- [ ] **Phase 6** — Camera calibration and 3D geometry. Release version to be decided when this
  phase ships.
- [ ] **Phase 7** — Video/camera capture and tracking (a lightweight IoU tracker only; SORT/
  ByteTrack-style tracking is left to integration with external libraries, not reimplemented here).
  Release version to be decided when this phase ships.
- [ ] Buffer/catch-up release band, if needed.
- [ ] Pre-1.0 stabilization pass.
- [ ] `1.0.0` — First stable API.

Explicitly out of scope for this project: ONNX Runtime inference and dedicated real-time multi-threading
pipeline infrastructure.
