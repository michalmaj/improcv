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
splitting that work into three separate releases after the fact, **`0.1.0a1` is designated as the
first public release and covers the accumulated scope of Phases 0–3 together** — a deliberate
decision (confirmed 2026-07-23), not an oversight. The phase-per-minor-version labels below
(`0.1.x`/`0.2.x`/`0.3.x`) are kept as a record of the *original* plan, not a claim about what
actually shipped under those version numbers — see [CHANGELOG.md](CHANGELOG.md)'s `[0.1.0a1]` entry
for the exact contents of that first release. Future phases (4 onward) will have their release
version decided at the time each is actually published, rather than mechanically pre-assigned now.

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
- [x] **Phase 4** — Quality metrics, perceptual hashing, photo/creative operations, non-local-means
  denoising, HDR imaging (exposure fusion, radiance merge, camera-response calibration, and tone
  mapping), and panorama/scan stitching. The quality metrics and perceptual hashes are implemented
  in-project using NumPy and base OpenCV, without requiring `opencv-contrib`. Advanced Stitcher
  masks/configuration, AlignMTB, seamless-clone `*_WIDE` modes, and contrib-only wrapper variants
  remain deliberately out of scope. Target release: `0.2.0a1`.
- [ ] **Phase 5** — Light ML, built as independent vertical slices rather than one combined phase
  (an earlier version of this line described a single `improcv[ml]` extra gating the whole phase;
  a scope/architecture audit (2026-07-27) found that premature -- no slice identified so far actually
  needs a dependency beyond base OpenCV/NumPy and the existing `viz` extra, so no `improcv[ml]` extra
  exists or is currently planned). Slices, not all necessarily shipped in the same release: DNN
  preprocessing/model-loading (`cv2.dnn` wrappers -- DNN preprocessing implemented; ONNX-only model
  loading implemented; generic inference and backend/target wrappers are out of scope for now, not
  started, see [CHANGELOG.md](CHANGELOG.md)), classification-evaluation utilities deferred from
  Phase 3 (confusion matrix and accuracy/precision/recall/F1 core implemented; binary one-vs-rest
  ROC/precision-recall curves, ROC AUC, and average precision implemented; a generic trapezoidal
  `auc(x, y)` helper implemented, also covering trapezoidal PR-curve area through the supported
  composition `auc(curve.recall, curve.precision)` -- no separate score-level function for it was
  added, deliberately, to avoid a symbol that could be confused with `average_precision_score`;
  `sample_weight` for the four binary ranking functions and for `confusion_matrix`/
  `classification_metrics` implemented; multiclass one-vs-rest score-level ROC AUC/average
  precision (`multiclass_roc_auc_score`/`multiclass_average_precision_score`,
  `average=None`/`"macro"`/`"weighted"`/`"micro"`) implemented; one-vs-one and public multiclass
  ROC/PR curve types remain planned, not started, see
  [CHANGELOG.md](CHANGELOG.md)), image augmentation
  (flip and crop sampling/replay for image + optional segmentation mask implemented; affine
  shear/rotation/translation/isotropic-and-anisotropic-scale sampling/replay implemented;
  affine canvas expansion (`expand_affine_canvas`) implemented; perspective sampling/replay
  implemented; perspective canvas expansion remains planned, not started), and
  dataset discovery (deterministic,
  extension-based image discovery implemented; deterministic image/mask pairing implemented;
  deterministic train/validation/test dataset splitting (`improcv.dataset.split_dataset`,
  `DatasetSplit`) released in `0.4.0a2` -- occurrence-based partitioning of any `Sequence[T]` via
  the Largest Remainder Method, with no stratification/grouping/leakage guarantee; manifests and
  batching/loading remain planned, not started), and
  dataset image similarity (deterministic pairwise similarity search over precomputed perceptual
  hashes -- `improcv.similarity.find_similar_image_pairs` -- implemented, shipping in `0.3.0a1`;
  portable, deterministic perceptual-hash manifests (`improcv.manifest.PerceptualHashManifest`)
  with strict schema-v1 JSON and atomic `save`/`load` implemented, shipping in `0.3.0a2`; a
  deterministic, sequential dataset-to-manifest builder (`improcv.dataset.
  build_perceptual_hash_manifest`) that discovers a local dataset root, decodes each file exactly
  once with fixed grayscale semantics (Unicode-path-safe on Windows via `Path.read_bytes`/
  `cv2.imdecode`), and produces root-relative canonical identifiers implemented, shipping in
  `0.3.0a3`; `0.3.0b1` begins beta stabilization after completion of the planned `0.3.0` functional
  scope above -- no new major feature slice; `0.4.0a1` adds deterministic, in-memory comparison of
  two `PerceptualHashManifest` snapshots (`improcv.manifest.compare_perceptual_hash_manifests`),
  classifying every path as added/removed/changed/unchanged by canonical manifest path identity
  alone via a linear merge-join, with a rename always reported as `removed` + `added`, no
  filesystem I/O, and no manifest schema-v1 changes; incremental caching/freshness validation,
  duplicate groups/clustering, indexed/subquadratic search, and parallel hashing/search remain
  planned, not started, with no version assigned yet). Bounding boxes/
  keypoints/polygons in augmentation, a
  `Compose`-style augmentation pipeline, dataset manifests/batching, and DNN
  inference/model-specific wrappers are deferred, not approved. Release version(s) to be decided as
  each slice ships.
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
