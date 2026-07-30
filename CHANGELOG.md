# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version strings (e.g. `0.1.0a1`) follow [PEP 440](https://peps.python.org/pep-0440/), Python's
packaging version standard — not SemVer's pre-release hyphen syntax (`0.1.0-alpha.1`) — though the
project's *compatibility policy* is still SemVer-style: pre-`1.0.0`, any `0.MINOR` bump may include
breaking changes; post-`1.0.0`, only a `MAJOR` bump may.

## [Unreleased]

### Added
- New `improcv.dnn` module, Phase 5 slice (DNN blob preprocessing): `create_dnn_blob` and
  `create_dnn_batch_blob`, wrapping `cv2.dnn.blobFromImage`/`blobFromImages` to turn `uint8`/
  `float32` image(s) into a 4-D NCHW `float32` blob for `cv2.dnn.Net.setInput`. This slice covers
  preprocessing only -- no model loading, no `cv2.dnn.Net` handling, and no inference. Input is
  restricted to `uint8`/`float32`, grayscale/`(H, W, 1)`/BGR/BGRA -- verified directly (source and
  empirically, across OpenCV 4.9/4.13/5.0) that other dtypes accepted by raw OpenCV on some versions
  (`int16`, `uint16`, `float64`) are silently converted on OpenCV >= 4.13 but raise a raw `cv2.error`
  on OpenCV 4.9 (this project's floor), and that OpenCV 5.0 removed the Caffe/Darknet/Torch model
  loaders present on 4.x -- both are reasons this slice deliberately stays narrow. Output is always
  `float32` (no `output_dtype`/`uint8`-output parameter yet -- `ddepth=CV_8U` forbids any non-default
  `scale`/`mean` in OpenCV itself, and mixing it with non-`uint8` input silently wraps around on
  OpenCV >= 4.13 instead of raising, so it was left out rather than exposed unsafely). A scalar `mean`
  is broadcast by this wrapper to every channel -- passing that same scalar directly to raw
  `cv2.dnn.blobFromImage` would not broadcast it, only set the first channel's mean. `size` is
  required (not optional) for `create_dnn_batch_blob` -- verified directly that without it, OpenCV
  silently resizes every image after the first to match the first image's native size. `images` for
  the batch function must be a real `Sequence` (a single `np.ndarray`, including a 4-D stack, is
  rejected, matching `stitch_images`'s existing container contract) whose elements must share dtype
  and channel count; spatial shape may differ, since the required `size` normalizes it. No new
  runtime dependency and no `improcv[ml]` extra were introduced -- `cv2.dnn` is part of the base
  OpenCV install already required by the existing `cv`/`cv-headless`/`cv-contrib`/
  `cv-contrib-headless` extras.
- `improcv.dnn`, Phase 5 slice (ONNX-only model loading): `load_onnx_network` (from a path) and
  `load_onnx_network_from_bytes` (from an immutable `bytes` buffer), both returning a `cv2.dnn.Net`.
  ONNX-only, not a general "load any DNN model" function -- other formats are out of scope. Two
  separate functions rather than one polymorphic function, because a bare `bytes` argument is
  genuinely ambiguous to raw OpenCV: verified directly that `cv2.dnn.readNetFromONNX(some_bytes)`
  (positional) is silently routed to the *path* overload on OpenCV 4.13/5.0 but to the *buffer*
  overload on OpenCV 4.9 -- a real behavioral difference between supported versions, stabilized here
  by always passing the buffer overload by keyword as a `uint8` array. `path` similarly stabilizes a
  real difference: a bare `pathlib.Path` is accepted directly by OpenCV 4.13/5.0 but raises a
  `cv2.error` on OpenCV 4.9, so this wrapper normalizes to `str` internally regardless of the
  installed version. Filesystem problems Python can detect directly (missing file, a directory, an
  empty file) raise native exceptions (`FileNotFoundError`, `IsADirectoryError`, `ValueError`) rather
  than OpenCV's own generic, often-indistinguishable error messages; anything OpenCV's parser itself
  rejects (a corrupt model, or a permission/ACL problem only visible once OpenCV opens the file)
  raises `RuntimeError` with the original `cv2.error` as `__cause__`. A successful load is checked
  against a `not net.empty()` postcondition. On OpenCV 5, this slice requests `ENGINE_CLASSIC` (the
  only engine that exists on 4.x) as a best-effort common behavior across the whole 4.9-5.x range --
  not a guarantee, since the process-level `OPENCV_FORCE_DNN_ENGINE` environment variable can override
  it; there is no public `engine` parameter. Every call loads a fresh, independent `Net` -- nothing is
  cached. This slice covers loading only: no `setInput`/`forward` wrapper, no backend/target API, no
  model-specific postprocessing, and no new dependency (`onnx` is used only to regenerate the test
  fixture in a throwaway environment, never a project or CI dependency).
- New `improcv.evaluation` module, Phase 5 slice (classification evaluation core): `confusion_matrix`,
  `classification_metrics`, and `classification_metrics_from_confusion_matrix`, covering confusion
  matrices, accuracy, and per-class/aggregated precision/recall/F1/support for single-label
  multiclass classification with integer labels. Rows are true labels, columns are predicted labels;
  `labels=None` infers the sorted union of observed values, an explicit `labels` fixes the exact
  order. Two deliberate departures from `scikit-learn.metrics`'s well-known behavior: a duplicate
  value in an explicit `labels` and an observed label outside an explicit `labels` both raise
  `ValueError`, rather than being silently accepted (duplicates) or silently dropping that sample
  from the result (unknown labels), as `sklearn.metrics.confusion_matrix` does. `average` is
  `None`/`"micro"`/`"macro"`/`"weighted"` (no `"binary"` in this slice); `zero_division` is `0.0`,
  `1.0`, or `"nan"` (no `"warn"`), computed without ever letting NumPy perform an actual `0/0`
  division, so no `RuntimeWarning` is raised regardless of the value chosen. `ConfusionMatrixResult`/
  `ClassificationMetrics` are frozen dataclasses with a hand-written `__eq__` (comparing `ndarray`
  fields via `np.array_equal`, with `equal_nan=True` for the float metric fields) and `__hash__ =
  None`, since the default dataclass-generated equality would hit NumPy's "truth value of an array is
  ambiguous" error for any non-trivial result; all `ndarray` fields on both types are new, independent
  arrays, marked read-only. No ROC/PR curves, AUC, plotting, multilabel support, sample weights, or
  `average="binary"` in this slice, and no new dependency -- `scikit-learn` was used only as a
  one-off, non-dependency oracle during development to cross-check every documented behavior and
  departure above, never imported by any committed code.
- New `improcv.augmentation` module, Phase 5 slice (geometric augmentation sampling and replay for
  flip and crop): `sample_flip`/`sample_crop` and `apply_flip`/`apply_crop`, plus the
  `FlipParameters`/`CropParameters`/`AugmentedImageMask` result types. Sampling requires an explicit
  `rng: np.random.Generator` (checked by `isinstance`, never a global/implicit RNG) and returns a
  small, independent, reusable parameter object; applying that object is a pure function of it, so
  the same sampled flip/crop can be replayed any number of times, always producing the same result.
  `apply_flip`/`apply_crop` optionally take a segmentation `mask=`, applying the identical
  transform to it and returning an `AugmentedImageMask` instead of a bare array, so an image and its
  mask always stay synchronized. `CropParameters` carries the `source_size` it was sampled for;
  `apply_crop` requires the image (and mask) to match that size exactly, refusing to replay
  parameters against a differently-sized image instead of silently cropping the wrong region. A
  segmentation mask is restricted to `uint8`/`uint16`/`int16` in this slice (not `bool`/`int32`/
  `int64`/floating-point, and not one-hot/multi-channel encodings) -- narrower than this project's
  general image dtype contract, deliberately, since a mask holds class labels rather than pixel
  intensities; widening it (e.g. to `int32`) later is a compatible extension. Both apply functions
  reuse the existing `improcv.transforms.flip`/`crop` unmodified for the actual pixel work -- no
  slicing or `cv2.flip` call is reimplemented here -- and always return a new, independent array
  (or pair), including for a no-op flip and for a crop covering the entire source image. This slice
  covers flip and crop only: no affine transforms (rotation/translation/scale/shear), no perspective
  warp, no photometric augmentation, no bounding box/keypoint/polygon support, no `Compose`-style
  pipeline, and no new dependency.
- `improcv.augmentation`: affine augmentation sampling and replay for rotation, translation, and
  isotropic scale -- `sample_affine`/`apply_affine`, plus the `AffineParameters` result type. Covers
  a stable similarity-transform subset of the general affine group; shear is a deliberately separate,
  future extension (its parameterization and singularity behavior need their own audit), and
  perspective, canvas expansion (a `rotate_bound`-style growing output), bounding boxes/keypoints/
  polygons, and any `Compose`-style pipeline remain out of scope. `sample_affine` follows the same
  `rng: np.random.Generator` contract as `sample_flip`/`sample_crop`; each of `angle_range`/
  `translation_x_range`/`translation_y_range`/`scale_range` is an independently sampled `(low, high)`
  tuple (`scale_range` additionally requires a positive `low`). The transform is built as rotation +
  isotropic scale around the image center (via `cv2.getRotationMatrix2D`, using the same center
  convention as `improcv.transforms.rotate`), then translated in the destination coordinate system --
  a fixed composition order, not an implementation detail, since translation does not commute with
  rotation/scaling in general. `AffineParameters.matrix` (a new, read-only, finite `float64` `(2, 3)`
  array) is the sole source of truth for replay; `angle`/`translation`/`scale` are sampling metadata
  for debugging/logging/`repr` only and are never used to reconstruct or cross-check the matrix.
  `apply_affine` requires the image's (and mask's) spatial size to match `AffineParameters.source_size`
  exactly, reuses `improcv.transforms.warp_affine` unmodified for both the image and the (always
  `INTER_NEAREST`, `BORDER_CONSTANT`) mask, and maps an unexpected `cv2.error` to `RuntimeError` (with
  the original error preserved as `__cause__`) since `warp_affine` itself does not. The mask dtype
  contract is unchanged from flip/crop (`uint8`/`uint16`/`int16`; `int32` support remains a possible,
  compatible future extension, not added here). No new dependency.
- `improcv.augmentation`: affine shear sampling and replay using sequential x-then-y,
  area-preserving shear coefficients -- `sample_affine` gains `shear_x_range`/`shear_y_range`
  (each an independently sampled `(low, high)` tuple, same contract as the other `*_range`
  parameters, no positivity restriction and no `abs(shear)` limit), and `AffineParameters` gains a
  keyword-only `shear: tuple[float, float] = (0.0, 0.0)` field -- the pre-existing five-positional-
  argument construction, `__match_args__` (unchanged, still the original five field names), and
  positional pattern matching all keep working exactly as before. Shear is parameterized as a raw,
  dimensionless coefficient (not degrees -- a degrees-based `tan()` conversion has a practically
  unguarded singularity at ±90° that a raw coefficient doesn't), applied as `Shy(shear_y) @
  Shx(shear_x)` (`[[1, shear_x], [shear_y, 1 + shear_x*shear_y]]`): determinant `1` mathematically
  for any finite coefficients (a statement about the exact real-number parameterization, not a
  promise of infinite `float64` precision -- see the `Fixed` entry below), unlike the naive-looking
  `[[1, shear_x], [shear_y, 1]]` (determinant `1 - shear_x*shear_y`), which `improcv` does not use.
  Composition order is shear x, then shear y, then rotation + isotropic scale around the same
  center as before, then translation -- shear does not commute with rotation, so this is a fixed,
  documented part of the contract. When `shear_x_range`/`shear_y_range` are left at their `(0.0,
  0.0)` default, the matrix is built via the pre-shear code path with no extra matrix
  multiplication (bit-for-bit identical to before this change, not just numerically close) and no
  extra `rng.uniform` call is made, so existing call sites keep sampling `angle`/`translation`/
  `scale` from exactly the same `rng` sequence, call after call, as they did before shear existed.
  `apply_affine`'s public signature, the mask dtype contract, and `transforms.py` are all
  unchanged. No new dependency.
- New `improcv.discovery` module, Phase 5 slice (deterministic extension-based dataset image
  discovery): `discover_images`, finding candidate image files under a directory by filename
  extension only -- a file's content is never opened or decoded, so an empty, corrupted, or
  non-image file with a matching extension is still discovered. `recursive` (default `True`)
  controls whether subdirectories are descended into; `extensions` (default: seven common raster
  extensions -- `.jpg`/`.jpeg`/`.png`/`.bmp`/`.tif`/`.tiff`/`.webp` -- covering five widely-used
  formats, not derived from what the local OpenCV build happens to support) accepts any
  `Collection[str]`, case-insensitively, with or without a leading dot, including multi-part
  extensions like `.nii.gz`; a bare `str`/`bytes`/`bytearray`, a `Mapping`, or a generator/iterator
  passed where a collection of extensions is expected are all rejected with `TypeError` rather than
  silently misinterpreted. `include_hidden` (default `False`) skips a descendant file or directory
  whose name starts with `.` (and everything under a skipped directory) -- this never applies to
  `root` itself, so an explicitly given dot-prefixed `root` is still searched. A symlink or Windows
  reparse point (including a junction) found while traversing `root`'s contents is always skipped,
  along with everything under it -- there is no `follow_symlinks` option -- but `root` itself may be
  a symlink to a directory. The result is a materialized, globally-sorted `tuple[Path, ...]` (sorted
  by each path's POSIX-style form relative to `root`, independent of traversal order or the
  platform's path separator), anchored under `root` exactly as given (never resolved or made
  absolute). Filesystem errors are fail-fast (a missing/inaccessible `root`, or a permission error
  encountered anywhere during traversal, propagates immediately as a native `OSError` subclass,
  never silently skipped or wrapped in a new exception type -- there is no OpenCV/NumPy involvement
  in this module at all). No image/mask pairing, class inference from directory names, dataset
  splits, manifests, or loading/decoding in this slice, and no new dependency.
- `improcv.evaluation`, Phase 5 slice: binary one-vs-rest ROC and precision-recall curves plus
  ROC AUC -- `roc_curve`, `precision_recall_curve`, `roc_auc_score`, and their `RocCurve`/
  `PrecisionRecallCurve` result dataclasses. Binary, one-vs-rest: `positive_label` is always
  required and explicit -- a sample is positive iff its `y_true` label equals `positive_label`;
  every other observed label is negative, regardless of how many distinct negative labels occur,
  and there is no automatic inference of which label is positive. `y_score` is a ranking score,
  not a predicted label or a probability -- it does not need to lie in `[0, 1]`, larger means more
  confident positive, and it is normalized to an independent `float64` array (Python/NumPy `int`/
  `float`, `float16`/`float32`/`float64` ndarrays, and any integer ndarray dtype are accepted;
  `bool`, complex, object, and wider-than-`float64` floating dtypes are rejected, as are NaN/Inf,
  and an integer value not exactly representable as `float64`, e.g. `2**53 + 1`). A threshold
  classifies a sample positive iff `score >= threshold`; every sample sharing the same score is
  aggregated into one threshold before FPR/TPR/precision/recall is computed there, so permuting
  the order of tied samples (or of the whole input) never changes the result -- verified directly
  that grouping by distinct score, not sort-algorithm stability, is what makes this true. Both
  curves start at the sentinel threshold `+inf` (ROC at `(FPR, TPR) = (0, 0)`, precision-recall at
  `(precision, recall) = (1, 0)` with no corresponding real threshold); `thresholds`/the two
  curve-value arrays always share length `K + 1` for `K` distinct observed scores, with
  `thresholds[1:]` strictly decreasing. `roc_curve`/`roc_auc_score` require at least one positive
  and one negative sample, raising `ValueError` for a degenerate `y_true` instead of
  `sklearn.metrics`'s `UndefinedMetricWarning` plus a degenerate result; `precision_recall_curve`
  only requires at least one positive sample -- a `y_true` with no negative sample is legal
  (`precision == 1.0` at every real threshold). `precision_recall_curve`'s `recall` is returned in
  ascending order paired with descending `thresholds`, matching `roc_curve`'s convention -- a
  deliberate departure from `sklearn.metrics.precision_recall_curve`'s descending `recall`.
  `roc_auc_score` integrates the ROC curve with the trapezoidal rule (verified directly equivalent
  to the probability that a random positive sample outranks a random negative one, with a tied
  pair counted as one-half) using plain array arithmetic -- never `np.trapz`/`np.trapezoid`, since
  neither name exists across this project's full supported NumPy range (`np.trapz` is removed in
  current NumPy; `np.trapezoid` does not exist on this project's NumPy floor). All three functions
  return new, independent, read-only `float64` arrays, never a view of `y_true`/`y_score`. No
  generic `auc(x, y)` helper, no average precision, no trapezoidal PR AUC, no multiclass score
  matrix/averaging, no sample weights, and no plotting in this slice.
- `improcv.evaluation`, Phase 5 slice: binary one-vs-rest non-interpolated average precision --
  `average_precision_score`, sharing the exact grouped-threshold tie contract, input contract, and
  error contract as `precision_recall_curve` (including that a `y_true` with no negative sample is
  legal). This is classification ranking average precision, not object-detection AP or mAP (no
  bounding boxes, no IoU matching, no per-class averaging). Defined as the non-interpolated
  weighted mean of precision using each recall increment as its weight -- `sum((recall[i] -
  recall[i - 1]) * precision[i] for i in 1..K)`, with `precision[i]` always taken from the right
  end of each recall increment -- computed directly from the private grouped-threshold core
  (without constructing a public `PrecisionRecallCurve`) using the exact same `float64` arithmetic
  order as `precision_recall_curve`'s public arrays would produce (`sum(diff(recall) *
  precision[1:])`), not an algebraically-equivalent but differently-rounded single-division-after-
  summing form -- verified directly that the two orders can disagree in the last bit on a
  constructed input. `average_precision_score` is not the trapezoidal area under the PR curve
  (a distinct quantity, not added in this slice): depending on the curve's shape and its ties, that
  trapezoidal area can be larger or smaller than average precision, never consistently one or the
  other -- verified with two concrete examples in opposite directions. A `y_true` with no negative
  sample returns exactly `1.0`; constant scores return exactly the positive prevalence. No
  trapezoidal PR AUC, no generic `auc(x, y)` helper, no interpolated/VOC/COCO-style AP, no
  object-detection AP/mAP, no new public result type, and no new dependency in this slice.
- `improcv.evaluation`, Phase 5 slice: a general-purpose trapezoidal area-under-curve helper,
  `auc(x, y)`, with no ranking semantics of its own (no `positive_label`, no tie-aggregation, no
  positive/negative samples) -- `x` must be non-decreasing or non-increasing throughout (duplicate
  `x` values are legal either way and contribute a zero-width segment; constant `x` gives exactly
  `0.0`); a non-increasing `x` gives the same positive geometric area a non-decreasing order of the
  same points would, not a signed integral. `y` may be negative and so may the result -- unlike
  `roc_auc_score`/`average_precision_score`, whose domain is always `[0, 1]` by construction, `auc`
  has no such bound. Ordinary calls -- `y` entirely non-negative or entirely non-positive, with no
  intermediate overflow/underflow -- use a fast, canonical `float64` summation (the path
  `roc_auc_score`'s always-non-negative TPR and `auc(curve.recall, curve.precision)`'s
  always-non-negative precision both take); `auc` falls back to computing the exact trapezoidal sum
  as a rational number (`fractions.Fraction`, standard library, used only on these rare paths) over
  the already-normalized `float64` values, converting only the final total back to `float`, when an
  intermediate segment width/height-sum/product would overflow `float64`, when a segment's own
  contribution would underflow in a way that could lose it before it has a chance to be summed with
  its neighbors, or when `y` contains both a positive and a negative value (opposite-signed
  contributions can cancel in the final sum in a way no NumPy overflow/underflow/invalid signal
  would ever catch) -- verified directly for a height-sum overflow with a finite result, a width
  overflow with a finite result, constant `x` with an extreme finite `y`, a width overflow combined
  with a tiny `y` giving a finite nonzero result, cancellation between huge intermediate
  contributions (summing exactly to `0.0` and to `1.0` in two constructed examples), a subnormal
  residual accumulated from several underflowing segments (summing exactly to `5e-324`, the
  smallest positive subnormal `float64`), a mixed-sign cancellation residual (summing exactly to
  `1e-20` where the fast path would silently give `0.0`), and a segment whose exact contribution
  genuinely rounds to `0.0` (legal, not an error), all without emitting a NumPy warning regardless
  of the caller's own `np.seterr`/`np.errstate` configuration; only an input whose true, exact area
  is not representable as a finite `float64` raises `ValueError`, never silently `inf`/`-inf`/`NaN`.
  `auc` computes the trapezoidal area under
  the precision-recall curve when called as
  `auc(curve.recall, curve.precision)` -- a distinct quantity from `average_precision_score` (see
  above), and the complete, supported way to obtain it: there is no separate score-level function
  for it, to avoid a symbol that could be confused with `average_precision_score`. `roc_auc_score`
  now shares its trapezoidal arithmetic with `auc` through a common private primitive (verified
  bit-identical to its previous result for representative curves, ties, constant scores, extreme
  accepted scores, and several deterministic random rankings) -- its own public behavior,
  `[0, 1]`-bounded postconditions, and input contract are unchanged. No new public result type, no
  `precision_recall_auc_score` or other score-level alias, no SciPy, and no new dependency in this
  slice.
- `improcv.evaluation`, Phase 5 slice: an optional, keyword-only `sample_weight` for `roc_curve`,
  `precision_recall_curve`, `roc_auc_score`, and `average_precision_score` -- `None` (the default)
  preserves each function's existing unweighted result, dtype, and error contract bit for bit;
  given explicitly, `sample_weight` must be the same length as `y_true`/`y_score`, hold the same
  accepted numeric types as `y_score`, and be non-negative with at least one positive value
  (negative weights are rejected, unlike `sklearn`'s ranking curves, which accept them -- verified
  directly that a negative weight can make `sklearn`'s own `true_positive_rate` non-monotonic in
  the threshold, the exact invariant this project's own postconditions already enforce). A sample
  with `sample_weight == 0.0` is removed from the effective set before thresholds are built (a
  score existing only at zero weight never produces a threshold, matching `scikit-learn`'s own
  documented "filters out zero-weighted samples" behavior, verified directly against its source);
  a class present only among zero-weight samples raises the same kind of `ValueError` as an
  unweighted `y_true` missing that class entirely, with a weighted-specific message.
  `TP(t)`/`FP(t)` become the sum of effective weights at or above each threshold, computed via a
  new private `_WeightedRankingCore` that groups each distinct score's positive/negative weights
  and sums each group exactly once with `math.fsum` over a canonically sorted sequence -- not a
  plain per-sample `np.cumsum`, which would make the existing "permuting samples within a tie never
  changes the result" contract depend on summation order (verified directly:
  `np.cumsum([1e16, 1.0, 1.0])[-1] != np.cumsum([1.0, 1.0, 1e16])[-1]`, but
  `math.fsum(sorted(...))` gives the same result regardless of order). The resulting per-group
  totals are then turned into cumulative arrays with a single `np.cumsum(..., dtype=np.float64)`
  under `np.errstate(over="raise", invalid="raise")`; an extreme `sample_weight` dynamic range that
  would make a whole positive-weight group fail to strictly increase the cumulative sum (verified
  directly: `np.cumsum([1e16, 1.0])` gives `[1e16, 1e16]`, silently absorbing the second group)
  raises `ValueError` rather than silently returning a curve that drops that group's contribution.
  Precision (`tp / (tp + fp)`) uses a new private helper that matches plain `float64` division bit
  for bit whenever it would not overflow, and only routes the individual overflowing entries
  through a scale-based stable formula instead (verified directly that a single global fallback the
  moment any entry overflowed would change every entry's bit pattern, not only the overflowing
  one's, silently breaking the `sample_weight=[1.0, ...]` all-ones bit-identity guarantee below) --
  so `precision`/`average_precision_score` stay correct (`0.5`, not a silently-underflowed `0.0`)
  even when both the effective positive and negative weight at a threshold are individually near
  `float64`'s max. `roc_auc_score`/`average_precision_score` do not call the public
  `roc_curve`/`precision_recall_curve` internally for the weighted path either (matching the
  existing unweighted design): both build their arrays from the same private weighted core, so
  `roc_auc_score(..., sample_weight=w)` always equals
  `auc(*that call's equivalent roc_curve(..., sample_weight=w) rate arrays*)` bit for bit.
  `sample_weight=None`, all-ones, and small-integer-weight-vs-physical-replication are all verified
  bit-identical to their unweighted counterparts across thresholds, rates, precision/recall, ROC
  AUC, and average precision; whole-input and within-tie permutation invariance stay bit-exact;
  scaling every weight by a positive constant stays only approximately invariant (verified directly
  that `false_positive_rate` is bit-identical under scaling but `true_positive_rate` is not, since
  the two are computed from independently-rounded cumulative sums). `average_precision_score`'s
  existing all-positive shortcut (returning exactly `1.0` directly, not via summing curve
  increments) still applies for zero effective negative weight -- verified directly that, for
  weighted input, naively summing `precision_recall_curve`'s own recall increments can round to
  `0.9999999999999999` even when the true result is exactly `1.0`. `sample_weight` for
  `confusion_matrix`/`classification_metrics`/`classification_metrics_from_confusion_matrix` is
  added separately, below in this same `[Unreleased]` series; `auc(x, y)` has no `sample_weight`
  (it operates on curve points, not per-observation weights). No new public result type, no
  multiclass score matrix, and no new dependency in this slice.
- `improcv.evaluation`, Phase 5 slice: an optional, keyword-only `sample_weight` for
  `confusion_matrix` and `classification_metrics`; `classification_metrics_from_confusion_matrix`
  gains no new parameter but now also accepts a `float64` confusion matrix.
  `ConfusionMatrixResult.matrix`/`ClassificationMetrics.support` are exactly `int64` when
  `sample_weight` was not given, exactly `float64` whenever it was -- regardless of the weights'
  own dtype or values (even `sample_weight=[1, 1, ...]` or an all-integer-valued weight sequence
  gives a `float64` matrix, never `int64`), mirroring the same dtype rule already established for
  ranking `sample_weight`. Equality (`==`) on both result types compares arrays purely by value,
  ignoring dtype -- an `int64` matrix/support and a `float64` one holding the same numbers compare
  equal, since dtype reflects how a result was computed, not part of its semantic value. A negative
  weight is rejected (unlike `sklearn.metrics.confusion_matrix`, which accepts one and produces a
  matrix with a negative cell -- verified directly). A sample with `sample_weight == 0.0`
  contributes to no cell and is removed from the effective set before class inference: with
  `labels=None`, a class present only among zero-weight samples never appears as a row/column (an
  explicit `labels` including that class still gives it a well-defined, all-zero row/column, since
  explicit `labels` validation always runs against every raw `y_true`/`y_pred` value regardless of
  weight -- an invalid label is never forgiven merely because its weight is zero). An all-zero
  `sample_weight` is legal for `confusion_matrix` only together with an explicit `labels`
  (mirroring its existing "empty input plus explicit labels" contract, now also covering
  `sample_weight=[]`); `classification_metrics` always requires at least one positive weight, since
  it has no equivalent well-defined empty/all-zero result. Matrix cells are built by grouping every
  same-cell sample and summing that group's weights exactly once via `math.fsum` over the weights
  sorted first -- not `np.bincount(..., weights=...)` or a plain running sum, both verified
  directly to be order-dependent for extreme weight ratios (three samples landing in the same cell
  with weights `[1e16, 1.0, 1.0]` vs `[1.0, 1.0, 1e16]` give a different final cell value either
  way, but the same, correctly-rounded `1.0000000000000002e16` through canonical grouped summation
  regardless of order) -- and scikit-learn's own `confusion_matrix` was verified directly to have
  this same order-dependence, unprotected. An `OverflowError` from a single cell's weights summing
  past `float64`'s range is mapped to `ValueError`; no total/row/column sum is required to be
  representable at the `confusion_matrix` level itself (a matrix with several individually-finite,
  huge cells is still a useful result on its own) -- only `classification_metrics`/
  `classification_metrics_from_confusion_matrix` reject a non-representable row/column/total sum,
  since only they need to reduce the matrix further, computed via the same canonical `math.fsum`-
  based summation (row/column/diagonal/total), never a bare `matrix.sum(axis=...)`.
  `classification_metrics_from_confusion_matrix` accepts exactly `int64` or exactly `float64`
  (never `float16`/`float32`/`bool`/any other integer dtype, never silently cast); a `float64`
  matrix is always treated as weighted, even with only whole-number values -- dtype, not content,
  decides. Precision/recall for a `float64` matrix divide directly by the already-safe column/row
  sum (`TP_i <= column_sum_i`/`row_sum_i` always, so this never overflows); F1
  (`2 * TP_i / (row_sum_i + column_sum_i)`) uses a new stable helper that matches plain division bit
  for bit except at the individual entries that would actually overflow (verified directly:
  `TP_i = row_sum_i = column_sum_i = np.finfo(np.float64).max` gives exactly `1.0`, not a silently
  overflowed `0.0`); `average="weighted"`'s own denominator is a further, independent canonical sum
  of the returned `support` array specifically (not the flat matrix total), since `support` is the
  documented, public set of per-class weights for that average and can legitimately differ from the
  flat total by a rounding residual. Legal underflow anywhere in this arithmetic (precision, recall,
  F1, accuracy, `"micro"`/`"macro"`/`"weighted"` reductions) never raises or warns regardless of the
  caller's own `np.seterr`/`np.errstate` configuration, mirroring the same fix already applied to
  the ranking core. `sample_weight=None`, all-ones, and small-integer-weight-vs-physical-
  replication are all verified bit-identical to their unweighted counterparts; whole-input and
  within-cell permutation invariance stay bit-exact; `classification_metrics(..., sample_weight=w)`
  is bit-exact with `classification_metrics_from_confusion_matrix(confusion_matrix(...,
  sample_weight=w))`; scaling every weight by a positive constant stays only approximately
  invariant. No `sample_weight` for `auc(x, y)`, no multiclass/multilabel support, no new public
  types or functions, no new batch-aggregation helper, and no new dependency in this slice.

### Fixed
- `improcv.evaluation`: `classification_metrics`/`classification_metrics_from_confusion_matrix`'s
  `average="macro"` and `average="weighted"` aggregates could differ by one or a few ULP after
  changing only the presentation order of an explicit `labels` sequence (with the confusion
  matrix's rows/columns permuted identically) -- the *set* of per-class values was unchanged, only
  their order, so all scalar aggregates must be bit-identical regardless of it. The cause:
  `average="macro"` used `np.mean(per_class_values)`, and `average="weighted"` used
  `np.sum(values * weights) / total_weight` -- both computed a `float64` sum whose final bit
  pattern depends on the order of its inputs, and that order was directly `labels`' own order.
  Both now go through a canonical, order-independent reduction instead: a new private
  `_canonical_mean` (used identically by the `int64` and `float64` branches) and a rewritten
  `_weighted_average`, each converting to plain Python `float`s, establishing a canonical order via
  `sorted()`, and summing via `math.fsum` before dividing -- `_weighted_average` sorts the actual
  `value * weight` products (the real terms being summed), not `values`/`weights` separately, which
  would not establish a canonical order for the products themselves. Per-class `precision`/
  `recall`/`f1`/`support`/`accuracy`, `average="micro"`, the confusion matrix itself, and every
  `sample_weight`/dtype/empty/zero-weight contract from prior slices are unchanged; only
  `average="macro"`/`"weighted"`'s own scalar results are affected, and only by up to a few ULP for
  ordinary data. `zero_division="nan"` still makes the whole aggregate `NaN` when any class is
  undefined, including a `weighted` aggregate where that class's own support is `0.0` (`NaN` is
  now checked explicitly before building any product, rather than relying on `NaN * 0.0` already
  being `NaN` under IEEE 754, which remains true but is no longer the mechanism this function's
  contract depends on). Legal underflow in either reducer still never depends on the caller's own
  `np.seterr`/`np.errstate` configuration.
- `improcv.evaluation`: the weighted ROC/precision-recall/ROC-AUC/average-precision path
  (`sample_weight`, added earlier in this same `[Unreleased]` series) could leak a raw
  `FloatingPointError` or `RuntimeWarning` for legal, extreme-but-finite weights whenever the
  caller had previously called `np.seterr(under="raise")`/`np.seterr(under="warn")` -- a public
  function's result must not depend on the caller's own global NumPy error-state configuration.
  Three call sites divided or multiplied values that can legitimately underflow to a
  correctly-rounded `0.0` for extreme weight ratios: `_compute_weighted_roc_rates`'s
  `false_positive_rate`/`true_positive_rate` normalization and
  `_compute_weighted_precision_recall_arrays`'s `recall` normalization (e.g. a weight of
  `np.nextafter(0.0, 1.0)` next to one of
  `np.finfo(np.float64).max`), `_precision_ratio`/`_stable_precision_ratio`'s ordinary and
  scale-based divisions (e.g. `sample_weight=[1.0, np.finfo(np.float64).max]`), and
  `average_precision_score`'s weighted `np.diff(recall) * precision[1:]` product/sum. Each now
  runs under a local `np.errstate(under="ignore")` around only that operation, rather than
  changing the caller's global `np.seterr` state -- `over`/`invalid`/`divide` are deliberately
  left at the caller's own sensitivity, since none of those should occur at these call sites by
  construction, and are still not silenced. This is unrelated to the existing dynamic-range
  `ValueError` for an absorbed positive-weight group or a genuine cumulative-weight overflow, both
  of which remain unchanged, explicitly-detected errors -- only ordinary, correctly-rounded
  underflow within a single ratio/product no longer depends on the caller's `np.seterr`
  configuration. No exact/`Fraction`-based fallback was added for this: the weighted core's
  cumulative sums, rates, precision, and average precision remain defined by plain, deterministic
  `float64` arithmetic, and a ratio or product that legitimately rounds to `0.0` is an accepted
  result of that contract, not an error.
- `improcv.evaluation`: `auc`'s fast `float64` path could silently lose a representable residual
  through catastrophic cancellation when `y` contained both a positive and a negative value --
  `x=[0.0, 1.0, 2.0], y=[1.0, 1e-20, -1.0]` has exact trapezoidal area `1e-20`, but plain `float64`
  rounds `1.0 + 1e-20` to `1.0` and `1e-20 - 1.0` to `-1.0`, so the two segments' contributions of
  `0.5` and `-0.5` cancelled to exactly `0.0` -- with no overflow, underflow, or invalid operation
  for `np.errstate` to catch, since every individual operation stayed finite and normal throughout.
  `auc` now checks whether `y` contains both a positive and a negative value (a global check across
  all of `y`, not just between adjacent points, since even non-adjacent contributions can cancel in
  the final sum) and, if so, routes directly to the existing exact `fractions.Fraction` fallback
  without attempting the fast path at all. Curves with entirely non-negative or entirely
  non-positive `y` -- including `roc_auc_score`'s TPR and `auc(curve.recall, curve.precision)`'s
  precision -- are unaffected and keep using the fast path, since same-signed contributions can
  never cancel against each other.
- `improcv.evaluation`: `auc`'s fast `float64` path (`_trapezoidal_area_float64`) previously only
  guarded against overflow and invalid (NaN-producing) operations, not underflow -- so an
  individual segment's own contribution could round to `0.0` under plain `float64` arithmetic
  before it ever had a chance to be summed with its neighbors, even when several such segments'
  *exact* contributions summed to a representable positive subnormal `float64`: `auc` could
  therefore silently return `0.0` for an input whose true trapezoidal area is a nonzero subnormal
  value. The fast path now also raises on underflow (`np.errstate(..., under="raise")`), routing
  such an input to the existing exact `fractions.Fraction` fallback (added in the previous fix,
  below) the same way an overflow already does -- that fallback's exact rational summation
  correctly distinguishes a genuinely-zero result from several separately-underflowing
  contributions whose true sum is a representable positive subnormal `float64`, without emitting
  a NumPy warning regardless of the caller's own `np.seterr`/`np.errstate` configuration. The
  ordinary fast path for non-underflowing, non-overflowing data (and therefore `roc_auc_score`'s
  own bit-identical result on its own `[0, 1]`-bounded domain, which never reaches either
  fallback trigger) is unchanged.
- `improcv.evaluation`: `auc`'s overflow fallback (introduced alongside `auc` itself, in this same
  `[Unreleased]` series) previously rescaled every value by a single `0.5`/`2.0` factor before
  recombining, which is not actually exact at the extremes: (1) it wrongly raised `ValueError` for
  inputs whose true trapezoidal area is a finite, representable `float64` reached only through
  cancellation between huge intermediate contributions (e.g. segments individually far larger than
  `float64`'s own range but summing to exactly `0.0` or to exactly `1.0`), and (2) it could silently
  underflow a genuine subnormal residual to `0.0` (halving a value already at the edge of the
  subnormal range loses it entirely). The fallback now recomputes the exact trapezoidal sum as a
  rational number (`fractions.Fraction`, standard library) over the already-normalized `float64`
  values, converting just the final total back to `float` -- since amended below in this same
  `[Unreleased]` series to also cover underflow and mixed-sign cancellation, not overflow alone.
  `ValueError` is now raised only when that exact total's `float()` conversion itself overflows
  (i.e. the true area genuinely is not representable as a finite `float64`), never merely because
  the fast, unscaled `float64` attempt happened to overflow. The ordinary, non-overflowing fast path
  (and therefore `roc_auc_score`'s own bit-identical result on its own `[0, 1]`-bounded domain,
  which never reaches this fallback) is unchanged.
- `improcv.discovery`: `discover_images` now classifies descendants using a fresh path-based
  non-following stat instead of cached `DirEntry` metadata, preserving its fail-fast contract when
  an entry disappears before inspection. `os.DirEntry.stat()` can return metadata captured during
  directory enumeration itself, without a fresh system call, on some platforms (notably Windows) --
  which could let a descendant that had already been deleted or replaced before classification pass
  through the check silently instead of raising the documented `FileNotFoundError`. Classification
  now always calls `os.stat(entry.path, follow_symlinks=False)`; hidden entries (skipped by name
  before any filesystem inspection) are unaffected and still incur no `stat` call at all.
- `improcv.augmentation`: reject sequential affine-shear coefficients when `float64` rounding
  removes the unit determinant term, preventing a mathematically invertible shear from being stored
  as a singular matrix. `sample_affine`'s sequential shear matrix (`[[1, shear_x], [shear_y, 1 +
  shear_x*shear_y]]`) has determinant `1` for any finite `shear_x`/`shear_y` as an exact real-number
  statement, but `1.0 + shear_x*shear_y` silently rounds down to exactly `shear_x*shear_y` in
  `float64` once `|shear_x*shear_y|` is roughly `2**52` or larger (e.g. `shear_x=shear_y=1e8`) --
  the previous code accepted this (the resulting matrix is still finite, so the existing
  finite-matrix check didn't catch it), silently storing a singular matrix for shear coefficients
  that were, mathematically, perfectly invertible. `sample_affine` now checks this specific
  condition directly and raises `ValueError` before the matrix is built, in addition to (not instead
  of) the pre-existing finite-matrix check for genuine overflow to `inf`/`NaN`. No arbitrary
  `abs(shear)` limit or condition-number threshold was introduced -- a large, but still
  representable and invertible, shear coefficient remains accepted even though it can produce a
  very poorly conditioned matrix.
- `improcv.augmentation`: `apply_affine` now rejects `WARP_INVERSE_MAP` and other non-interpolation
  flag bits passed through the `interpolation` parameter, preventing a saved affine matrix from
  being silently interpreted in the opposite direction.
- `improcv.evaluation`: F1 was computed as `2*precision*recall/(precision+recall)`, which loses the
  distinction between "precision and recall are both correctly `0`" (real, nonzero `FP`/`FN`, `TP =
  0` -- F1 is well-defined as `0`) and "precision and recall are both undefined" (class completely
  absent from `y_true`/`y_pred` -- F1 should use `zero_division`); the affected public result was
  `zero_division=1.0` producing `F1 = 1.0`, or `zero_division="nan"` producing `F1 = NaN`, for a
  class that was simply misclassified, not one that was actually undefined. F1 is now computed
  directly from counts (`2*TP / (2*TP + FP + FN)`), using `zero_division` only when
  `2*TP + FP + FN == 0`; `zero_division=0.0` (the default) was unaffected, since `0` happened to be
  the correct fallback either way. Also fixed: `classification_metrics_from_confusion_matrix` could
  silently wrap a very large (hand-constructed) confusion matrix's total count past `int64`'s range,
  producing a negative `support`; the total is now computed exactly (falling back to arbitrary-
  precision Python `int` arithmetic when a native `int64` sum could overflow) and raises `ValueError`
  if it exceeds what `int64` can represent, rather than silently wrapping around.
- `improcv.evaluation`: binary ranking score normalization now maps oversized Python integers to
  the documented `ValueError`, rejects wider-than-`float64` NumPy floating scalars consistently
  with `ndarray` inputs, and canonicalizes signed zero so tied zero thresholds are
  permutation-deterministic.

## [0.2.0a1] - 2026-07-26

Phase 4 release: quality metrics, perceptual hashing, photo and creative
operations, non-local-means denoising, HDR imaging, and panorama/scan
stitching.

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
- `improcv.quality.gmsd`, the second Phase 4 quality-metrics slice: Gradient Magnitude Similarity
  Deviation (Xue, Zhang, Mou, Bovik, IEEE TIP 2014). Matches the reference MATLAB implementation the
  authors shared (`GMSD.m`) rather than the paper's own rounded prose -- notably, the paper's text
  states `c=0.0026` for images normalized to `[0, 1]`, but the authors' own code uses `T=170` directly
  on `0-255` data; `170 / 255**2 == 0.0026144...`, measurably different from `0.0026` (cross-checked
  to produce score deltas of `~1e-5` to `~1e-4`), and `T=170` is what this implementation uses, since
  it is the variant reproduced by the authors' reference implementation. Pooling uses sample standard
  deviation (`ddof=1`, matching MATLAB's `std2`/`std` default),
  not the paper's own written population-based equation. Unlike `ssim`/`psnr`, `gmsd` is grayscale-only
  (2D, or 3D with exactly 1 channel) -- multi-channel input is rejected with a message pointing to
  `improcv.ensure_gray`, since GMSD has no reference definition for color and this project never hides
  an automatic color conversion. Lower scores mean higher quality; `0.0` exactly for identical images.
  Images that downsample to fewer than 2 gradient-magnitude-similarity-map samples (`1x1`, `1x2`,
  `2x1`, `2x2`) raise `ValueError` instead of matching the reference's own behavior of returning `0.0`
  for such degenerate inputs regardless of whether the two images are identical or completely
  different -- a deliberate, safer departure from `GMSD.m`, documented as such. Two different
  *constant* images can give a non-zero score: the reference's zero-padded border convolution makes
  the gradient at the image edge artificially non-zero, which is expected reference behavior, not a
  bug in this port. Cross-checked numerically against the exact, unmodified `GMSD.m` file (run via GNU
  Octave, an isolated, throwaway environment) across identical/constant/noise/edge images and
  even/odd/mixed/small spatial sizes -- agreement at floating-point precision (~1e-14 to exact) once
  two non-obvious implementation details were matched: the `2x2` averaging filter's anchor
  (MATLAB's `conv2(..., 'same')` anchors an even-sized kernel at its top-left corner, not
  `cv2.filter2D`'s default) and zero-padding border (`cv2.BORDER_CONSTANT`, not `cv2.filter2D`'s own
  default border mode) for every filter call. Octave/`GMSD.m` were used only for one-off verification,
  never added as a project dependency.
- New `improcv.hashing` module, Phase 4 slice 2 (perceptual hashing — core): `average_hash`, `phash`,
  `PerceptualHash`, `PerceptualHashAlgorithm`. Both functions reproduce `cv2.img_hash.AverageHash`'s
  and `cv2.img_hash.PHash`'s own bit decisions exactly for `uint8` input at `hash_size=8` (verified
  bit-for-bit against `opencv-contrib-python` 4.13.0.90 and 5.0.0.93 in an isolated, throwaway
  environment across hundreds of random grayscale/BGR/BGRA images) -- but generalize `hash_size` to
  any value in `[2, 256]`, which `cv2.img_hash` itself hardcodes to `8`. `average_hash` resizes to
  `hash_size x hash_size` (`cv2.INTER_LINEAR_EXACT`), converts BGR/BGRA to grayscale **after**
  resizing (not before -- verified that the opposite order, e.g. via `improcv.ensure_gray` first,
  does not reproduce the same bits for color input, since `uint8` rounding at each stage does not
  commute), and thresholds each pixel against the mean rounded to the nearest integer with
  round-half-to-even (matching `cv::cvRound`). `phash` resizes to `(hash_size*4) x (hash_size*4)`,
  converts to `float32`, runs `cv2.dct`, takes the top-left `hash_size x hash_size` block, zeroes its
  DC term, and thresholds against that block's mean -- computed with `cv2.mean`, not `np.mean`
  (verified to disagree with it in the last few bits for the same input), cast to `float32` (matching
  the reference implementation's own `cv::mean`/`float` storage) -- this is one of several genuinely different,
  non-interchangeable "pHash" variants in circulation (distinct from both `ImageHash.phash`'s
  median-of-the-full-block and the original hackerfactor.com blog's mean-of-63-AC-terms recipes);
  this implementation specifically reproduces `cv2.img_hash.PHash`. Only `uint8` is supported in this
  first slice (grayscale `(H, W)`/`(H, W, 1)`, BGR `(H, W, 3)`, or BGRA `(H, W, 4)`; 2-channel and
  non-`uint8` input rejected) -- `uint16`/`float32`/`float64` support is deferred to a later slice.
  No `data_range` parameter: this slice supports only `uint8`, so the input's value range and
  quantization are already unambiguous, and the pipeline deliberately reproduces OpenCV's own
  resize/threshold operations rather than a scale-invariant formula -- exact invariance to brightness/
  contrast transforms is **not** guaranteed (verified directly: a plain `2x + 10` transform with no
  clipping changes 1 bit out of 64 for both `average_hash` and `phash` on real test images, from
  `uint8` rounding inside `cv2.resize`/`cv2.dct`, not from the algorithms' own definitions).
  `PerceptualHash` is an immutable (frozen, slotted) dataclass wrapping the hash's algorithm,
  `hash_size`, and bit value; equality/hashability follow from the dataclass, `distance()` computes
  the Hamming distance (raising `ValueError` if the two hashes have a different `algorithm` or
  `hash_size` -- same bit length alone
  is not sufficient, since `average_hash` and `phash` can produce same-length but non-comparable
  hashes), `str()` gives a fixed-width, lowercase, zero-padded hex string, and `from_hex` parses one
  back given an explicit `algorithm`/`hash_size` (a hex string alone cannot reveal which algorithm
  produced it). improcv's own bit/hex serialization (row-major, first bit is most significant)
  coincides with `ImageHash`'s own row-major MSB-first hex convention for the same bit grid, but the
  algorithms, metadata, object model, validation, and comparison compatibility are different --
  `ImageHash.average_hash`/`phash` values are usually different from improcv's, since the underlying
  algorithms differ in resize, thresholding, and (for pHash) the statistic itself; matching hex
  ordering does not imply matching algorithm output. Serialization is **not** the same as
  `cv2.img_hash`'s internal packed-byte layout (LSB-first per byte, one image row per byte) -- only the
  underlying bit *decisions* are verified to match, not the serialized bytes; raw byte import/export
  is not offered in this slice. `ImageHash`, the popular third-party library, was deliberately not
  reused as this type's name, since it has an incompatible representation and
  semantics under the same name. No new runtime dependency -- `opencv-contrib-python` and `ImageHash`
  were used only for one-off, throwaway-environment verification.
- New `improcv.photo` module, Phase 4 slice 3 (photo/creative — single-image stylization):
  `pencil_sketch`, `stylize`, `detail_enhance`, `PencilSketchResult`. Thin wrappers around
  `cv2.pencilSketch`/`cv2.stylization`/`cv2.detailEnhance` (all in base `opencv-python`, no contrib)
  adding validation those functions don't perform themselves -- verified directly that none of the
  three has any dtype/channel-count assertion at all. Require exactly `uint8`, 3-channel BGR input;
  grayscale (`(H, W)`/`(H, W, 1)`), 2-channel, and BGRA (`(H, W, 4)`) are all rejected before the
  OpenCV call, with no automatic conversion or alpha handling -- convert a `(H, W)` grayscale image
  with the new `improcv.ensure_bgr` directly; `(H, W, 1)` needs the trailing axis dropped first
  (`improcv.ensure_bgr(image[..., 0])`), since `ensure_bgr` itself rejects `(H, W, 1)`; 2-channel input
  has no supported conversion at all; BGRA needs alpha explicitly dropped or composited. Unsupported channel
  layouts are rejected before the OpenCV call to avoid raw errors and build-dependent unsafe behavior
  observed for BGRA inputs (a one-off empirical reproduction, not a claim that every OpenCV version or
  platform crashes the same way). `sigma_s`/`sigma_r`/`shade_factor` are restricted to the ranges
  OpenCV's own API documents (`0 < sigma_s <= 200`, `0 < sigma_r <= 1`, `0 <= shade_factor <= 0.1`;
  `shade_factor=0` is a valid documented extreme, not a degenerate case, and is accepted) -- `sigma_r`
  is used as a divisor internally, and values outside the documented range are unsupported by OpenCV's
  own contract (stored as a C++ `float`, so an extreme value can silently degrade toward a useless
  result). `pencil_sketch` returns `PencilSketchResult(grayscale, color)` -- both fields always
  populated, since OpenCV computes both in a single internal pass regardless, so a variant-selecting
  parameter or two separate functions would add API surface without saving any computation.
  `detail_enhance` lives here, not in `improcv.restoration`: it shares `pencil_sketch`/`stylize`'s
  exact validation contract and OpenCV source module, and has no mask/inpainting semantics.
  Cross-checked that all three functions produce bit-identical output between OpenCV 4.13.0 and 5.0.0
  for the same input. Also adds `improcv.ensure_bgr` to `improcv.color`, symmetric to the existing
  `ensure_gray`: converts grayscale to BGR, passes through an already-3-channel image as a copy, and
  (deliberately, like the `photo` functions) rejects BGRA rather than guessing how to handle alpha.
  No new runtime dependency.
- New `improcv.denoising` module, Phase 4 slice 4 (non-local means denoising): `nl_means_denoise`
  (grayscale, wraps `cv2.fastNlMeansDenoising`) and `nl_means_denoise_colored` (BGR, wraps
  `cv2.fastNlMeansDenoisingColored`), both in base `opencv-python`, no contrib. Two explicit
  functions rather than one dispatching on channel count, matching the rest of the library's
  avoidance of implicit, channel-count-driven behavior. `nl_means_denoise` requires 2D `uint8`;
  `(H, W, 1)` is rejected (message points at `image[..., 0]`), as is BGR/BGRA (message points at
  `improcv.ensure_gray`). `nl_means_denoise_colored` requires exactly 3-channel `uint8` BGR;
  grayscale, `(H, W, 1)`, 2-channel, and BGRA are all rejected -- verified directly that
  `cv2.fastNlMeansDenoisingColored` technically accepts a 4-channel (BGRA) image in some builds, but
  silently replaces the output's alpha channel with a constant `255` regardless of the input alpha's
  actual content; improcv rejects BGRA outright rather than silently discarding it this way. `h`
  (and `h_luminance`/`h_color` for the colored function) must be non-negative and finite -- `0` is a
  legal, verified no-op for grayscale denoising (verified: `h=0` reproduces the input exactly; OpenCV's
  own documentation does not state this guarantee explicitly), but
  for the colored function `h_luminance=0`/`h_color=0` does **not** guarantee an identical result,
  since the BGR/CIELAB round trip alone can shift values by a few of the smallest bits regardless of
  filtering strength. No OpenCV-documented upper bound exists for `h`, so (unlike `photo.py`'s
  `sigma_s`/`sigma_r`) values are validated on their `float32`-converted form for both underflow to
  `0.0` (silently bypassing the "positive" case) and overflow to `inf` (verified directly: converting
  an extreme value with plain `np.float32(...)` raises an uncontrolled `RuntimeWarning`, now
  contained). `template_window_size`/`search_window_size` must be a positive odd integer within a
  C++ `int`'s range -- verified directly that OpenCV silently canonicalizes an even size to the next
  odd value instead of rejecting it (e.g. `templateWindowSize=2` gives the same result as `3`), which
  is why an explicit, odd value is still required rather than accepted and silently reinterpreted.
  `search_window_size` and `template_window_size` are independent parameters with no required
  relationship between them -- verified directly that `search_window_size < template_window_size` is
  not a no-op, it produces real, different output. No upper bound is imposed on window size beyond
  the `int` range: verified that a larger
  `search_window_size` substantially increases execution time, but no OpenCV-documented maximum
  exists, and any threshold picked from timing one machine/image size would both reject legitimate
  uses and fail to bound cost for a larger image anyway. Also fixes `improcv.photo`'s
  `pencil_sketch`/`stylize`/`detail_enhance`: the error message for a 2D grayscale image now points
  at `improcv.ensure_bgr(image)`, instead of only stating the dimension-count mismatch. No new
  runtime dependency.
- `improcv.photo.seamless_clone`, Phase 4 slice 5 (Poisson image editing): wraps
  `cv2.seamlessClone`, in base `opencv-python`, no contrib. `SeamlessCloneMode` exposes the three
  base modes (`"normal"`, `"mixed"`, `"monochrome_transfer"`); the `*_WIDE` variants are not
  exposed in this slice, since they use a different ROI-placement geometry (verified directly, both
  source and empirically, on both OpenCV versions). `source`/`destination` require exactly `uint8`
  `(H, W, 3)` BGR, with no automatic conversion and no size relationship required between them --
  only the region selected by `mask` matters. Verified directly that a 1-channel `destination` can
  crash the OpenCV process outright (a raw, non-Python-catchable segfault on OpenCV 4.13;
  nondeterministically a crash or a raw `cv2.error` on 5.0), so channel count is validated before
  ever reaching OpenCV. `mask` must be `uint8`, 2D, matching `source`'s spatial shape, with only `0`
  and `255` accepted -- OpenCV's own contract describes the mask as binary, matching improcv's
  existing `Mask` type convention, even though the underlying implementation happens to use the
  mask's raw value as a continuous blend weight internally. `seamless_clone` always passes a copy of
  `mask` to OpenCV: verified directly that OpenCV's own implementation mutates a single-channel mask
  array in place (zeroing its outer 1-pixel border, then eroding its active region), which the
  caller's own array must never be exposed to. That same border-zeroing is replicated before
  computing the mask's active bounding box, whose width and height must each be at least `3x3`
  pixels -- verified directly that a smaller active region makes `cv2.seamlessClone` raise an
  opaque, unhelpful native exception (sometimes just the string `"vector"`, with no file/line
  information) on both OpenCV versions. `center=(x, y)` is validated as a genuinely integral,
  `int32`-range point (rejecting `bool` and any `float`, even a whole number) via a new local
  validator, since the existing `require_point_2d` accepts floats; it is the point in
  `destination`'s coordinate system where the mask's active bounding-box center is placed (not the
  center of `source` or of all of `mask`), and its placement is validated against `destination`'s
  bounds before ever calling OpenCV, raising a `ValueError` naming the specific edge crossed rather
  than a raw `cv2.error`. An all-zero mask (or one active only on the border, which OpenCV always
  ignores) returns an exact copy of `destination` directly, without calling OpenCV. This is Poisson
  gradient-domain cloning, not alpha blending or pixel copying -- a flat-colored `source` region can
  produce little or no visible change, and even a fully `255` mask does not reproduce `source`
  exactly. `_require_valid_photo_image` (shared with `pencil_sketch`/`stylize`/`detail_enhance`) now
  takes an optional `name` keyword to customize its error messages for `seamless_clone`'s two
  image-shaped parameters (`source`/`destination`); existing callers are unaffected, since it
  defaults to `"image"`. No new runtime dependency.
- New `improcv.hdr` module, Phase 4 slice 6 (exposure fusion, the first of three planned HDR-related
  slices -- radiance HDR merge is a separate, later slice below; tone mapping remains not-yet-designed):
  `fuse_exposures`, wrapping `cv2.createMergeMertens`, in base `opencv-python`, no contrib. This
  blends a stack of differently-exposed images directly (in the domain of a Laplacian pyramid,
  weighted by local contrast/saturation/well-exposedness) -- it does not require exposure times and
  does not reconstruct a physical HDR radiance map, unlike the future radiance-merge slice, so its
  result does not need tone mapping. `images` must be a real `collections.abc.Sequence` (list, tuple,
  or another actual `Sequence`) of at least 2 `uint8` images -- a single array (including a 4D stack,
  which OpenCV's own Python binding happens to accept in place of a list), `str`/`bytes`, and a
  generator/iterator are all rejected explicitly. Every image must be non-empty, `uint8`, and have
  exactly the same shape as `images[0]`: either 2D grayscale or 3D BGR `(H, W, 3)` -- `(H, W, 1)`,
  2-channel, BGRA, and mixing grayscale/BGR within one stack are all rejected, with every error
  message naming the offending index (e.g. `images[3]`) rather than reaching an unindexed, raw
  `cv2.error`. `contrast_weight`/`saturation_weight`/`exposure_weight` must be non-negative and
  finite (validated on their `float32`-converted value, like `denoising.py`'s `h`, since there is no
  OpenCV-documented upper bound and an extreme-but-finite value can overflow to `inf` after
  conversion); `0` is legal for all three, matching `exposure_weight`'s own default. Verified directly
  that an extreme but individually representable weight (e.g. `contrast_weight=1e10`) makes OpenCV
  produce a silently non-finite (`NaN`) result on both OpenCV 4.13 and 5.0 -- `fuse_exposures` checks
  the result with `np.all(np.isfinite(...))` and raises `RuntimeError` rather than returning it.
  Output is `float32`, nominally close to `[0, 1]` but **not** guaranteed to lie exactly within it --
  verified directly that the underlying Laplacian-pyramid reconstruction can produce a small
  undershoot below `0` or overshoot above `1`; not clipped or quantized by this function. Verified
  directly that two independent calls with identical arguments are not always bit-for-bit identical
  (OpenCV's implementation uses internal parallel summation), on both OpenCV versions -- this is not
  presented as a guarantee in either direction. No new runtime dependency.
- `improcv.hdr.merge_hdr_debevec`/`merge_hdr_robertson`, Phase 4 slice 7 (radiance HDR merge -- the
  second of three planned HDR-related slices; camera-response calibration is a separate slice below,
  tone mapping remains separate and not yet designed): wrap `cv2.createMergeDebevec`/
  `cv2.createMergeRobertson`, in base `opencv-python`, no contrib. Unlike `fuse_exposures`, this
  reconstructs a physical HDR radiance map from the image stack **and** its exposure times, via a
  weighted log-average (Debevec) or weighted-
  linear (Robertson) combination of the camera response -- the two are different algorithms, not
  interchangeable variants, and are not guaranteed to produce comparable absolute radiance scales for
  the same input. `images` share `fuse_exposures`' stack contract (real `Sequence`, at least 2,
  identical shape/dtype, indexed error messages), except that **neither function supports grayscale
  input -- both require 3-channel BGR only**, for two different reasons. `cv2.MergeRobertson` raises a
  raw, unhelpful `cv2.error` for grayscale regardless of dtype, verified directly (an OpenCV
  limitation not previously documented in this project's own HDR design work). `cv2.MergeDebevec`'s
  own default (no explicit `response_curve`) linear-response construction has a confirmed bug in
  OpenCV's own C++ source: it builds a correctly-1-channel response array for grayscale input, then
  unconditionally writes through a hardcoded 3-channel accessor on the very next line, corrupting
  memory -- undefined behavior that stayed silent through this project's own local verification (3
  OpenCV builds, all macOS) but caused a real, reproducible process crash (a non-catchable native
  abort, not a raisable error) in this project's own Linux CI. Both functions therefore reject
  grayscale input unconditionally, before ever reaching OpenCV, rather than only in the specific
  triggering case. BGR input additionally accepts `uint16` and `float32` (`uint8`-only was
  reconsidered after further audit) -- `float32` values must be finite and within `[0, 1]`, since
  verified directly that OpenCV silently clips out-of-range float32 values instead of rejecting them.
  `uint16`/`float32` support in OpenCV's own HDR merge was verified directly to be version-dependent
  -- absent on `4.9.0` (this project's documented minimum OpenCV, which raises a raw, unindexed
  `CV_Assert(images[0].depth() == CV_8U)` for anything else), present on `4.13.0`/`5.0.0`, with the
  exact version boundary not pinned down. Rather than guessing a version cutoff, a new
  `improcv._compat.opencv.merge_hdr_supports_dtype` helper detects the real capability directly (a
  minimal, cached probe call against the installed OpenCV build, checked independently for
  `MergeDebevec` and `MergeRobertson` since the two are not assumed to have gained the capability
  together), and `merge_hdr_debevec`/`merge_hdr_robertson` raise a clear `ValueError` instead of
  letting that raw assertion surface on an older OpenCV build.
  `exposure_times` must contain exactly one positive, finite value per image, paired by index; always
  rebuilt into a fresh, contiguous `float32` array before reaching OpenCV -- verified directly that
  passing a `float64` array with numerically identical values can silently produce a fully non-finite
  (`inf`) result via OpenCV's own Python
  binding, with no error, and that OpenCV performs no validation on exposure times at all (zero,
  negative, and non-finite values are all silently accepted otherwise). An optional `response_curve`
  is validated structurally (dtype `float32`, finite, shape depending on the image stack's dtype --
  `256`-entry LUT for `uint8`, `65536`-entry for `uint16`/`float32`, not a universal `(256, 1, 3)`)
  and per-algorithm: `merge_hdr_debevec` requires every entry strictly positive (its logarithm is used
  internally), `merge_hdr_robertson` tolerates zero entries but rejects a negative one or an all-zero
  curve -- verified directly that OpenCV enforces neither rule itself, instead silently producing
  huge/corrupted-looking or outright `NaN` output. `response_curve=None` does not calibrate anything;
  OpenCV uses its own fixed linear response. Output is a raw `float32` radiance map, not clipped,
  normalized, or tone-mapped; checked for being a finite array of the expected shape/dtype, raising
  `RuntimeError` otherwise (the same postcondition pattern now also applied to `fuse_exposures`, see
  below). Unlike `fuse_exposures`, both merge algorithms are bit-deterministic across repeated calls
  with identical arguments, verified directly on both OpenCV 4.13 and 5.0. No new runtime dependency.
- `improcv.hdr.calibrate_camera_response_debevec`/`calibrate_camera_response_robertson`, Phase 4
  slice 8 (camera-response calibration -- the last of the radiance-HDR slices; tone mapping remains
  separate and not yet designed): wrap `cv2.createCalibrateDebevec`/`cv2.createCalibrateRobertson`,
  in base `opencv-python`, no contrib. Their output is meant for `merge_hdr_debevec`/
  `merge_hdr_robertson`'s `response_curve` parameter; neither merge function calibrates implicitly,
  matching the existing "no hidden calibration" contract. Unlike the merge functions, calibration is
  **`uint8`-only** -- verified directly, in OpenCV's own C++ source, that both calibrators assert this
  unconditionally. `calibrate_camera_response_debevec` accepts grayscale or BGR (its per-channel,
  smoothness-regularized SVD solve has no hardcoded channel-count assumption, unlike `MergeDebevec`'s
  buggy default-response path); `calibrate_camera_response_robertson` is BGR-only, rejecting grayscale
  explicitly with a message pointing at the Debevec calibrator, since verified directly that
  `cv2.CalibrateRobertson` raises a raw, unhelpful error for anything else. `samples`
  (`CalibrateDebevec`) is a positive, `int32`-range integer; for the default grid-sampling mode
  (`random_sampling=False`), this project's own validator replicates OpenCV's exact grid formula
  (`x_points = int(sqrt(samples * width / height))`, `y_points = samples // x_points`) to reject a
  `samples` value with no valid grid for the image size *before* calling OpenCV, since verified
  directly that OpenCV itself raises a raw, unindexed `CV_Assert` there -- `samples` is only ever a
  *target* count in this mode (verified directly: `samples=4` and `samples=5` can round to the
  identical grid via integer truncation). For `random_sampling=True`, no grid or pixel-count bound
  applies, matching OpenCV's own with-replacement sampling; that mode has no seed parameter in
  OpenCV's own API, so its result is not guaranteed reproducible. `smoothness` (`CalibrateDebevec`'s
  `lambda`) must be strictly positive and `float32`-safe -- verified directly that `smoothness=0` can
  produce an `inf`-valued curve, and that `NaN`/`inf` reaching OpenCV's internal SVD solver triggers
  low-level LAPACK warnings rather than a clean error, so both are rejected before ever calling
  OpenCV. `max_iterations` (`CalibrateRobertson`) is a positive, `int32`-range integer (`0` rejected
  as a misleading no-op: verified directly that it silently returns the untouched initial linear
  response). `threshold` (`CalibrateRobertson`) must be non-negative and `float32`-safe, but --
  unlike every other float32-safe parameter in this module -- a positive value underflowing to `0.0f`
  is accepted rather than rejected, since `threshold=0` is itself a legal value (it only disables
  early stopping, not iteration itself). **Real finding, not hypothetical**: verified directly that
  `CalibrateRobertson`'s per-intensity-level histogram normalization divides by the count of pixels
  observed at each of the 256 levels across the whole stack, so any level that never appears yields
  `NaN` at that entry -- an all-black or all-white image stack (or one with very few distinct
  intensity values) therefore can never produce a finite curve here, regardless of image size.
  `calibrate_camera_response_debevec` is generally more robust to sparse or degenerate intensity
  histograms because of its smoothness regularization, but a finite result is not guaranteed on every
  supported OpenCV build. Neither degenerate case is heuristically rejected before calling OpenCV;
  both go through the same `RuntimeError` postcondition as any other non-finite result.
  `exposure_times` reuses the existing, already-verified validator shared with the merge functions
  (fresh, contiguous `float32` array; strictly positive; no automatic reordering). No new runtime
  dependency.
- `improcv.hdr.tone_map`/`tone_map_drago`/`tone_map_reinhard`/`tone_map_mantiuk`, Phase 4 slice 9
  (tone mapping — the last of this phase's HDR-related slices, and a distinct operation from Mertens
  exposure fusion, radiance merge, and camera-response calibration above): wrap OpenCV's
  `cv2.createTonemap`/`createTonemapDrago`/`createTonemapReinhard`/`createTonemapMantiuk`, in base
  `opencv-python`, no contrib. Four separate functions rather than one dispatcher, since each operator
  has its own parameter set with no shared meaning across operators. `hdr` must be a non-empty,
  finite, `float32`, 3-channel BGR `(H, W, 3)` array for all four — verified directly, in OpenCV's own
  C++ source, that the base `Tonemap` class asserts exactly this, and that `TonemapDrago`/
  `TonemapReinhard`/`TonemapMantiuk` each enforce it identically, indirectly, by calling the base
  `Tonemap` internally as their own first step. Negative values are explicitly allowed (neither merge
  function guarantees non-negative radiance). `gamma` (shared by all four) must be strictly positive,
  finite, and `float32`-safe — verified directly that `gamma<=0` does not raise inside OpenCV, but
  silently produces a meaningless result (`gamma=0` reliably introduces `NaN` via `pow` applied to a
  tiny negative floating-point rounding artifact present even in well-behaved normalized output;
  negative `gamma` produces enormous, non-physical, but still finite values). `tone_map_drago`'s
  `saturation` must be strictly positive; its `bias` must be within OpenCV's own documented `[0, 1]`
  range (rejected outside it, even though specific out-of-range values were empirically observed to
  still return finite output on some test images — not treated as a stable, version-independent
  guarantee). `tone_map_reinhard`'s `intensity` must be within `[-8, 8]`, `light_adaptation`/
  `color_adaptation` (OpenCV's own parameter names: `light_adapt`/`color_adapt`, spelled out in full
  here since the mapping is unambiguous) within `[0, 1]`. `tone_map_mantiuk`'s `scale` must be nonzero
  and `float32`-safe (both positive and negative values are legal — verified directly that
  `TonemapMantiuk`'s internal `signedPow` explicitly preserves sign — but `scale=0` deterministically
  produces a non-finite result on every supported OpenCV version); its `saturation` follows the same
  contract as Drago's. **Real, non-hypothetical findings from a full four-class audit across OpenCV
  4.9.0, 4.13.0, and 5.0.0** (identical behavior across all three versions except where noted):
  a spatially constant `hdr` deterministically produces a non-finite result for `tone_map_reinhard`
  (any constant value, not just black) and `tone_map_mantiuk` (a raw, low-level `cv2.error`), so both
  reject it with a clear `ValueError` before calling OpenCV; `tone_map_drago` and `tone_map_mantiuk`
  share OpenCV's `mapLuminance` helper, which divides each pixel's channels by that same pixel's own
  luminance with no protection against zero — since OpenCV's base linear normalization maps `hdr`'s
  global minimum to exactly `0.0`, any pixel whose three channels are all already at that global
  minimum (a common case: any `hdr` whose darkest point is a true black pixel, not a synthetic corner
  case) is detected and rejected with a `ValueError` before calling OpenCV, rather than surfacing as a
  `NaN` at that exact pixel; `tone_map_mantiuk` additionally requires both spatial dimensions to be at
  least `2` (verified directly that its internal contrast pyramid has zero levels otherwise, surfacing
  as an unrelated raw `cv2.error`). Verified directly, in OpenCV's own C++ source, that
  `cv2.TonemapReinhard.process()` mutates its own object's `intensity` field in place as a side
  effect, so a fresh operator object is constructed on every call for all four functions (never
  cached or reused) — this alone neutralizes that bug for every caller of this module. Any `cv2.error`
  that still reaches OpenCV despite passing every documented validation (e.g. an internal
  conjugate-gradient solver assertion in `TonemapMantiuk`) is converted into a `RuntimeError` with the
  original exception preserved as `__cause__`, rather than leaking as a raw OpenCV error. Output is
  raw `float32`, same shape as `hdr`, never clipped or normalized — verified directly, in OpenCV's own
  C++ source, that a spatially constant, non-degenerate `hdr` bypasses `tone_map`/`tone_map_drago`'s
  own normalization branch entirely and passes the input value straight through, so OpenCV's own
  documented `[0, 1]` output range is not an unconditional guarantee. **Real, cross-architecture
  finding**: a finite result is not unconditionally guaranteed even for well-formed, non-degenerate
  `hdr` whenever an operator's internal exponent (`gamma`, `tone_map_drago`'s `bias`, or an internal,
  non-parametrized exponent for `tone_map_reinhard`/`tone_map_mantiuk`) is not an exact integer —
  OpenCV's own floating-point rounding can leave the raised value very slightly negative, and a
  negative base to a non-integer power is `NaN`. Confirmed directly that this is CPU-architecture/
  SIMD-dispatch-dependent, not just data-dependent: identical seeds and parameters that tone-map
  finitely on Apple Silicon produced a non-finite result, correctly caught by this `RuntimeError`, on
  x86_64 CI (both Linux and Windows) — this project's full CI matrix (not just local, single-
  architecture verification) is what surfaced it. No new runtime dependency.
- New `improcv.stitching` module, Phase 4 slice 10 (panorama/scan stitching — the last candidate
  slice of Phase 4): `stitch_images`, wrapping OpenCV's high-level `cv2.Stitcher`, in base
  `opencv-python`, no contrib. One function with a `mode: Literal["panorama", "scans"]` parameter
  rather than two separate functions or a status-carrying result type, since both modes share an
  identical input/output contract and differ only in the internal geometric model (homography/
  perspective for `"panorama"`, affine for `"scans"` — verified directly that `"scans"` is not limited
  to pure translation). `images` must be a real `Sequence` of at least 2 `uint8`, 3-channel BGR
  `(H, W, 3)` arrays — a single `np.ndarray` (including a 4D stack), `str`/`bytes`/`bytearray`, and a
  generator/iterator are all rejected explicitly, as are grayscale, `(H, W, 1)`, 2-channel, BGRA,
  `uint16`, `float32`, and `float64` elements, with a message naming the offending index. **Real
  findings from a full audit across OpenCV 4.9.0, 4.13.0, and 5.0.0** (identical behavior on all
  three): grayscale/`(H, W, 1)`/2-channel/BGRA each raise a different, unindexed, low-level
  `cv2.error` from deep inside the stitching pipeline; `uint16`/`float32` are silently accepted by the
  Python binding but produce a misleading generic "not enough images" failure instead of any error —
  both categories are now caught before ever reaching OpenCV. Images may have different spatial
  shapes — verified directly that OpenCV stitches differently-sized images without complaint, so this
  is not required to match. `cv2.Stitcher`'s four status codes (`OK`, `ERR_NEED_MORE_IMGS`,
  `ERR_HOMOGRAPHY_EST_FAIL`, `ERR_CAMERA_PARAMS_ADJUST_FAIL`) are stable across all three audited
  versions; a non-`OK` status raises `RuntimeError` naming the symbolic status, its numeric code, and
  a short category — **never `ValueError`, even for insufficient overlap**, since the input images are
  structurally valid and the algorithm simply could not relate them. Output postcondition (`RuntimeError`
  on violation): a non-empty `uint8` `(H, W, 3)` array — no specific shape, aspect ratio, or relationship
  to the input sizes is required, since verified directly that a successful panorama can be smaller
  than either input and is not exactly reproducible across repeated calls with identical arguments.
  **Not deterministic, even within a single process**: verified directly that OpenCV's RANSAC-based
  feature matching and geometry estimation draw from OpenCV's global RNG, so the same input images at
  a borderline amount of overlap can succeed on one call and fail on the next; `stitch_images` never
  calls `cv2.setRNGSeed` itself, since that would be a silent, global side effect on every other OpenCV
  call in the process. **Real, verified finding**: a poorly-conditioned geometry estimate (structurally
  valid images, just an orientation `mode` does not expect) can make OpenCV allocate — and report as a
  *successful* result — a panorama and internal buffers many times larger than the inputs; since that
  allocation happens inside OpenCV before this wrapper ever sees a return value, no check on the
  returned array could prevent it, so none is attempted (documented honestly in the docstring/README
  instead). A fresh `Stitcher` object is created for every call; no per-image feature masks and no
  `cv2.Stitcher` registration/seam/compositing/confidence settings are exposed in this first version.
  No new runtime dependency.

### Fixed
- `tone_map_drago`/`tone_map_mantiuk`: fixed `_require_no_zero_luminance_pixel`'s copy-through branch
  (when `hdr` is spatially constant to within `DBL_EPSILON`, which OpenCV passes through unnormalized)
  to check every pixel for zero luminance, not only `hdr[0, 0]` — a near-constant (but not
  bit-identical) `hdr` can have its zero-luminance pixel anywhere. Confirmed with a deterministic
  counterexample (`hdr = np.full((4, 4, 3), 1e-20, dtype=np.float32); hdr[2, 2] = 0.0`) that previously
  reached OpenCV and surfaced only as this module's own `RuntimeError` postcondition instead of the
  intended `ValueError` precondition.
- `calibrate_camera_response_debevec`/`calibrate_camera_response_robertson`: the output postcondition
  now also checks the returned curve's value compatibility with the corresponding merge function's
  `response_curve` contract, not just dtype/shape/finiteness. **Real finding, not hypothetical**:
  verified directly, with a deterministic counterexample (`samples=1, smoothness=1e-4` on a small
  random stack), that `CalibrateDebevec` can return a *finite* curve containing exact-zero entries --
  it estimates in log-space and then exponentiates, so a very negative but finite intermediate value
  can underflow `float32` to exactly `0.0`. Such a curve previously passed the postcondition
  unmodified, then failed less clearly downstream in `merge_hdr_debevec`, which takes the curve's
  logarithm. `calibrate_camera_response_debevec` now additionally requires every entry to be strictly
  positive, raising `RuntimeError` (never clipping or substituting a value) if not.
  `calibrate_camera_response_robertson` now additionally requires every entry to be non-negative and
  at least one entry to be positive, matching `merge_hdr_robertson`'s own, looser `response_curve`
  contract.
- `fuse_exposures`: strengthened the output postcondition to check, in order, that OpenCV's
  `MergeMertens` actually returned a `np.ndarray`, of dtype `float32`, of the expected shape, before
  checking finiteness -- previously only finiteness was checked, which would have raised a confusing
  low-level error (or silently returned a wrong result) had OpenCV ever returned something
  unexpected on either of the earlier properties. Shared with the new `merge_hdr_debevec`/
  `merge_hdr_robertson` (see above) via one common postcondition helper.
- `nl_means_denoise`/`nl_means_denoise_colored`: removed the incorrect requirement that
  `search_window_size >= template_window_size` -- the original justification (that OpenCV silently
  no-ops for a smaller search window) was false; verified directly with `h=100` that
  `search_window_size < template_window_size` can produce real, substantial filtering, matching a
  direct `cv2` call with the same arguments exactly rather than reproducing the input unchanged, so
  this is no longer rejected, documented, or tested as an error case -- instead covered by
  integration tests comparing the wrapper's output exactly against a direct `cv2` call for this
  parameter combination. Also corrected the justification for still rejecting an even window size:
  OpenCV does not silently no-op for one, it silently canonicalizes it to the next odd value (e.g.
  `templateWindowSize=2` gives the exact same result as `3`; `searchWindowSize=20` the same as `21`)
  -- the odd-size rejection itself is unchanged, only its docstring/comment justification and a new
  reference test documenting the canonicalization directly against raw `cv2` calls. Also replaced a
  test asserting a hard-coded `max_difference <= 3` bound for `nl_means_denoise_colored`'s
  `h_luminance=0, h_color=0` case with an exact-equality comparison against a direct `cv2` call plus
  no-mutation/shape/dtype checks -- the previous bound was an accidental property of one test seed
  (verified directly: a different deterministic image on the same OpenCV reaches a difference of
  `4`), not a real guarantee, so no specific difference bound is documented or tested anymore.
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
- `ssim`: the same rescaling only guarded against overflow, not against a `data_range`/image
  magnitude small enough (e.g. `1e-100`, `1e-300`) to underflow the formula's internal squared
  products to `0.0` and produce the same `NaN` from `0/0` -- including for exactly-identical images,
  which must always give `1.0` regardless of scale. Fixed with two changes: an exact-equality fast
  path (`np.array_equal`, after full validation) that always returns `1.0` for pixel-for-pixel
  identical inputs without needing any of the numeric machinery below it, and widening the rescaling
  guard to trigger symmetrically for a magnitude below `1e-75` (the exact reciprocal of the existing
  `1e75` upper bound), not just above it. Re-verified against `scikit-image` for the existing
  reference vectors (still no precision change) -- `scikit-image` itself was found to produce `NaN`
  for the new small-magnitude cases (it has no equivalent rescaling), so those are instead validated
  by confirming the result matches an equivalent unit-scale computation, which the invariance
  guarantees.
- `pencil_sketch`/`stylize`/`detail_enhance`: `sigma_s`/`sigma_r` were validated for positivity on
  their original Python value, before conversion to the `float32` OpenCV actually receives -- a
  positive-but-tiny value (e.g. `1e-46`) previously passed validation but underflowed to exactly
  `0.0` once converted, silently bypassing the `> 0` contract and reaching OpenCV as `0.0` (the same
  degenerate, all-black case the contract exists to reject). Both parameters are now validated on
  their converted `float32` value, and that exact converted value -- not a freshly re-derived one --
  is what gets passed to OpenCV. `shade_factor` is unaffected: `0.0` is already a valid, documented
  value for it, so underflowing to `0.0` reaches an already-legal case, not a hidden one.
- `pencil_sketch`/`stylize`/`detail_enhance`: the `(H, W, 1)` rejection message suggested calling
  `improcv.ensure_bgr` directly on the offending image, but `ensure_bgr` itself rejects `(H, W, 1)` --
  the message now points at dropping the trailing axis first (`improcv.ensure_bgr(image[..., 0])`).
  The 2-channel rejection message no longer mentions `ensure_bgr` at all, since there is no supported
  conversion for that channel count.

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
