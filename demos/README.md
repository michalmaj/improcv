# Demos

`demos/` and [`examples/`](../examples/README.md) serve different purposes:

- **`examples/`** holds copyable, end-to-end workflows. You are meant to open one, read it, and
  adapt it into your own code.
- **`demos/`** holds reproducible *generators* for the visual assets committed under
  [`docs/assets/`](../docs/assets/) -- each script's only job is to produce one documentation
  image, deterministically, from `improcv`'s public API. You are not meant to copy a `demos/`
  script into your own project; you are meant to run it, or read it to see exactly how a
  committed image was produced.

Every asset under `docs/assets/` has exactly one generator here that is its source of truth. If
you find a mismatch between an asset and its generator, the asset is stale and should be
regenerated -- **never edit a PNG under `docs/assets/` by hand.**

## Installation

Development (from a clone, editable install with the extras the demos need):

```bash
uv sync --extra cv-headless --extra viz
```

Using a published release:

```bash
pip install "improcv[cv-headless,viz]==0.4.0a2"
```

`viz` (Matplotlib) is required to run any demo; `import improcv` itself never requires it.

## Regenerating an asset

```bash
uv run python demos/augmentation_gallery.py
uv run python demos/classification_report.py
uv run python demos/pairing_diagnostics.py
uv run python demos/image_similarity_gallery.py
```

Each writes its own asset under `docs/assets/`, overwriting it in place. To write elsewhere
instead (e.g. to inspect a candidate render before committing it), pass `--output`:

```bash
uv run python demos/augmentation_gallery.py \
  --output /tmp/augmentation-gallery.png

uv run python demos/classification_report.py \
  --output /tmp/classification-report.png

uv run python demos/pairing_diagnostics.py \
  --output /tmp/pairing-diagnostics.png

uv run python demos/image_similarity_gallery.py \
  --output /tmp/image-similarity-gallery.png
```

## What each demo shows

### `augmentation_gallery.py`

Builds a small synthetic BGR image and its matching integer-label segmentation mask, then applies
`improcv`'s replayable affine (both fixed and expanded canvas) and perspective transforms
identically to the image and the mask. The rendered gallery shows:

- the same sampled parameter object applied to an image and its mask;
- that a mask keeps only its original discrete labels plus one explicit border value after a
  warp (never an interpolated, illegal label);
- the difference between a fixed-size affine canvas (content can be cropped) and an expanded one
  (`improcv.expand_affine_canvas`, content is never cropped, but the canvas grows);
- that the current perspective transform always renders back to the source canvas size.

For a copyable, executable version of the same underlying contract (dataset discovery + one
affine replay, without any rendering), see
[`examples/discovery_and_augmentation.py`](../examples/discovery_and_augmentation.py).

### `classification_report.py`

<img
  src="https://raw.githubusercontent.com/michalmaj/improcv/main/docs/assets/classification-report.png"
  alt="improcv multiclass classification report showing an explicitly ordered confusion matrix, per-class precision recall F1 support, and per-class plus macro weighted and micro ROC AUC and average precision scores"
  width="880"
>

Runs the existing public evaluation workflow -- `confusion_matrix` -> `classification_metrics` ->
`multiclass_roc_auc_score` -> `multiclass_average_precision_score` -- on a small, explicit,
hand-written multiclass example (the same one used by
[`examples/classification_evaluation.py`](../examples/classification_evaluation.py), duplicated
here so this generator stays self-contained). Shows:

- `labels = [20, 10, 30]`, deliberately unsorted: it fixes both the confusion matrix's row/column
  order and which `y_score` column belongs to which class -- `y_score[:, i]` corresponds to
  `labels[i]`, never a sorted order;
- the confusion matrix (rows are true labels, columns are predicted labels);
- per-class precision, recall, F1, and support, plus accuracy and macro F1;
- per-class ROC AUC and average precision, plus macro/weighted/micro summaries for both;
- that `y_score`'s rows do not need to sum to `1.0` -- `improcv`'s multiclass ranking functions
  never require a probability simplex.

This demo does not call `roc_curve`/`precision_recall_curve` and does not render any curve --
`improcv` does not yet have a public multiclass curve result type, so this shows the stable
score-based API only, not a design for API that does not exist yet.

For a copyable, executable version of the same workflow, see
[`examples/classification_evaluation.py`](../examples/classification_evaluation.py).

### `pairing_diagnostics.py`

<img
  src="https://raw.githubusercontent.com/michalmaj/improcv/main/docs/assets/pairing-diagnostics.png"
  alt="improcv pairing diagnostics showing deterministic successful image-mask pairing, a silent mismatch produced by positional zip, and precise errors for unmatched files and duplicate pairing keys"
  width="880"
>

Builds three tiny synthetic datasets and pairs each with `improcv.discover_image_mask_pairs`,
comparing the real result against a naive positional `zip(sorted(images), sorted(masks))`. Shows:

- a successful, deterministic key-based pairing;
- the same dataset paired naively by position -- silently correct only by coincidence;
- an unmatched image and an unmatched mask, and the real `ValueError`
  `discover_image_mask_pairs` raises identifying both;
- a duplicate pairing key (two image files reducing to the same key), and the real `ValueError`
  identifying the colliding key and both competing paths.

Every directory tree, naive-zip result, and exception message shown is computed from a real
filesystem and a real call to the public API -- none of it is a hand-typed example message.

For a copyable, executable version of the underlying discovery contract, see
[`examples/discovery_and_augmentation.py`](../examples/discovery_and_augmentation.py).

### `image_similarity_gallery.py`

<img
  src="https://raw.githubusercontent.com/michalmaj/improcv/main/docs/assets/image-similarity-gallery.png"
  alt="improcv image similarity showing four synthetic 8x8 images with their average_hash hex values, a symmetric 4x4 Hamming distance matrix with two in-threshold pairs highlighted, and the two pairs returned by find_similar_image_pairs at max_distance=2 with a_base.png and c_variant_more.png shown as a non-transitive gap at distance 4"
  width="880"
>

Builds four small, deterministic 8x8 grayscale images -- three close variants of one another (A,
B, C) and one unrelated image (D) -- hashes each with `average_hash`, and computes the real
pairwise Hamming-distance matrix and `find_similar_image_pairs` result. Shows:

- each image alongside its exact `average_hash` hex value;
- the full 4x4 distance matrix, with the two in-threshold pairs (A-B, B-C) visually highlighted
  and A-C's distance (4) shown outside the threshold;
- the two pairs `find_similar_image_pairs` actually returns at `max_distance=2`, and why A-C is
  not a third pair even though A~B and B~C both match -- threshold similarity is not transitive.

`average_hash` is used here (rather than `phash`, as in the main README's realistic-photo
snippet) specifically because its bit decisions on a small checkerboard are easy to verify by
hand; `max_distance=2` is specific to this tiny synthetic example, not a general recommendation.
Every hash, matrix value, and pair shown is computed via the public API, never hand-typed.

For a copyable, executable version of the same workflow, see
[`examples/image_similarity.py`](../examples/image_similarity.py).

## Guarantees and non-guarantees

- **Headless.** No GUI, no `cv2.imshow`, no `cv2.waitKey`. Every demo runs to completion under
  `MPLBACKEND=Agg` with no display attached.
- **No external data or network access.** Inputs are generated locally:
  `augmentation_gallery.py` uses a seeded synthetic scene, `classification_report.py` uses small
  explicit values, `pairing_diagnostics.py` uses an automatically cleaned temporary directory so
  it can exercise the real filesystem discovery API, and `image_similarity_gallery.py` builds its
  four images in memory from fixed pixel values.
- **Semantics, not bytes.** Regenerating an asset on a different OS, Python, NumPy, OpenCV, or
  Matplotlib version is guaranteed to reproduce the same geometry, labels, and shapes (checked by
  `tests/test_demos.py`), but **not** a bitwise-identical PNG -- font rasterization and layout
  rounding can differ slightly across Matplotlib versions and platforms. A committed asset is
  refreshed by re-running its generator in a normal development environment and reviewing the
  diff, not by pinning a specific rendering environment.
