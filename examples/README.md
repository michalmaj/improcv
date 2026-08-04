# improcv examples

Five small, self-contained, runnable recipes for the main first-use workflows. Each script is
deterministic, needs no network access, no GUI, no Matplotlib, and no external data files -- it
creates whatever tiny input it needs (in a temporary directory it cleans up itself) and prints a
short, stable summary. All five run identically on Windows, Linux, and macOS.

## Installation

```bash
pip install "improcv[cv-headless]==0.3.0b1"
```

For a development checkout of this repository:

```bash
uv sync
```

## Running the examples

```bash
uv run python examples/discovery_and_augmentation.py
uv run python examples/classification_evaluation.py
uv run python examples/image_similarity.py
uv run python examples/image_similarity_manifest.py
uv run python examples/dataset_manifest_builder.py
```

(Without `uv`, `python examples/discovery_and_augmentation.py` works too, as long as `improcv` and
one of its OpenCV extras are installed in the active environment.)

## Spatial convention

The discovery/augmentation and DNN-related material below rely on one convention that is easy to
get backwards:

```text
NumPy array shape: (height, width[, channels])
improcv/OpenCV size tuples: (width, height)
```

`source_size`/`output_size` on augmentation parameters, and `size=` on the DNN preprocessing
helpers, are all `(width, height)` -- the reverse of `array.shape[:2]`.

## What's here

- [`discovery_and_augmentation.py`](discovery_and_augmentation.py) -- deterministic dataset
  image/mask pairing (`discover_image_mask_pairs`), then replayable affine augmentation
  (`sample_affine`/`apply_affine`) applied identically to an image and its segmentation mask,
  including replay and canvas expansion (`expand_affine_canvas`).
- [`classification_evaluation.py`](classification_evaluation.py) -- confusion matrix and
  per-class metrics (`confusion_matrix`/`classification_metrics`), then multiclass one-vs-rest
  ranking evaluation (`multiclass_roc_auc_score`/`multiclass_average_precision_score`) across
  `average=None`/`"macro"`/`"weighted"`/`"micro"`.
- [`image_similarity.py`](image_similarity.py) -- four small, deterministic synthetic images
  discovered with `discover_images`, decoded and hashed exactly once with `average_hash`, then
  searched for similar pairs with `find_similar_image_pairs`. The same, already-computed hash
  mapping is re-searched at a second, wider threshold without any redecoding or rehashing, and the
  result demonstrates that threshold similarity is not transitive: two overlapping pairs are
  reported, never one merged group of three. This is a deliberately tiny, hand-checkable
  demonstration of the workflow, not a benchmark or a production deduplication tool.
- [`image_similarity_manifest.py`](image_similarity_manifest.py) -- the same four synthetic
  images, but focused entirely on persistence rather than hashing or thresholds: each image is
  decoded and hashed exactly once, the resulting hashes are stored under relative, portable
  identifiers in a `PerceptualHashManifest`, and the manifest is written out with `manifest.save()`.
  The dataset directory is then physically moved to a different path, the manifest is reloaded
  with `PerceptualHashManifest.load()`, and a similarity search runs entirely against the reloaded
  hashes -- with no re-decoding and no re-hashing. A `PerceptualHashManifest` is a snapshot, not a
  cache: it never checks freshness or whether an image still exists, so reusing a reloaded
  manifest like this is a deliberate choice made once, not something the manifest verifies for you.
  **Manual, educational equivalent** of `dataset_manifest_builder.py`, below.
- [`dataset_manifest_builder.py`](dataset_manifest_builder.py) -- the same four synthetic images
  as the two `image_similarity*.py` scripts, but discovered, decoded, hashed, and turned into
  relative identifiers by one call to `build_perceptual_hash_manifest`, under a Unicode dataset
  root (`żółw-dataset/obrazy/`) to exercise its cross-platform, Unicode-safe decode contract and
  its nested, portable relative identifiers directly. The fixed grayscale decode policy is the
  same one the manual scripts use. The resulting manifest is saved, the dataset root is physically
  moved, the manifest is reloaded, and a similarity search runs against the reloaded hashes -- the
  same persistence/move/reload/search shape as `image_similarity_manifest.py`, with the discovery/
  decode/hash/relativize/`from_hashes` steps collapsed into one function call. Like every
  `PerceptualHashManifest`, it is a snapshot, not a cache: no freshness check, and no external
  data, network, or GUI is used anywhere in this script. **Concise, recommended workflow** -- see
  `image_similarity_manifest.py`, above, for the manual, educational equivalent it corresponds to.

## DNN preprocessing as a supporting layer

`improcv` does not (yet) provide a full inference or postprocessing wrapper. `create_dnn_blob`/
`create_dnn_batch_blob` (preprocessing) and `load_onnx_network`/`load_onnx_network_from_bytes`
(ONNX loading) exist to get you *to* a model's raw output; from there, that raw output is exactly
the `y_score` the classification-evaluation functions above expect:

```python
import improcv as im

blob = im.create_dnn_batch_blob(images, size=(224, 224), scale=1.0 / 255.0, swap_rb=True)
net = im.load_onnx_network("model.onnx")
net.setInput(blob)
y_score = net.forward()  # raw OpenCV output -- shape and meaning depend entirely on the model

scores = im.multiclass_roc_auc_score(y_true, y_score, labels=labels)
```

`net.setInput`/`net.forward` are plain, unwrapped `cv2.dnn.Net` API -- `improcv` does not change
their behavior. What `y_score`'s columns mean (which column is which class, whether it needs a
softmax first, etc.) is entirely determined by the model you loaded, not by `improcv`. This
snippet is an **integration pattern**, not a standalone, runnable example -- it has no
`examples/*.py` file of its own, and this repository does not ship a test ONNX model for it to run
against.
