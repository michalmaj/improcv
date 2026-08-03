# improcv examples

Small, self-contained, runnable recipes for the three main first-use workflows. Each script is
deterministic, needs no network access, no GUI, no Matplotlib, and no external data files -- it
creates whatever tiny input it needs (in a temporary directory it cleans up itself) and prints a
short, stable summary. Both run identically on Windows, Linux, and macOS.

## Installation

```bash
pip install "improcv[cv-headless]==0.3.0a1"
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
