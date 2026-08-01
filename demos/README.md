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
pip install "improcv[cv-headless,viz]==0.2.0a2"
```

`viz` (Matplotlib) is required to run any demo; `import improcv` itself never requires it.

## Regenerating an asset

```bash
uv run python demos/augmentation_gallery.py
uv run python demos/pairing_diagnostics.py
```

Each writes its own asset under `docs/assets/`, overwriting it in place. To write elsewhere
instead (e.g. to inspect a candidate render before committing it), pass `--output`:

```bash
uv run python demos/augmentation_gallery.py \
  --output /tmp/augmentation-gallery.png

uv run python demos/pairing_diagnostics.py \
  --output /tmp/pairing-diagnostics.png
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

## Guarantees and non-guarantees

- **Headless.** No GUI, no `cv2.imshow`, no `cv2.waitKey`. Every demo runs to completion under
  `MPLBACKEND=Agg` with no display attached.
- **No external data.** Every input is generated in-process from a synthetic scene and a fixed
  `numpy.random.Generator` seed; nothing is downloaded or read from disk.
- **Semantics, not bytes.** Regenerating an asset on a different OS, Python, NumPy, OpenCV, or
  Matplotlib version is guaranteed to reproduce the same geometry, labels, and shapes (checked by
  `tests/test_demos.py`), but **not** a bitwise-identical PNG -- font rasterization and layout
  rounding can differ slightly across Matplotlib versions and platforms. A committed asset is
  refreshed by re-running its generator in a normal development environment and reviewing the
  diff, not by pinning a specific rendering environment.
