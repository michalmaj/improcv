"""Benchmarks for improcv's multiclass evaluation API, scaling with sample and class count.

Only ever collected by an explicit `uv run --group benchmark pytest benchmarks/` --
`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never sees this file. See `benchmarks/README.md` for what these cases answer, how to run a
stable local baseline, and how to read the results.

Four `@pytest.mark.benchmark(group=...)` groups (real `pytest-benchmark` groupings, reflected in
each benchmark entry's own `group` field, not just this comment):

- `evaluation-confusion-matrix` -- `confusion_matrix`, at three sample counts, fixed 10 classes.
- `evaluation-classification-metrics` -- `classification_metrics(..., average="macro")`, same
  sample-count axis, sharing its dataset with `evaluation-confusion-matrix`.
- `evaluation-roc-auc-macro` -- `multiclass_roc_auc_score(..., average="macro")`, at three
  sample counts (fixed 10 classes) plus two class counts (fixed 10,000 samples).
- `evaluation-average-precision-macro` -- `multiclass_average_precision_score(...,
  average="macro")`, the same five scenarios, sharing its dataset with `evaluation-roc-auc-macro`.

There is no raw NumPy or scikit-learn baseline here: no single, simple reference implementation
shares this API's full contract -- an explicit, arbitrarily ordered `labels` sequence (never
silently sorted), full type/value validation, no probability-simplex requirement on `y_score`,
strict support validation, deterministic order-independent ("canonical") reductions, and
read-only result arrays. `sklearn.metrics.roc_auc_score`'s multiclass mode, for comparison,
requires `y_score` to hold calibrated probabilities summing to `1.0` per row and requires an
explicit `labels` to already be sorted -- two real contract differences, not implementation
details, that would make any raw-vs-`improcv` ratio compare different semantics rather than the
same workflow at a different speed. This first evaluation slice measures how the public API
itself scales with sample and class count, not a ratio against a semantically different
implementation.

Every dataset is deterministic and synthetic, generated once per scenario in a session-scoped
fixture, entirely before the timed `benchmark(...)` call:

- `labels` is an explicit, deliberately *unsorted* tuple (`(1, 2, ..., n_classes - 1, 0)`) --
  this exercises the real explicit-label contract, not a sorted-by-coincidence shortcut.
- `y_true`/`y_pred` are `int64` ndarrays built from a balanced-by-construction class assignment
  (`index % n_classes`) with a deterministic error policy (every fifth sample's prediction is
  bumped to the next label in label order) -- no randomness in either.
- `y_score` (ranking scenarios only) is a `float64`, C-contiguous `(n_samples, n_classes)` matrix
  from a seeded `np.random.default_rng`, with a `+0.75` boost on each row's true-class column so
  the true class usually, but not always, ranks highest. Rows are deliberately **not**
  probability-normalized (`y_score.sum(axis=1)` is asserted not close to `1.0` in the fixture) --
  this is not an omission, since neither ranking function requires a probability simplex.

Every timed call passes the fixture's `y_score` directly, with no extra copy built in the timed
closure: `multiclass_roc_auc_score`/`multiclass_average_precision_score` perform their own
contractual copy and validation internally, and that cost is deliberately part of what this
benchmark measures, not something to shave off beforehand.
"""

from __future__ import annotations

import math
import time
from typing import NamedTuple

import numpy as np
import pytest

import improcv as im

_LABEL_SAMPLE_COUNTS: tuple[int, ...] = (1_000, 10_000, 100_000)
_LABEL_N_CLASSES = 10

_RANKING_SCENARIOS: tuple[tuple[int, int], ...] = (
    (1_000, 10),
    (10_000, 10),
    (100_000, 10),
    (10_000, 3),
    (10_000, 100),
)


class LabelDataset(NamedTuple):
    """Data shared by `confusion_matrix`/`classification_metrics` benchmarks -- no score matrix."""

    labels: tuple[int, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    n_samples: int
    n_classes: int


class RankingDataset(NamedTuple):
    """Data shared by the two multiclass ranking benchmarks -- includes the score matrix."""

    labels: tuple[int, ...]
    y_true: np.ndarray
    y_score: np.ndarray
    n_samples: int
    n_classes: int
    seed: int


def _make_labels(n_classes: int) -> tuple[int, ...]:
    """Explicit, deliberately unsorted label order: `(1, 2, ..., n_classes - 1, 0)`."""
    return (*range(1, n_classes), 0)


def _make_true_pred_columns(n_samples: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced-by-construction class-index columns, with a deterministic every-fifth error."""
    true_column = np.arange(n_samples, dtype=np.int64) % n_classes
    pred_column = true_column.copy()
    error_mask = np.arange(n_samples) % 5 == 0
    pred_column[error_mask] = (pred_column[error_mask] + 1) % n_classes
    return true_column, pred_column


def _build_label_dataset(n_samples: int, n_classes: int) -> LabelDataset:
    labels = _make_labels(n_classes)
    labels_array = np.array(labels, dtype=np.int64)
    true_column, pred_column = _make_true_pred_columns(n_samples, n_classes)
    y_true = labels_array[true_column]
    y_pred = labels_array[pred_column]
    return LabelDataset(
        labels=labels, y_true=y_true, y_pred=y_pred, n_samples=n_samples, n_classes=n_classes
    )


def _build_ranking_dataset(n_samples: int, n_classes: int) -> RankingDataset:
    labels = _make_labels(n_classes)
    labels_array = np.array(labels, dtype=np.int64)
    true_column, _ = _make_true_pred_columns(n_samples, n_classes)
    y_true = labels_array[true_column]

    seed = n_samples * 1_000 + n_classes
    rng = np.random.default_rng(seed)
    y_score = rng.random((n_samples, n_classes), dtype=np.float64)
    y_score[np.arange(n_samples), true_column] += 0.75

    assert y_score.flags.c_contiguous
    assert not np.allclose(y_score.sum(axis=1), 1.0)

    return RankingDataset(
        labels=labels,
        y_true=y_true,
        y_score=y_score,
        n_samples=n_samples,
        n_classes=n_classes,
        seed=seed,
    )


def _label_id(count: int) -> str:
    return f"{count}x{_LABEL_N_CLASSES}"


def _ranking_id(scenario: tuple[int, int]) -> str:
    n_samples, n_classes = scenario
    return f"{n_samples}x{n_classes}"


@pytest.fixture(scope="session", params=_LABEL_SAMPLE_COUNTS, ids=_label_id)
def label_dataset(request: pytest.FixtureRequest) -> LabelDataset:
    """One label-only dataset per sample count, built once per session, shared by two benchmarks."""
    n_samples = request.param
    setup_start = time.perf_counter()
    dataset = _build_label_dataset(n_samples, _LABEL_N_CLASSES)
    setup_elapsed = time.perf_counter() - setup_start
    # Diagnostic only -- never a benchmark result or a performance claim; visible with `-s`.
    print(
        f"\n[evaluation benchmark setup] label dataset {n_samples}x{_LABEL_N_CLASSES} "
        f"took {setup_elapsed:.3f}s"
    )
    return dataset


@pytest.fixture(scope="session", params=_RANKING_SCENARIOS, ids=_ranking_id)
def ranking_dataset(request: pytest.FixtureRequest) -> RankingDataset:
    """One ranking dataset per scenario, built once per session, shared by two benchmarks."""
    n_samples, n_classes = request.param
    setup_start = time.perf_counter()
    dataset = _build_ranking_dataset(n_samples, n_classes)
    setup_elapsed = time.perf_counter() - setup_start
    # Diagnostic only -- never a benchmark result or a performance claim; visible with `-s`.
    print(
        f"\n[evaluation benchmark setup] ranking dataset {n_samples}x{n_classes} took "
        f"{setup_elapsed:.3f}s, y_score.nbytes={dataset.y_score.nbytes}"
    )
    return dataset


# --- evaluation-confusion-matrix ---------------------------------------------------------------


@pytest.mark.benchmark(group="evaluation-confusion-matrix")
def test_confusion_matrix(benchmark: object, label_dataset: LabelDataset) -> None:
    """Label normalization + explicit-label validation + dense index mapping + bincount, at scale.

    Times the complete public `confusion_matrix` call against an explicit, unsorted `labels`
    order and `int64` ndarray inputs.
    """
    expected = im.confusion_matrix(
        y_true=label_dataset.y_true, y_pred=label_dataset.y_pred, labels=label_dataset.labels
    )
    assert expected.labels == label_dataset.labels
    assert expected.matrix.shape == (label_dataset.n_classes, label_dataset.n_classes)
    assert expected.matrix.dtype == np.int64
    assert not expected.matrix.flags.writeable
    assert int(expected.matrix.sum()) == label_dataset.n_samples

    result = benchmark(  # type: ignore[operator]
        im.confusion_matrix,
        y_true=label_dataset.y_true,
        y_pred=label_dataset.y_pred,
        labels=label_dataset.labels,
    )

    assert result == expected

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "confusion_matrix",
            "implementation": "improcv",
            "n_samples": label_dataset.n_samples,
            "n_classes": label_dataset.n_classes,
            "labels_explicit": True,
            "labels_sorted": False,
            "labels_order_policy": "rotated-zero-last",
            "y_true_container": "ndarray",
            "y_true_dtype": "int64",
            "y_pred_container": "ndarray",
            "y_pred_dtype": "int64",
            "prediction_policy": "every-fifth-next-label",
            "sample_weight": False,
            "class_distribution": "balanced-by-construction",
            "average": None,
            "score_matrix": False,
        }
    )


# --- evaluation-classification-metrics ---------------------------------------------------------


@pytest.mark.benchmark(group="evaluation-classification-metrics")
def test_classification_metrics_macro(benchmark: object, label_dataset: LabelDataset) -> None:
    """The full public workflow (confusion matrix -> precision/recall/F1/support/accuracy), at
    scale.

    Times the complete public `classification_metrics(..., average="macro")` call, which builds
    its own confusion matrix internally and then reduces it -- not a difference against
    `confusion_matrix` in isolation, but its own independently measured public workflow.
    """
    expected = im.classification_metrics(
        y_true=label_dataset.y_true,
        y_pred=label_dataset.y_pred,
        labels=label_dataset.labels,
        average="macro",
    )
    assert expected.labels == label_dataset.labels
    assert expected.average == "macro"
    assert expected.support.shape == (label_dataset.n_classes,)
    assert expected.support.dtype == np.int64
    assert not expected.support.flags.writeable
    assert int(expected.support.sum()) == label_dataset.n_samples
    assert isinstance(expected.accuracy, float) and math.isfinite(expected.accuracy)
    assert isinstance(expected.precision, float) and math.isfinite(expected.precision)
    assert isinstance(expected.recall, float) and math.isfinite(expected.recall)
    assert isinstance(expected.f1, float) and math.isfinite(expected.f1)

    result = benchmark(  # type: ignore[operator]
        im.classification_metrics,
        y_true=label_dataset.y_true,
        y_pred=label_dataset.y_pred,
        labels=label_dataset.labels,
        average="macro",
    )

    assert result == expected

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "classification_metrics",
            "implementation": "improcv",
            "n_samples": label_dataset.n_samples,
            "n_classes": label_dataset.n_classes,
            "labels_explicit": True,
            "labels_sorted": False,
            "labels_order_policy": "rotated-zero-last",
            "y_true_container": "ndarray",
            "y_true_dtype": "int64",
            "y_pred_container": "ndarray",
            "y_pred_dtype": "int64",
            "prediction_policy": "every-fifth-next-label",
            "sample_weight": False,
            "class_distribution": "balanced-by-construction",
            "average": "macro",
            "score_matrix": False,
        }
    )


# --- evaluation-roc-auc-macro -------------------------------------------------------------------


@pytest.mark.benchmark(group="evaluation-roc-auc-macro")
def test_multiclass_roc_auc_macro(benchmark: object, ranking_dataset: RankingDataset) -> None:
    """Per-class one-vs-rest ranking + canonical macro reduction, scaling with samples and classes.

    Times the complete public `multiclass_roc_auc_score(..., average="macro")` call, including
    `y_score`'s own contractual copy/validation, one binary ranking per class, and the final
    canonical (order-independent) mean.
    """
    expected = im.multiclass_roc_auc_score(
        ranking_dataset.y_true,
        ranking_dataset.y_score,
        labels=ranking_dataset.labels,
        average="macro",
    )
    assert isinstance(expected, float)
    assert math.isfinite(expected)
    assert 0.0 <= expected <= 1.0

    result = benchmark(  # type: ignore[operator]
        im.multiclass_roc_auc_score,
        ranking_dataset.y_true,
        ranking_dataset.y_score,
        labels=ranking_dataset.labels,
        average="macro",
    )

    assert result == expected

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "multiclass_roc_auc_score",
            "implementation": "improcv",
            "n_samples": ranking_dataset.n_samples,
            "n_classes": ranking_dataset.n_classes,
            "labels_explicit": True,
            "labels_sorted": False,
            "labels_order_policy": "rotated-zero-last",
            "y_true_container": "ndarray",
            "y_true_dtype": "int64",
            "sample_weight": False,
            "class_distribution": "balanced-by-construction",
            "average": "macro",
            "score_matrix": True,
            "score_dtype": "float64",
            "score_layout": "C",
            "score_policy": "seeded-uniform-plus-true-class-boost",
            "score_seed": ranking_dataset.seed,
            "probability_simplex_required": False,
            "rows_sum_to_one": False,
        }
    )


# --- evaluation-average-precision-macro ----------------------------------------------------------


@pytest.mark.benchmark(group="evaluation-average-precision-macro")
def test_multiclass_average_precision_macro(
    benchmark: object, ranking_dataset: RankingDataset
) -> None:
    """Per-class one-vs-rest average precision + canonical macro reduction, at scale.

    Times the complete public `multiclass_average_precision_score(..., average="macro")` call,
    sharing its dataset (and the same scaling axes) with `evaluation-roc-auc-macro`.
    """
    expected = im.multiclass_average_precision_score(
        ranking_dataset.y_true,
        ranking_dataset.y_score,
        labels=ranking_dataset.labels,
        average="macro",
    )
    assert isinstance(expected, float)
    assert math.isfinite(expected)
    assert 0.0 <= expected <= 1.0

    result = benchmark(  # type: ignore[operator]
        im.multiclass_average_precision_score,
        ranking_dataset.y_true,
        ranking_dataset.y_score,
        labels=ranking_dataset.labels,
        average="macro",
    )

    assert result == expected

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "multiclass_average_precision_score",
            "implementation": "improcv",
            "n_samples": ranking_dataset.n_samples,
            "n_classes": ranking_dataset.n_classes,
            "labels_explicit": True,
            "labels_sorted": False,
            "labels_order_policy": "rotated-zero-last",
            "y_true_container": "ndarray",
            "y_true_dtype": "int64",
            "sample_weight": False,
            "class_distribution": "balanced-by-construction",
            "average": "macro",
            "score_matrix": True,
            "score_dtype": "float64",
            "score_layout": "C",
            "score_policy": "seeded-uniform-plus-true-class-boost",
            "score_seed": ranking_dataset.seed,
            "probability_simplex_required": False,
            "rows_sum_to_one": False,
        }
    )
