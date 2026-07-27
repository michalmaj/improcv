"""Classification evaluation core: confusion matrices and precision/recall/F1/support.

Single-label multiclass only: exactly one true class and one predicted class
per sample, integer labels only. Does not cover ROC/PR curves, AUC, plotting,
multilabel classification, sample weights, or `average="binary"` -- those are
separate, later concerns, not part of this core. Two deliberate departures
from `scikit-learn.metrics`'s well-known behavior (documented at the call
sites that enforce them): a duplicate value in an explicit `labels` sequence
raises `ValueError` rather than being silently accepted, and an observed
label outside an explicit `labels` sequence raises `ValueError` rather than
silently dropping that sample from the result.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from improcv._validation import require_integral, require_one_of

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrixResult",
    "classification_metrics",
    "classification_metrics_from_confusion_matrix",
    "confusion_matrix",
]

_AVERAGE_VALUES: tuple[str | None, ...] = (None, "micro", "macro", "weighted")


@dataclass(frozen=True, slots=True, eq=False)
class ConfusionMatrixResult:
    """The result of `confusion_matrix`: the matrix plus the label each row/column stands for.

    `matrix[i, j]` is the number of samples whose true class is `labels[i]`
    and predicted class is `labels[j]` -- rows are true labels, columns are
    predicted labels. `matrix` is always a new, independent, read-only
    `int64` array; it is never a view of `y_true`/`y_pred`.

    Equality (`==`) compares `labels` structurally and `matrix` by value
    (via `np.array_equal`), never by identity -- unlike the default
    dataclass-generated equality, which would compare `matrix` with `==`
    directly and hit NumPy's "truth value of an array is ambiguous" error
    for any non-trivial matrix. Instances are unhashable (`hash()` raises
    `TypeError`), since a mutable-looking `ndarray` field makes a stable
    hash impossible to promise honestly.
    """

    matrix: npt.NDArray[np.int64]
    labels: tuple[int, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfusionMatrixResult):
            return NotImplemented
        return self.labels == other.labels and bool(np.array_equal(self.matrix, other.matrix))

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class ClassificationMetrics:
    """The result of `classification_metrics`/`classification_metrics_from_confusion_matrix`.

    `support` is always a per-class, read-only `int64` array of shape
    `(len(labels),)`, regardless of `average`. `accuracy` is always a plain
    Python `float`. `precision`/`recall`/`f1` depend on `average`:
    `average=None` gives a per-class, read-only `float64` array of shape
    `(len(labels),)`; any other `average` gives a plain Python `float`
    (never both forms in the same result -- call this function again with
    a different `average` for the other form; `..._from_confusion_matrix`
    does not recompute the underlying matrix, only the requested reduction).

    Equality (`==`) compares `labels`/`average` structurally, `support` (and
    `precision`/`recall`/`f1` when they are arrays) by value via
    `np.array_equal(..., equal_nan=True)`, and scalar float fields (including
    `accuracy` and `precision`/`recall`/`f1` when `average` is not `None`)
    by value with two `NaN`s treated as equal -- never by identity. Unlike
    the default dataclass-generated equality, which would hit the same
    "truth value of an array is ambiguous" error `ConfusionMatrixResult`
    documents. Instances are unhashable (`hash()` raises `TypeError`).
    """

    labels: tuple[int, ...]
    precision: npt.NDArray[np.float64] | float
    recall: npt.NDArray[np.float64] | float
    f1: npt.NDArray[np.float64] | float
    support: npt.NDArray[np.int64]
    accuracy: float
    average: Literal["micro", "macro", "weighted"] | None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClassificationMetrics):
            return NotImplemented
        if self.labels != other.labels or self.average != other.average:
            return False
        if not _values_equal(self.precision, other.precision):
            return False
        if not _values_equal(self.recall, other.recall):
            return False
        if not _values_equal(self.f1, other.f1):
            return False
        if not bool(np.array_equal(self.support, other.support)):
            return False
        return _float_equal_nan(self.accuracy, other.accuracy)

    __hash__ = None  # type: ignore[assignment]


def _float_equal_nan(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _values_equal(a: npt.NDArray[np.float64] | float, b: npt.NDArray[np.float64] | float) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        return bool(np.array_equal(a, b, equal_nan=True))
    return _float_equal_nan(a, b)


def confusion_matrix(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_pred: Sequence[int] | npt.NDArray[np.integer],
    *,
    labels: Sequence[int] | npt.NDArray[np.integer] | None = None,
) -> ConfusionMatrixResult:
    """Count how often each true class was predicted as each class.

    `y_true`/`y_pred` are single-label multiclass: one integer class per
    sample (a plain `Sequence` of Python/NumPy integers, or a 1-D integer
    `ndarray` -- not a 2-D array, and not a generator/iterator, which is
    rejected rather than consumed). `labels=None` infers the class universe
    as the sorted union of every value observed in `y_true` and `y_pred`.
    An explicit `labels` fixes the exact row/column order instead (not
    sorted) and must contain no duplicates; every observed value in
    `y_true`/`y_pred` must then be present in `labels` -- unlike
    `sklearn.metrics.confusion_matrix`, which silently drops a sample whose
    predicted (or true) label isn't in `labels`, this raises `ValueError`
    instead, since silently discarding part of the input is exactly the
    kind of surprise a caller is unlikely to notice on their own.

    `y_true`/`y_pred` may both be empty only when `labels` is given
    explicitly, in which case the result is a well-defined all-zero
    `len(labels) x len(labels)` matrix -- `labels=None` with empty input
    raises `ValueError` instead, since there is nothing to infer the class
    universe from.

    The returned matrix is always `int64`, shape `(len(labels), len(labels))`,
    a new array with no aliasing to `y_true`/`y_pred`, and read-only. Classes
    are mapped to dense matrix indices through an explicit label-to-index
    mapping, never by allocating `max(label) + 1` rows/columns -- so sparse
    labels (e.g. `0` and `1_000_000_000` together) cost exactly
    `len(labels) ** 2`, not `max(label) ** 2`. Building this dense matrix
    still costs `O(len(labels) ** 2)` memory regardless: a very large
    explicit `labels` can exhaust process memory even though this function
    checks that the allocation is at least *representable* first (see
    `_check_allocation_representable`) -- representability is not the same
    guarantee as "fits comfortably in available RAM".

    Raises
    ------
    TypeError
        If `y_true`/`y_pred`/`labels` is not a `Sequence` or 1-D integer
        `ndarray` (including `str`/`bytes`/`bytearray`, a 2-D array, or a
        non-integer-dtype array), or contains a non-integral element
        (including `bool`/`np.bool_`/`float`/`str`/`None`).
    ValueError
        If `y_true` and `y_pred` have different lengths, if `labels` is
        empty or contains a duplicate, if a value observed in
        `y_true`/`y_pred` is not in an explicit `labels`, if `labels=None`
        and both inputs are empty, or if `len(labels) ** 2` is not
        representable as a dense array on this platform.
    RuntimeError
        If the computed matrix fails this function's own postconditions
        (shape, dtype, or total count).
    """
    true_list = _normalize_label_sequence(y_true, "y_true")
    pred_list = _normalize_label_sequence(y_pred, "y_pred")
    if len(true_list) != len(pred_list):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(true_list)} and "
            f"{len(pred_list)}"
        )
    if labels is None and len(true_list) == 0:
        raise ValueError(
            "cannot infer labels from empty y_true/y_pred -- pass an explicit labels sequence"
        )

    resolved_labels = _resolve_labels(true_list, pred_list, labels)
    matrix = _build_confusion_matrix(true_list, pred_list, resolved_labels)
    return ConfusionMatrixResult(matrix=matrix, labels=resolved_labels)


def classification_metrics(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_pred: Sequence[int] | npt.NDArray[np.integer],
    *,
    labels: Sequence[int] | npt.NDArray[np.integer] | None = None,
    average: Literal["micro", "macro", "weighted"] | None = None,
    zero_division: float | Literal["nan"] = 0.0,
) -> ClassificationMetrics:
    """Compute precision/recall/F1/support (and accuracy) directly from labels.

    Builds the confusion matrix internally (see `confusion_matrix` for the
    exact `y_true`/`y_pred`/`labels` contract) and delegates to
    `classification_metrics_from_confusion_matrix` -- see that function for
    the full `average`/`zero_division` contract, which applies identically
    here. Unlike `confusion_matrix`, this function never accepts empty
    input, even with an explicit `labels`: `zero_division` resolves an
    undefined value for one class among others, not the complete absence of
    any observation to evaluate at all.

    Raises
    ------
    TypeError
        Same as `confusion_matrix`, plus a non-`bool`-rejecting type error
        for `zero_division`.
    ValueError
        Same as `confusion_matrix`, plus an empty `y_true`/`y_pred` (with
        or without explicit `labels`), an invalid `average`, or an invalid
        `zero_division`.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    require_one_of(average, _AVERAGE_VALUES, "average")
    zero_division_value = _normalize_zero_division(zero_division)

    true_list = _normalize_label_sequence(y_true, "y_true")
    pred_list = _normalize_label_sequence(y_pred, "y_pred")
    if len(true_list) != len(pred_list):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(true_list)} and "
            f"{len(pred_list)}"
        )
    if len(true_list) == 0:
        raise ValueError("classification_metrics requires at least one observation")

    resolved_labels = _resolve_labels(true_list, pred_list, labels)
    matrix = _build_confusion_matrix(true_list, pred_list, resolved_labels)
    confusion = ConfusionMatrixResult(matrix=matrix, labels=resolved_labels)
    return _compute_classification_metrics(confusion, average, zero_division_value)


def classification_metrics_from_confusion_matrix(
    confusion: ConfusionMatrixResult,
    *,
    average: Literal["micro", "macro", "weighted"] | None = None,
    zero_division: float | Literal["nan"] = 0.0,
) -> ClassificationMetrics:
    """Compute precision/recall/F1/support (and accuracy) from an existing confusion matrix.

    Accepts only a `ConfusionMatrixResult` (e.g. one returned by
    `confusion_matrix`, or one you build yourself after aggregating several
    batches, such as `ConfusionMatrixResult(matrix=sum(batch_matrices),
    labels=original_labels)`) -- never a bare `ndarray`, whose row/column
    order would otherwise be ambiguous. `confusion` is re-validated in
    full regardless of its declared type, since a `ConfusionMatrixResult`
    can be constructed by hand with an inconsistent `matrix`/`labels` pair:
    `matrix` must be a square, non-empty, non-negative, exactly-`int64`
    `ndarray` whose side length equals `len(labels)`, and `labels` must be
    plain (non-`bool`) Python `int`s with no duplicates. A float matrix
    (even with whole-number values), a `bool` matrix, or a matrix of any
    integer dtype other than `int64` are all rejected -- cast explicitly
    with `matrix.astype(np.int64)` first if needed.

    For each class `i`: `TP_i` is the diagonal entry, `FP_i` is that
    column's sum minus `TP_i`, `FN_i` is that row's sum minus `TP_i`, and
    `support_i` is that row's sum (`TP_i + FN_i`). `precision_i = TP_i /
    (TP_i + FP_i)`, `recall_i = TP_i / (TP_i + FN_i)`, `f1_i` is their
    harmonic mean -- each division that would divide by zero uses
    `zero_division` instead, computed without ever letting NumPy actually
    perform a `0/0`-style division (so no `RuntimeWarning` is ever raised
    here, regardless of `zero_division`). `accuracy` is always
    `trace(matrix) / matrix.sum()`, independent of `average`.

    `average=None` returns per-class `precision`/`recall`/`f1` as read-only
    `float64` arrays aligned with `confusion.labels`, always including
    every declared class (even one with zero support). `average="micro"`
    sums `TP`/`FP`/`FN` across all classes before dividing (for this
    single-label multiclass case, the three micro values are always
    numerically equal to `accuracy` -- documented, not special-cased: this
    function does not shortcut to returning `accuracy` directly for
    `average="micro"`, it always goes through the same sum-then-divide
    computation). `average="macro"` is the unweighted mean over every class
    in `confusion.labels`, including zero-support classes (each
    contributing its own `zero_division` value to the mean).
    `average="weighted"` is the mean weighted by `support`; for this same
    single-label multiclass case, weighted recall is always numerically
    equal to `accuracy` (also not special-cased).

    `zero_division="nan"` makes every undefined per-class value `NaN`;
    `"macro"`/`"weighted"` then use plain averaging (`np.mean`/a weighted
    sum), never `np.nanmean` or another NaN-skipping reduction -- so a
    single `NaN` per-class value makes the whole aggregate `NaN`, including
    under `"weighted"` when that class's own support (and therefore weight)
    is zero: `NaN * 0` is still `NaN` in IEEE 754 arithmetic, and this
    function does not mask that away. If you want zero-support classes
    excluded from an aggregate instead, filter `confusion.labels` yourself
    before calling this function.

    Raises
    ------
    TypeError
        If `confusion` is not a `ConfusionMatrixResult`, if its `matrix` is
        not an `ndarray` or not exactly `int64`, or if any of its `labels`
        is not a plain (non-`bool`) `int`, or `zero_division` is a `bool`.
    ValueError
        If `confusion`'s `matrix`/`labels` are inconsistent or invalid
        (wrong ndim, not square, empty, negative counts, mismatched
        lengths, duplicate labels, or a total count of zero), or `average`/
        `zero_division` is not one of the accepted values.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    require_one_of(average, _AVERAGE_VALUES, "average")
    zero_division_value = _normalize_zero_division(zero_division)
    _require_confusion_matrix_result(confusion)
    return _compute_classification_metrics(confusion, average, zero_division_value)


def _compute_classification_metrics(
    confusion: ConfusionMatrixResult,
    average: Literal["micro", "macro", "weighted"] | None,
    zero_division: float,
) -> ClassificationMetrics:
    matrix = confusion.matrix
    labels = confusion.labels
    n_classes = matrix.shape[0]

    tp = np.diagonal(matrix).astype(np.int64)
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)
    fp = col_sum - tp
    fn = row_sum - tp
    support = row_sum.astype(np.int64, copy=True)
    support.flags.writeable = False

    total_samples = int(matrix.sum())
    accuracy = float(np.trace(matrix)) / total_samples

    precision_per_class = _safe_divide_array(
        tp.astype(np.float64), (tp + fp).astype(np.float64), zero_division
    )
    recall_per_class = _safe_divide_array(
        tp.astype(np.float64), (tp + fn).astype(np.float64), zero_division
    )
    f1_per_class = _safe_divide_array(
        2.0 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        zero_division,
    )

    if average is None:
        precision_per_class.flags.writeable = False
        recall_per_class.flags.writeable = False
        f1_per_class.flags.writeable = False
        result = ClassificationMetrics(
            labels=labels,
            precision=precision_per_class,
            recall=recall_per_class,
            f1=f1_per_class,
            support=support,
            accuracy=accuracy,
            average=None,
        )
    elif average == "micro":
        tp_sum, fp_sum, fn_sum = float(tp.sum()), float(fp.sum()), float(fn.sum())
        precision = _safe_divide_scalar(tp_sum, tp_sum + fp_sum, zero_division)
        recall = _safe_divide_scalar(tp_sum, tp_sum + fn_sum, zero_division)
        f1 = _safe_divide_scalar(2.0 * precision * recall, precision + recall, zero_division)
        result = ClassificationMetrics(
            labels=labels,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            accuracy=accuracy,
            average="micro",
        )
    elif average == "macro":
        result = ClassificationMetrics(
            labels=labels,
            precision=float(np.mean(precision_per_class)),
            recall=float(np.mean(recall_per_class)),
            f1=float(np.mean(f1_per_class)),
            support=support,
            accuracy=accuracy,
            average="macro",
        )
    else:
        weights = support.astype(np.float64)
        total_weight = float(weights.sum())
        result = ClassificationMetrics(
            labels=labels,
            precision=_weighted_average(precision_per_class, weights, total_weight),
            recall=_weighted_average(recall_per_class, weights, total_weight),
            f1=_weighted_average(f1_per_class, weights, total_weight),
            support=support,
            accuracy=accuracy,
            average="weighted",
        )

    _check_metrics_postconditions(result, n_classes)
    return result


def _weighted_average(values: np.ndarray, weights: np.ndarray, total_weight: float) -> float:
    return float(np.sum(values * weights) / total_weight)


def _safe_divide_array(
    numerator: np.ndarray, denominator: np.ndarray, zero_division: float
) -> np.ndarray:
    result = np.full(numerator.shape, zero_division, dtype=np.float64)
    nonzero = denominator != 0
    np.divide(numerator, denominator, out=result, where=nonzero)
    return result


def _safe_divide_scalar(numerator: float, denominator: float, zero_division: float) -> float:
    if denominator == 0:
        return zero_division
    return numerator / denominator


def _normalize_zero_division(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"zero_division must be 0.0, 1.0, or 'nan', got bool ({value!r})")
    if isinstance(value, str):
        if value == "nan":
            return math.nan
        raise ValueError(f"zero_division must be 0.0, 1.0, or 'nan', got {value!r}")
    if isinstance(value, (int, float)) and (value == 0.0 or value == 1.0):
        return float(value)
    raise ValueError(f"zero_division must be 0.0, 1.0, or 'nan', got {value!r}")


def _normalize_label_sequence(value: object, name: str) -> list[int]:
    """Raise TypeError/ValueError unless `value` is a valid label container; return as `list[int]`.

    Accepts a real `collections.abc.Sequence` (explicitly excluding
    `str`/`bytes`/`bytearray`, even though they technically satisfy the
    protocol) or a 1-D integer `ndarray` -- a generator/iterator, a 2-D (or
    0-D) array, and a non-integer-dtype array (`bool`, `float`, `object`,
    ...) are all rejected. Every element must be integral (Python `int`,
    NumPy integer scalar, or `IntEnum` -- anything `numbers.Integral`
    except `bool`), normalized to a plain Python `int` so labels larger
    than `int64` remain exact and hashable in a plain `dict`.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a Sequence of integers or a 1-D integer ndarray, not "
            f"{type(value).__name__}"
        )
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {value.shape}")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")
        return [int(v) for v in value]
    if not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a Sequence of integers (e.g. a list or tuple) or a 1-D integer "
            f"ndarray, not {type(value).__name__}"
        )
    normalized: list[int] = []
    for index, element in enumerate(value):
        require_integral(element, f"{name}[{index}]")
        normalized.append(int(element))
    return normalized


def _normalize_explicit_labels(labels: object) -> tuple[int, ...]:
    normalized = _normalize_label_sequence(labels, "labels")
    if len(normalized) == 0:
        raise ValueError("labels must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must not contain duplicate values")
    return tuple(normalized)


def _resolve_labels(true_list: list[int], pred_list: list[int], labels: object) -> tuple[int, ...]:
    if labels is None:
        return tuple(sorted(set(true_list) | set(pred_list)))

    resolved = _normalize_explicit_labels(labels)
    allowed = set(resolved)
    for index, value in enumerate(true_list):
        if value not in allowed:
            raise ValueError(f"y_true[{index}] = {value} is not present in labels")
    for index, value in enumerate(pred_list):
        if value not in allowed:
            raise ValueError(f"y_pred[{index}] = {value} is not present in labels")
    return resolved


def _check_allocation_representable(n_classes: int) -> None:
    """Raise ValueError unless a dense `n_classes x n_classes` `int64` matrix is representable.

    Checked before any NumPy allocation: `cells` must fit in `np.intp` (the
    index type `np.bincount`'s flat index uses) and the total byte count
    must fit in `sys.maxsize`. Representability is a much weaker guarantee
    than "will not exhaust available memory" -- a legitimately large
    `labels` can still trigger a real, uncatchable `MemoryError`/OOM kill;
    this only rejects allocations that are not representable at all,
    before NumPy gets a chance to try.
    """
    cells = n_classes * n_classes
    intp_max = int(np.iinfo(np.intp).max)
    if cells > intp_max:
        raise ValueError(
            f"labels has {n_classes} classes; a dense {n_classes}x{n_classes} confusion "
            f"matrix has {cells} cells, which is not representable as np.intp indices on "
            "this platform"
        )
    if cells > sys.maxsize // np.dtype(np.int64).itemsize:
        raise ValueError(
            f"labels has {n_classes} classes; a dense {n_classes}x{n_classes} int64 "
            "confusion matrix is not representable as a single allocation on this platform"
        )


def _build_confusion_matrix(
    true_list: list[int], pred_list: list[int], labels: tuple[int, ...]
) -> npt.NDArray[np.int64]:
    n_classes = len(labels)
    _check_allocation_representable(n_classes)
    index_of = {label: index for index, label in enumerate(labels)}

    if len(true_list) == 0:
        matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    else:
        true_idx = np.fromiter(
            (index_of[value] for value in true_list), dtype=np.intp, count=len(true_list)
        )
        pred_idx = np.fromiter(
            (index_of[value] for value in pred_list), dtype=np.intp, count=len(pred_list)
        )
        flat = true_idx * n_classes + pred_idx
        counts = np.bincount(flat, minlength=n_classes * n_classes)
        matrix = counts.reshape(n_classes, n_classes).astype(np.int64, copy=True)

    matrix.flags.writeable = False
    _check_matrix_postconditions(matrix, expected_samples=len(true_list))
    return matrix


def _check_matrix_postconditions(matrix: np.ndarray, expected_samples: int) -> None:
    if matrix.dtype != np.int64:
        raise RuntimeError(
            f"internal error: confusion matrix has dtype {matrix.dtype}, expected int64"
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(
            f"internal error: confusion matrix has shape {matrix.shape}, expected square"
        )
    total = int(matrix.sum())
    if total != expected_samples:
        raise RuntimeError(
            f"internal error: confusion matrix sums to {total}, expected {expected_samples}"
        )
    if np.any(matrix < 0):
        raise RuntimeError("internal error: confusion matrix contains a negative count")


def _require_confusion_matrix_result(confusion: object) -> None:
    if not isinstance(confusion, ConfusionMatrixResult):
        raise TypeError(
            f"confusion must be a ConfusionMatrixResult, got {type(confusion).__name__}"
        )
    matrix = confusion.matrix
    labels = confusion.labels

    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"confusion.matrix must be an ndarray, got {type(matrix).__name__}")
    if matrix.ndim != 2:
        raise ValueError(f"confusion.matrix must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"confusion.matrix must be square, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("confusion.matrix must not be empty (0x0)")
    if matrix.dtype != np.int64:
        raise TypeError(f"confusion.matrix must have dtype int64, got {matrix.dtype}")
    if np.any(matrix < 0):
        raise ValueError("confusion.matrix must not contain negative counts")

    if not isinstance(labels, tuple):
        raise TypeError(f"confusion.labels must be a tuple, got {type(labels).__name__}")
    for index, label in enumerate(labels):
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError(f"confusion.labels[{index}] must be an int, got {type(label).__name__}")
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            f"confusion.labels has {len(labels)} entries, but confusion.matrix has "
            f"{matrix.shape[0]} rows/columns"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("confusion.labels must not contain duplicate values")
    if int(matrix.sum()) == 0:
        raise ValueError(
            "classification_metrics_from_confusion_matrix requires at least one observation "
            "(confusion.matrix sums to zero)"
        )


def _check_metrics_postconditions(result: ClassificationMetrics, n_classes: int) -> None:
    if not (isinstance(result.accuracy, float) and math.isfinite(result.accuracy)):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not a finite float")
    if not (0.0 <= result.accuracy <= 1.0):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not in [0, 1]")

    if result.support.shape != (n_classes,) or result.support.dtype != np.int64:
        raise RuntimeError("internal error: support has unexpected shape/dtype")
    if result.support.flags.writeable:
        raise RuntimeError("internal error: support is writeable")

    if result.average is None:
        for name, array in (
            ("precision", result.precision),
            ("recall", result.recall),
            ("f1", result.f1),
        ):
            if not isinstance(array, np.ndarray):
                raise RuntimeError(f"internal error: {name} is not an ndarray for average=None")
            if array.shape != (n_classes,) or array.dtype != np.float64:
                raise RuntimeError(f"internal error: {name} has unexpected shape/dtype")
            if array.flags.writeable:
                raise RuntimeError(f"internal error: {name} is writeable")
            if not np.all(np.isnan(array) | ((array >= 0.0) & (array <= 1.0))):
                raise RuntimeError(f"internal error: {name} has a value outside [0, 1] or NaN")
    else:
        for name, value in (
            ("precision", result.precision),
            ("recall", result.recall),
            ("f1", result.f1),
        ):
            if not isinstance(value, float):
                raise RuntimeError(
                    f"internal error: {name} is not a float for average={result.average!r}"
                )
            if not (math.isnan(value) or 0.0 <= value <= 1.0):
                raise RuntimeError(
                    f"internal error: aggregate {name} {value!r} is not in [0, 1] or NaN"
                )
