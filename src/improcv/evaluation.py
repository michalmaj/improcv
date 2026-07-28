"""Classification evaluation core: confusion matrices, precision/recall/F1/support, and
binary ranking curves (ROC, precision-recall, ROC AUC, average precision).

`confusion_matrix`/`classification_metrics` are single-label multiclass only: exactly one
true class and one predicted class per sample, integer labels only. `roc_curve`/
`precision_recall_curve`/`roc_auc_score`/`average_precision_score` are binary, one-vs-rest: an
explicit `positive_label` picks the positive class, every other observed label is negative,
regardless of how many distinct negative labels occur -- there is no automatic inference of
which label is positive. `average_precision_score` is classification ranking average
precision, not object-detection AP or mAP. Does not cover plotting, multilabel classification,
sample weights, `average="binary"`, multiclass ranking curves/averaging, a generic `auc(x, y)`
helper, or trapezoidal PR AUC -- those are separate, later concerns, not part of this core.

Several deliberate departures from `scikit-learn.metrics`'s well-known behavior (documented at
the call sites that enforce them): a duplicate value in an explicit `labels` sequence raises
`ValueError` rather than being silently accepted; an observed label outside an explicit
`labels` sequence raises `ValueError` rather than silently dropping that sample from the
result; `roc_curve`/`roc_auc_score` raise `ValueError` for a `y_true` with no positive or no
negative sample rather than emitting `UndefinedMetricWarning` and returning a degenerate
result; and `precision_recall_curve`'s `recall` is returned in ascending order paired with
descending `thresholds` (matching `roc_curve`'s convention), not scikit-learn's descending
`recall`.
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from improcv._validation import require_integral, require_one_of

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrixResult",
    "PrecisionRecallCurve",
    "RocCurve",
    "average_precision_score",
    "classification_metrics",
    "classification_metrics_from_confusion_matrix",
    "confusion_matrix",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
]

_AVERAGE_VALUES: tuple[str | None, ...] = (None, "micro", "macro", "weighted")
_INT64_MAX = int(np.iinfo(np.int64).max)


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
        If `y_true`/`y_pred`/`labels` is not a `Sequence` or an `ndarray`
        (including `str`/`bytes`/`bytearray`, or an `ndarray` with a
        non-integer dtype), or contains a non-integral element (including
        `bool`/`np.bool_`/`float`/`str`/`None`).
    ValueError
        If `y_true` and `y_pred` have different lengths, if any of
        `y_true`/`y_pred`/`labels` is an `ndarray` that is not 1-D, if
        `labels` is empty or contains a duplicate, if a value observed in
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
    `support_i` is that row's sum (`TP_i + FN_i`). Every count involved
    (total, per-row, per-column) is verified to fit in `int64` before any
    of the following is computed -- a `ConfusionMatrixResult` built by hand
    from huge counts that would silently wrap around in raw `int64`
    arithmetic raises `ValueError` instead (see `_exact_nonnegative_int64_sum`).

    `precision_i`, `recall_i`, and `f1_i` each have their *own* zero-check,
    computed directly from `TP_i`/`FP_i`/`FN_i` -- not from each other, and
    never by adding two already-`zero_division`-filled values together:

    - `precision_i = TP_i / (TP_i + FP_i)`, using `zero_division` only when
      `TP_i + FP_i == 0` (class `i` was never predicted at all).
    - `recall_i = TP_i / (TP_i + FN_i)`, using `zero_division` only when
      `TP_i + FN_i == 0` (class `i` never occurs in the true labels at all).
    - `f1_i = 2 TP_i / (2 TP_i + FP_i + FN_i)`, using `zero_division` only
      when `2 TP_i + FP_i + FN_i == 0` (class `i` has no true positives, no
      false positives, and no false negatives -- i.e. it is completely
      absent from both `y_true` and `y_pred`).

    This matters because a class can have `TP_i = 0` with real, nonzero
    `FP_i`/`FN_i` (e.g. a class that was always confused for another) --
    there, `precision_i`/`recall_i` may or may not individually hit their
    own zero case, but `f1_i` is well-defined as plain `0.0`, *not*
    `zero_division`: verified directly that computing `f1_i` from
    `precision_i`/`recall_i` instead (`2 P_i R_i / (P_i + R_i)`) wrongly
    treats `P_i = R_i = 0` as an undefined `0/0`, which silently turns a
    correct `f1_i = 0` into `1.0` (for `zero_division=1.0`) or `NaN` (for
    `zero_division="nan"`) -- `zero_division` never changes an otherwise
    well-defined `f1_i = 0`. Each division is computed without ever letting
    NumPy actually perform a `0/0`-style division (so no `RuntimeWarning`
    is ever raised here, regardless of `zero_division`). `accuracy` is
    always `trace(matrix) / total`, independent of `average`.

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
        lengths, duplicate labels, a total count of zero, or a total count
        that exceeds what fits in `int64`), or `average`/`zero_division`
        is not one of the accepted values.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    require_one_of(average, _AVERAGE_VALUES, "average")
    zero_division_value = _normalize_zero_division(zero_division)
    _require_confusion_matrix_result(confusion)
    return _compute_classification_metrics(confusion, average, zero_division_value)


def _exact_nonnegative_int64_sum(values: np.ndarray) -> int:
    """Return the exact sum of a non-negative `int64` array's elements, without overflow.

    `values.sum(dtype=np.int64)` silently wraps around (to a negative or
    misleadingly-small positive number) once the true total exceeds
    `int64`'s range -- verified directly with a hand-constructed confusion
    matrix whose true total is exactly representable in Python but not in
    `int64`. When every element is small enough that summing all of them
    cannot possibly overflow (`max element <= INT64_MAX // element_count`),
    the fast `int64` sum is used directly; otherwise this falls back to an
    exact, arbitrary-precision Python-`int` sum via `dtype=object`.
    """
    if values.size == 0:
        return 0
    maximum = int(values.max())
    if maximum <= _INT64_MAX // values.size:
        return int(values.sum(dtype=np.int64))
    return int(values.sum(dtype=object))


def _compute_classification_metrics(
    confusion: ConfusionMatrixResult,
    average: Literal["micro", "macro", "weighted"] | None,
    zero_division: float,
) -> ClassificationMetrics:
    matrix = confusion.matrix
    labels = confusion.labels
    n_classes = matrix.shape[0]

    # `matrix` is already validated non-negative (by `_require_confusion_matrix_result`
    # for the from-confusion-matrix path, or by construction for the direct path), so
    # this exact total is also an exact upper bound for every row/column sum below --
    # once it is confirmed to fit in int64, so does every row and column sum.
    total_samples = _exact_nonnegative_int64_sum(matrix)
    if total_samples == 0:
        raise ValueError(
            "classification_metrics_from_confusion_matrix requires at least one "
            "observation (confusion.matrix sums to zero)"
        )
    if total_samples > _INT64_MAX:
        raise ValueError(
            f"confusion matrix total count ({total_samples}) exceeds what fits in a "
            f"signed 64-bit int (max {_INT64_MAX}); support cannot be represented as int64"
        )

    tp = np.diagonal(matrix).astype(np.int64)
    row_sum = matrix.sum(axis=1, dtype=np.int64)
    col_sum = matrix.sum(axis=0, dtype=np.int64)
    fp = col_sum - tp
    fn = row_sum - tp
    if np.any(tp < 0) or np.any(fp < 0) or np.any(fn < 0):
        raise RuntimeError("internal error: TP/FP/FN must not be negative")
    support = row_sum.astype(np.int64, copy=True)
    support.flags.writeable = False

    accuracy = float(np.trace(matrix)) / total_samples

    tp_float = tp.astype(np.float64)
    fp_float = fp.astype(np.float64)
    fn_float = fn.astype(np.float64)

    precision_per_class = _safe_divide_array(tp_float, tp_float + fp_float, zero_division)
    recall_per_class = _safe_divide_array(tp_float, tp_float + fn_float, zero_division)
    f1_per_class = _safe_divide_array(
        2.0 * tp_float, 2.0 * tp_float + fp_float + fn_float, zero_division
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
        f1 = _safe_divide_scalar(2.0 * tp_sum, 2.0 * tp_sum + fp_sum + fn_sum, zero_division)
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

    _check_metrics_postconditions(result, n_classes, total_samples)
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
    # The "at least one observation" check is intentionally not here: it requires
    # summing (potentially huge) matrix values, which _compute_classification_metrics
    # does safely via _exact_nonnegative_int64_sum, right before it's needed.


def _check_metrics_postconditions(
    result: ClassificationMetrics, n_classes: int, expected_total: int
) -> None:
    if not (isinstance(result.accuracy, float) and math.isfinite(result.accuracy)):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not a finite float")
    if not (0.0 <= result.accuracy <= 1.0):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not in [0, 1]")

    if result.support.shape != (n_classes,) or result.support.dtype != np.int64:
        raise RuntimeError("internal error: support has unexpected shape/dtype")
    if result.support.flags.writeable:
        raise RuntimeError("internal error: support is writeable")
    if np.any(result.support < 0):
        raise RuntimeError("internal error: support contains a negative count")
    support_total = _exact_nonnegative_int64_sum(result.support)
    if support_total != expected_total:
        raise RuntimeError(
            f"internal error: support sums to {support_total}, expected {expected_total}"
        )

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


# --- binary one-vs-rest ranking curves: RocCurve / PrecisionRecallCurve / roc_auc_score ---


@dataclass(frozen=True, slots=True, eq=False)
class RocCurve:
    """The result of `roc_curve`: false/true positive rate at each distinct score threshold.

    `false_positive_rate[i]`/`true_positive_rate[i]` are the FPR/TPR obtained by predicting
    positive for every sample with `score >= thresholds[i]`. `thresholds[0]` is always `+inf`
    (predicting nothing positive, giving `(FPR, TPR) = (0.0, 0.0)`); `thresholds[1:]` holds
    every distinct observed score in strictly decreasing order, so all three arrays share the
    same length `K + 1` for `K` distinct scores. The final point is always `(1.0, 1.0)`
    (predicting everything positive). All three arrays are new, independent, read-only
    `float64` arrays -- never views of `y_true`/`y_score` or of each other.

    Equality (`==`) compares `positive_label` and all three arrays by value (via
    `np.array_equal`), never by identity -- unlike the default dataclass-generated equality,
    which would hit the same "truth value of an array is ambiguous" error
    `ConfusionMatrixResult` documents. Instances are unhashable (`hash()` raises `TypeError`).
    """

    false_positive_rate: npt.NDArray[np.float64]
    true_positive_rate: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]
    positive_label: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RocCurve):
            return NotImplemented
        return (
            self.positive_label == other.positive_label
            and bool(
                np.array_equal(
                    self.false_positive_rate,
                    other.false_positive_rate,
                )
            )
            and bool(
                np.array_equal(
                    self.true_positive_rate,
                    other.true_positive_rate,
                )
            )
            and bool(np.array_equal(self.thresholds, other.thresholds))
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class PrecisionRecallCurve:
    """The result of `precision_recall_curve`: precision/recall at each distinct score threshold.

    `precision[i]`/`recall[i]` are the precision/recall obtained by predicting positive for
    every sample with `score >= thresholds[i]`. `thresholds[0]` is always `+inf` (predicting
    nothing positive, giving the synthetic point `(precision, recall) = (1.0, 0.0)`, with no
    corresponding real threshold); `thresholds[1:]` holds every distinct observed score in
    strictly decreasing order, so all three arrays share the same length `K + 1` for `K`
    distinct scores. `recall` is non-decreasing and its final value is always `1.0`; the final
    `precision` equals the overall positive prevalence `P / len(y_true)` (predicting everything
    positive). All three arrays are new, independent, read-only `float64` arrays -- never views
    of `y_true`/`y_score` or of each other.

    Equality (`==`) compares `positive_label` and all three arrays by value (via
    `np.array_equal`), never by identity. Instances are unhashable (`hash()` raises
    `TypeError`).
    """

    precision: npt.NDArray[np.float64]
    recall: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]
    positive_label: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrecisionRecallCurve):
            return NotImplemented
        return (
            self.positive_label == other.positive_label
            and bool(np.array_equal(self.precision, other.precision))
            and bool(np.array_equal(self.recall, other.recall))
            and bool(np.array_equal(self.thresholds, other.thresholds))
        )

    __hash__ = None  # type: ignore[assignment]


def roc_curve(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
) -> RocCurve:
    """Compute a binary, one-vs-rest ROC curve: false/true positive rate at each threshold.

    Binary one-vs-rest: a sample is positive iff its `y_true` label equals `positive_label`
    (compared with plain Python `==`, so arbitrarily large integers stay exact); every other
    observed label is negative, regardless of how many distinct negative labels occur. There is
    no automatic inference of which label is positive -- `positive_label` is always required,
    and is never inferred from label sorting or majority/minority class size.

    `y_score` is a ranking score, not a predicted label or a probability: it does not need to
    lie in `[0, 1]`, and a larger score means greater confidence in the positive class. A
    threshold classifies a sample positive iff `score >= threshold`; every sample that shares
    the same score is aggregated into a single threshold before FPR/TPR is computed there, so
    permuting the order of tied samples (or of the whole input) never changes the result.

    Requires at least one positive sample (`y_true` contains `positive_label`) and at least one
    negative sample -- with either missing, FPR or TPR would be an undefined `0/0`, so this
    raises `ValueError` rather than emitting a warning and returning a degenerate curve (compare
    `sklearn.metrics.roc_curve`, which emits `UndefinedMetricWarning` instead).

    Raises
    ------
    TypeError
        If `y_true` is not a `Sequence`/1-D integer `ndarray`, if any of its elements is not
        integral (including `bool`/`np.bool_`), if `positive_label` is not a Python/NumPy
        integral value (including `bool`/`np.bool_`), if `y_score` is not a `Sequence`/1-D
        float-or-integer `ndarray`, if `y_score` has a `bool`/complex/object/wider-than-`float64`
        floating dtype, or if a `y_score` element is not a Python/NumPy int or float.
    ValueError
        If `y_true`/`y_score` is empty, if their lengths differ, if any `y_true`/`y_score`
        `ndarray` is not 1-D, if a `y_score` element is NaN/`Inf` (or an integer not exactly
        representable as `float64`), if `y_true` has no sample equal to `positive_label`, or if
        `y_true` has no sample different from `positive_label`.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=True)
    false_positive_rate = core.false_positives.astype(np.float64) / core.n_negative
    true_positive_rate = core.true_positives.astype(np.float64) / core.n_positive
    false_positive_rate.flags.writeable = False
    true_positive_rate.flags.writeable = False
    thresholds = core.thresholds
    thresholds.flags.writeable = False

    result = RocCurve(
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        thresholds=thresholds,
        positive_label=core.positive_label,
    )
    _check_roc_curve_postconditions(result)
    return result


def precision_recall_curve(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
) -> PrecisionRecallCurve:
    """Compute a binary, one-vs-rest precision-recall curve at each distinct score threshold.

    See `roc_curve` for the shared one-vs-rest/`positive_label`/`y_score`/tie-aggregation
    contract, which applies identically here. Requires at least one positive sample; unlike
    `roc_curve`, a `y_true` with no negative sample is legal (precision is then `1.0` at every
    real threshold, since there is never a false positive to count) -- a deliberate departure
    from `sklearn.metrics.precision_recall_curve`'s own length/ordering conventions, made so that
    `precision`/`recall`/`thresholds` always share one length (`K + 1` for `K` distinct scores)
    and the same descending-threshold/ascending-recall pairing as `roc_curve`.

    `thresholds[0]` is `+inf`, giving the synthetic starting point
    `(precision, recall) = (1.0, 0.0)` with no corresponding real threshold. `recall` is
    non-decreasing and its final value is `1.0`; the final `precision` is the overall positive
    prevalence `P / len(y_true)` (predicting everything positive).

    Raises
    ------
    TypeError
        Same as `roc_curve`.
    ValueError
        Same as `roc_curve`, except a `y_true` with no negative sample is accepted rather than
        rejected.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=False)
    precision, recall = _compute_precision_recall_arrays(core)

    precision.flags.writeable = False
    recall.flags.writeable = False
    thresholds = core.thresholds
    thresholds.flags.writeable = False

    result = PrecisionRecallCurve(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        positive_label=core.positive_label,
    )
    _check_pr_curve_postconditions(result)
    return result


def average_precision_score(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
) -> float:
    """Compute binary, one-vs-rest average precision: a non-interpolated ranking-quality score.

    This is classification ranking average precision, not object-detection AP or mAP (which
    additionally require matching predictions to ground truth by IoU) -- see `roc_curve` for
    the shared one-vs-rest/`positive_label`/`y_score`/tie-aggregation contract, which applies
    identically here (same input and error contract as `precision_recall_curve`, including that
    a `y_true` with no negative sample is legal).

    Defined as the weighted mean of precision, using each recall increment as its weight:
    `AP = sum((recall[i] - recall[i - 1]) * precision[i] for i in 1..K)`, over the same `K + 1`
    grouped-threshold points `precision_recall_curve` would return (`precision[i]` is always
    taken from the *right* end of each recall increment, never `precision[i - 1]`) -- this is
    not linear interpolation and not the trapezoidal area under the PR curve, which is a
    distinct quantity that this function does not compute: depending on the shape of the curve
    and its ties, the trapezoidal area can be larger or smaller than this average precision,
    never consistently one or the other.

    A perfectly reversed ranking does not give `0.0` -- unlike ROC AUC, average precision has no
    symmetric complement relation (`average_precision_score` of the negated scores is not
    `1 - average_precision_score`). A `y_true` with no negative sample gives exactly `1.0`
    (returned directly, not computed through a `0/0`-adjacent division); constant scores (no
    discriminative power) give exactly the positive prevalence `P / len(y_true)` -- these are
    the only two inputs with a closed-form result documented here; no other ranking's average
    precision is claimed to equal prevalence.

    Raises
    ------
    TypeError
        Same as `roc_curve`.
    ValueError
        Same as `roc_curve`, except a `y_true` with no negative sample is accepted rather than
        rejected (same relaxation as `precision_recall_curve`).
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=False)

    if core.n_negative == 0:
        result = 1.0
    else:
        precision, recall = _compute_precision_recall_arrays(core)
        result = float(
            np.sum(
                np.diff(recall) * precision[1:],
                dtype=np.float64,
            )
        )

    _check_average_precision_postconditions(result)
    return result


def roc_auc_score(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
) -> float:
    """Compute the area under the binary, one-vs-rest ROC curve.

    Uses the same false/true positive rate points `roc_curve` would return (see `roc_curve` for
    the full `positive_label`/`y_score`/tie-aggregation/error contract, which applies
    identically here), integrated with the trapezoidal rule -- equivalent to the probability
    that a uniformly random positive sample outranks a uniformly random negative sample, with a
    tied pair counted as one-half. Does not call `np.trapz` or `np.trapezoid`: neither name
    exists across the full range of NumPy versions this project supports (verified directly:
    `np.trapz` is gone in current NumPy, `np.trapezoid` does not exist on this project's NumPy
    floor), so the trapezoidal sum is computed directly from basic array arithmetic instead.

    A perfect ranking gives `1.0`; a perfectly reversed ranking gives `0.0`; a `y_score` with no
    discriminative power (e.g. every sample given the same score) gives `0.5`.

    Raises
    ------
    TypeError
        Same as `roc_curve`.
    ValueError
        Same as `roc_curve`.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=True)
    false_positive_rate = core.false_positives.astype(np.float64) / core.n_negative
    true_positive_rate = core.true_positives.astype(np.float64) / core.n_positive

    area = float(
        np.sum(
            np.diff(false_positive_rate) * (true_positive_rate[:-1] + true_positive_rate[1:]) * 0.5,
            dtype=np.float64,
        )
    )
    _check_roc_auc_postconditions(area)
    return area


class _RankingCore(NamedTuple):
    """Grouped ranking thresholds shared by `roc_curve`/`precision_recall_curve`/`roc_auc_score`.

    Not a public result type -- `thresholds`/`true_positives`/`false_positives` all share length
    `K + 1` for `K` distinct observed scores, with index 0 always the `+inf` sentinel
    (`true_positives[0] = false_positives[0] = 0`) and indices `1..K` the cumulative counts at
    each of the `K` distinct scores, in strictly decreasing order.
    """

    thresholds: npt.NDArray[np.float64]
    true_positives: npt.NDArray[np.int64]
    false_positives: npt.NDArray[np.int64]
    n_positive: int
    n_negative: int
    positive_label: int


def _compute_ranking_core(
    y_true: object,
    y_score: object,
    positive_label: object,
    *,
    require_negative: bool,
) -> _RankingCore:
    require_integral(positive_label, "positive_label")
    positive_label_value = int(positive_label)  # type: ignore[call-overload]

    true_list = _normalize_label_sequence(y_true, "y_true")
    if len(true_list) == 0:
        raise ValueError("y_true must not be empty")

    scores = _normalize_score_sequence(y_score, "y_score")
    if len(true_list) != scores.shape[0]:
        raise ValueError(
            f"y_true and y_score must have the same length, got {len(true_list)} and "
            f"{scores.shape[0]}"
        )

    is_positive = np.fromiter(
        (label == positive_label_value for label in true_list),
        dtype=bool,
        count=len(true_list),
    )
    n_positive = int(np.count_nonzero(is_positive))
    n_negative = len(true_list) - n_positive
    if n_positive == 0:
        raise ValueError(
            f"y_true contains no sample equal to positive_label={positive_label_value!r}"
        )
    if require_negative and n_negative == 0:
        raise ValueError(
            "y_true contains no negative sample (every label equals "
            f"positive_label={positive_label_value!r})"
        )

    # Descending order via a stable ascending sort, reversed: cheaper than a dedicated
    # descending sort, and the within-tie order it produces is irrelevant either way, since
    # every sample sharing a score is aggregated into one threshold below.
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_is_positive = is_positive[order]

    # Direct comparison, not `np.diff(sorted_scores) != 0`: subtracting two finite scores near
    # +-float64 max can overflow to +-inf and raise a spurious "overflow encountered in
    # subtract" warning -- verified directly.
    distinct = sorted_scores[1:] != sorted_scores[:-1]
    group_end_mask = np.concatenate((distinct, np.array([True], dtype=bool)))
    group_end_indices = np.flatnonzero(group_end_mask)

    cumulative_positive = np.cumsum(sorted_is_positive, dtype=np.int64)
    cumulative_total = np.arange(1, len(true_list) + 1, dtype=np.int64)

    group_true_positive = cumulative_positive[group_end_indices]
    group_total = cumulative_total[group_end_indices]
    group_false_positive = group_total - group_true_positive

    thresholds = np.empty(group_end_indices.shape[0] + 1, dtype=np.float64)
    thresholds[0] = np.inf
    thresholds[1:] = sorted_scores[group_end_indices]

    true_positives = np.empty(group_end_indices.shape[0] + 1, dtype=np.int64)
    true_positives[0] = 0
    true_positives[1:] = group_true_positive

    false_positives = np.empty(group_end_indices.shape[0] + 1, dtype=np.int64)
    false_positives[0] = 0
    false_positives[1:] = group_false_positive

    return _RankingCore(
        thresholds=thresholds,
        true_positives=true_positives,
        false_positives=false_positives,
        n_positive=n_positive,
        n_negative=n_negative,
        positive_label=positive_label_value,
    )


def _compute_precision_recall_arrays(
    core: _RankingCore,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the precision/recall arrays shared by `precision_recall_curve` and
    `average_precision_score`.

    Returns new, independent, still-writeable `float64` buffers (never a view of `core`'s own
    `int64` arrays) -- the caller is responsible for setting them read-only before exposing them
    in a public result, or may consume them directly (e.g. `average_precision_score` never makes
    them read-only at all, since they never leave this function's caller as a public value).
    """
    true_positive = core.true_positives.astype(np.float64)
    false_positive = core.false_positives.astype(np.float64)

    precision = np.empty_like(true_positive)
    precision[0] = 1.0
    precision[1:] = true_positive[1:] / (true_positive[1:] + false_positive[1:])

    recall = true_positive / core.n_positive
    return precision, recall


_ALLOWED_SCORE_FLOAT_DTYPES = (
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
)


def _normalize_score_sequence(value: object, name: str) -> npt.NDArray[np.float64]:
    """Raise TypeError/ValueError unless `value` is a valid score container; return `float64`
    ndarray.

    Accepts a real `collections.abc.Sequence` (explicitly excluding `str`/`bytes`/`bytearray`)
    of Python `int`/`float` or NumPy `float16`/`float32`/`float64` scalars, or a 1-D `ndarray`
    with an integer or `float16`/`float32`/`float64` dtype -- a generator/iterator, a 2-D (or
    0-D) array, a `bool`/complex/object/wider-than-`float64` floating dtype array or scalar
    (e.g. `np.longdouble` where it is genuinely wider than `float64` on the current platform),
    and an empty container are all rejected -- the same floating-width policy applies whether
    the wide dtype arrives as an `ndarray` or as an individual `Sequence` element. Every
    element/value is normalized into an independent, newly allocated `float64` array (never a
    view of `value`); NaN and +/-Inf are rejected, and an integer value that is not exactly
    representable as `float64` -- whether because it loses precision (e.g. `2**53 + 1`) or
    because it is too large to convert to `float` at all (e.g. `10**400`) -- is rejected with
    `ValueError` rather than silently rounded or left to raise a raw `OverflowError`. Every zero
    in the result is canonicalized to positive zero (`+0.0`), so a threshold derived from a tied
    `+0.0`/`-0.0` group never depends on which sign of zero happened to appear first in `value`.
    """
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a Sequence of real numbers or a 1-D float/integer ndarray, "
            f"not {type(value).__name__}"
        )
    if isinstance(value, np.ndarray):
        return _normalize_score_ndarray(value, name)
    if not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a Sequence of real numbers (e.g. a list or tuple) or a 1-D "
            f"float/integer ndarray, not {type(value).__name__}"
        )
    if len(value) == 0:
        raise ValueError(f"{name} must not be empty")

    normalized = np.empty(len(value), dtype=np.float64)
    for index, element in enumerate(value):
        normalized[index] = _normalize_score_scalar(element, f"{name}[{index}]")
    _canonicalize_signed_zero(normalized)
    return normalized


def _normalize_score_scalar(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, numbers.Integral):
        integer = int(value)
        try:
            converted = float(integer)
        except OverflowError as exc:
            raise ValueError(f"{name} is not exactly representable as float64") from exc
        if not math.isfinite(converted) or int(converted) != integer:
            raise ValueError(f"{name} is not exactly representable as float64")
        return converted
    if isinstance(value, np.floating):
        # A NumPy floating scalar's own dtype can be wider than float64 (e.g. `np.longdouble`
        # on a platform where it is a genuine extended-precision type) -- checked explicitly
        # here rather than relying on `float(value)`, which would silently narrow it and could
        # collapse two distinct `longdouble` values into the same `float64` value.
        dtype = np.asarray(value).dtype
        if dtype not in _ALLOWED_SCORE_FLOAT_DTYPES:
            raise TypeError(
                f"{name} must be a Python float or a NumPy float16/float32/float64 scalar, "
                f"got {dtype}"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite, got {value}")
        return converted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        return value
    raise TypeError(f"{name} must be a Python/NumPy int or float, got {type(value).__name__}")


def _normalize_score_ndarray(value: np.ndarray, name: str) -> npt.NDArray[np.float64]:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {value.shape}")
    if value.size == 0:
        raise ValueError(f"{name} must not be empty")

    dtype = value.dtype
    if np.issubdtype(dtype, np.floating):
        if dtype not in _ALLOWED_SCORE_FLOAT_DTYPES:
            raise TypeError(
                f"{name} must have dtype float16, float32, or float64, got {dtype} -- a wider "
                "floating dtype could narrow values in a platform-dependent way"
            )
        converted = value.astype(np.float64, copy=True)
    elif np.issubdtype(dtype, np.integer):
        converted = _exact_integer_ndarray_to_float64(value, name)
    else:
        raise TypeError(f"{name} must have a floating or integer dtype, got {dtype}")

    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must contain only finite values (no NaN or Inf)")
    _canonicalize_signed_zero(converted)
    return converted


def _canonicalize_signed_zero(array: npt.NDArray[np.float64]) -> None:
    """Replace every negative zero in `array` with positive zero, in place.

    `array` is always a freshly allocated buffer owned exclusively by the caller at this point
    (never yet exposed to the public result), so mutating it here is safe and keeps the public
    contract simple: a threshold derived from a tied `+0.0`/`-0.0` group is always `+0.0`,
    regardless of which sign of zero happened to appear first in the input.
    """
    array[array == 0.0] = 0.0


def _exact_integer_ndarray_to_float64(value: np.ndarray, name: str) -> npt.NDArray[np.float64]:
    """Widen an integer ndarray to float64, rejecting any value that would lose precision.

    `value.tolist()` gives exact Python ints for any NumPy integer dtype, including `uint64`
    magnitudes that don't fit in `int64` -- comparing each against its own `float64` round-trip
    catches precision loss above `2**53` (e.g. `2**53 + 1`) that a bare `.astype(np.float64)`
    would otherwise silently round away.
    """
    converted = value.astype(np.float64, copy=True)
    pairs = zip(value.tolist(), converted.tolist(), strict=True)
    for index, (original, as_float) in enumerate(pairs):
        if int(as_float) != original:
            raise ValueError(f"{name}[{index}] is not exactly representable as float64")
    return converted


def _check_curve_postconditions(
    thresholds: np.ndarray, values_a: np.ndarray, values_b: np.ndarray, kind: str
) -> None:
    for label, array in (
        ("thresholds", thresholds),
        (f"{kind}_a", values_a),
        (f"{kind}_b", values_b),
    ):
        if not isinstance(array, np.ndarray):
            raise RuntimeError(f"internal error: {label} is not an ndarray")
        if array.dtype != np.float64:
            raise RuntimeError(f"internal error: {label} has dtype {array.dtype}, expected float64")
        if array.ndim != 1:
            raise RuntimeError(f"internal error: {label} has ndim {array.ndim}, expected 1")

    length = thresholds.shape[0]
    if values_a.shape[0] != length or values_b.shape[0] != length:
        raise RuntimeError("internal error: curve arrays have mismatched lengths")
    if length < 2:
        raise RuntimeError(f"internal error: curve has length {length}, expected at least 2")

    if thresholds[0] != np.inf:
        raise RuntimeError("internal error: thresholds[0] must be +inf")
    rest = thresholds[1:]
    if not np.all(np.isfinite(rest)):
        raise RuntimeError("internal error: thresholds[1:] must be finite")
    if rest.shape[0] >= 2 and not np.all(rest[1:] < rest[:-1]):
        raise RuntimeError("internal error: thresholds[1:] must be strictly decreasing")

    for label, array in (
        ("thresholds", thresholds),
        (f"{kind}_a", values_a),
        (f"{kind}_b", values_b),
    ):
        if array.flags.writeable:
            raise RuntimeError(f"internal error: {label} is writeable")

    if (
        np.shares_memory(thresholds, values_a)
        or np.shares_memory(thresholds, values_b)
        or np.shares_memory(values_a, values_b)
    ):
        raise RuntimeError("internal error: curve arrays alias each other")


def _check_roc_curve_postconditions(result: RocCurve) -> None:
    _check_curve_postconditions(
        result.thresholds, result.false_positive_rate, result.true_positive_rate, "roc"
    )
    for label, array in (
        ("false_positive_rate", result.false_positive_rate),
        ("true_positive_rate", result.true_positive_rate),
    ):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"internal error: {label} contains a non-finite value")
        if not np.all((array >= 0.0) & (array <= 1.0)):
            raise RuntimeError(f"internal error: {label} has a value outside [0, 1]")
        if array.shape[0] >= 2 and not np.all(array[1:] >= array[:-1]):
            raise RuntimeError(f"internal error: {label} must be non-decreasing")

    if result.false_positive_rate[0] != 0.0 or result.true_positive_rate[0] != 0.0:
        raise RuntimeError("internal error: ROC curve must start at (0, 0)")
    if result.false_positive_rate[-1] != 1.0 or result.true_positive_rate[-1] != 1.0:
        raise RuntimeError("internal error: ROC curve must end at (1, 1)")


def _check_pr_curve_postconditions(result: PrecisionRecallCurve) -> None:
    _check_curve_postconditions(result.thresholds, result.precision, result.recall, "pr")
    for label, array in (("precision", result.precision), ("recall", result.recall)):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"internal error: {label} contains a non-finite value")
        if not np.all((array >= 0.0) & (array <= 1.0)):
            raise RuntimeError(f"internal error: {label} has a value outside [0, 1]")

    if result.recall.shape[0] >= 2 and not np.all(result.recall[1:] >= result.recall[:-1]):
        raise RuntimeError("internal error: recall must be non-decreasing")
    if result.precision[0] != 1.0 or result.recall[0] != 0.0:
        raise RuntimeError("internal error: PR curve must start at (precision=1, recall=0)")
    if result.recall[-1] != 1.0:
        raise RuntimeError("internal error: PR curve must end with recall == 1")


def _check_roc_auc_postconditions(value: float) -> None:
    if not isinstance(value, float):
        raise RuntimeError(
            f"internal error: roc_auc_score result is not a float, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise RuntimeError(f"internal error: roc_auc_score result {value!r} is not finite")
    tolerance = 1e-9
    if not (-tolerance <= value <= 1.0 + tolerance):
        raise RuntimeError(f"internal error: roc_auc_score result {value!r} is outside [0, 1]")


def _check_average_precision_postconditions(value: float) -> None:
    if not isinstance(value, float):
        raise RuntimeError(
            f"internal error: average_precision_score result is not a float, got "
            f"{type(value).__name__}"
        )
    if not math.isfinite(value):
        raise RuntimeError(
            f"internal error: average_precision_score result {value!r} is not finite"
        )
    tolerance = 1e-9
    if not (-tolerance <= value <= 1.0 + tolerance):
        raise RuntimeError(
            f"internal error: average_precision_score result {value!r} is outside [0, 1]"
        )
