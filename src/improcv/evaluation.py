"""Classification evaluation core: confusion matrices, precision/recall/F1/support, and
binary ranking curves (ROC, precision-recall, ROC AUC, average precision), plus a generic
trapezoidal `auc(x, y)` helper.

`confusion_matrix`/`classification_metrics` are single-label multiclass only: exactly one
true class and one predicted class per sample, integer labels only. `roc_curve`/
`precision_recall_curve`/`roc_auc_score`/`average_precision_score` are binary, one-vs-rest: an
explicit `positive_label` picks the positive class, every other observed label is negative,
regardless of how many distinct negative labels occur -- there is no automatic inference of
which label is positive. All six of these functions accept an optional, keyword-only
`sample_weight` (see each function's own docstring for the full contract); `classification_metrics_
from_confusion_matrix` has no `sample_weight` parameter of its own but accepts a `float64`
confusion matrix (built with weights, or by hand) as readily as the `int64` one `confusion_matrix`
returns without `sample_weight` -- `ConfusionMatrixResult.matrix`/`ClassificationMetrics.support`
are `int64` exactly when no weights were involved, `float64` whenever they were, regardless of the
weights' own dtype or values. `auc`'s `x`/`y` are curve points, not observations, so a
per-observation weight has no meaning there, and it has no `sample_weight` parameter.
`average_precision_score` is classification ranking average precision, not object-detection AP
or mAP. `auc` is a general-purpose trapezoidal area-under-curve helper with no ranking
semantics of its own -- it also computes the trapezoidal area under the precision-recall curve
(`auc(curve.recall, curve.precision)`), a distinct quantity from `average_precision_score` that
this module deliberately does not expose under a separate name (see `auc`'s docstring). Does
not cover plotting, multilabel classification, `average="binary"`, or multiclass ranking
curves/averaging -- those are separate, later concerns, not part of this core.

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
from fractions import Fraction
from typing import Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from improcv._validation import require_integral, require_one_of

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrixResult",
    "PrecisionRecallCurve",
    "RocCurve",
    "auc",
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

    `matrix[i, j]` is the number of samples (or, with `sample_weight`, the total effective
    weight of samples) whose true class is `labels[i]` and predicted class is `labels[j]` --
    rows are true labels, columns are predicted labels. `matrix` is always a new, independent,
    read-only array; it is never a view of `y_true`/`y_pred`. Its dtype depends only on whether
    `sample_weight` was given, never on the content of the weights: exactly `int64` when
    `confusion_matrix` was called with `sample_weight=None` (the default), exactly `float64`
    when it was called with an explicit `sample_weight` -- even one holding only `1.0`s or
    exact-integer values. A `ConfusionMatrixResult` can also be constructed by hand with a
    `float64` matrix (e.g. after aggregating weighted batches); see `classification_metrics_from_
    confusion_matrix` for that matrix's full contract.

    Equality (`==`) compares `labels` structurally and `matrix` by value (via `np.array_equal`),
    never by identity and never by dtype -- an `int64` matrix and a `float64` matrix holding the
    same values at the same shape compare equal, since dtype reflects how a result was computed
    (unweighted vs. weighted), not part of its semantic value. Unlike the default
    dataclass-generated equality, which would compare `matrix` with `==` directly and hit NumPy's
    "truth value of an array is ambiguous" error for any non-trivial matrix. Instances are
    unhashable (`hash()` raises `TypeError`), since a mutable-looking `ndarray` field makes a
    stable hash impossible to promise honestly.
    """

    matrix: npt.NDArray[np.int64] | npt.NDArray[np.float64]
    labels: tuple[int, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConfusionMatrixResult):
            return NotImplemented
        return self.labels == other.labels and bool(np.array_equal(self.matrix, other.matrix))

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class ClassificationMetrics:
    """The result of `classification_metrics`/`classification_metrics_from_confusion_matrix`.

    `support` is always a per-class, read-only array of shape `(len(labels),)`, regardless of
    `average` -- exactly `int64` when computed without `sample_weight`, exactly `float64` when
    computed with an explicit `sample_weight` (or from a hand-built `float64` confusion matrix),
    mirroring `ConfusionMatrixResult.matrix`'s own dtype rule. `accuracy` is always a plain Python
    `float`. `precision`/`recall`/`f1` depend on `average`: `average=None` gives a per-class,
    read-only `float64` array of shape `(len(labels),)` regardless of `sample_weight` (these were
    already `float64` ratios, never counts); any other `average` gives a plain Python `float`
    (never both forms in the same result -- call this function again with a different `average`
    for the other form; `..._from_confusion_matrix` does not recompute the underlying matrix,
    only the requested reduction).

    Equality (`==`) compares `labels`/`average` structurally, `support` (and `precision`/
    `recall`/`f1` when they are arrays) by value via `np.array_equal(..., equal_nan=True)`, and
    scalar float fields (including `accuracy` and `precision`/`recall`/`f1` when `average` is not
    `None`) by value with two `NaN`s treated as equal -- never by identity and never by
    `support`'s dtype: an `int64` `support` and a `float64` `support` holding the same values
    compare equal, for the same reason `ConfusionMatrixResult`'s `matrix` equality does. Unlike
    the default dataclass-generated equality, which would hit the same "truth value of an array
    is ambiguous" error `ConfusionMatrixResult` documents. Instances are unhashable (`hash()`
    raises `TypeError`).
    """

    labels: tuple[int, ...]
    precision: npt.NDArray[np.float64] | float
    recall: npt.NDArray[np.float64] | float
    f1: npt.NDArray[np.float64] | float
    support: npt.NDArray[np.int64] | npt.NDArray[np.float64]
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
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
) -> ConfusionMatrixResult:
    """Count how often each true class was predicted as each class.

    `y_true`/`y_pred` are single-label multiclass: one integer class per
    sample (a plain `Sequence` of Python/NumPy integers, or a 1-D integer
    `ndarray` -- not a 2-D array, and not a generator/iterator, which is
    rejected rather than consumed). `labels=None` infers the class universe
    as the sorted union of every value observed in `y_true` and `y_pred`
    (or, with `sample_weight`, every value observed among only the *effective*
    positive-weight samples -- see below). An explicit `labels` fixes the
    exact row/column order instead (not sorted) and must contain no
    duplicates; every observed value in `y_true`/`y_pred` must then be
    present in `labels` -- unlike `sklearn.metrics.confusion_matrix`, which
    silently drops a sample whose predicted (or true) label isn't in
    `labels`, this raises `ValueError` instead, since silently discarding
    part of the input is exactly the kind of surprise a caller is unlikely
    to notice on their own. This label validation always runs against every
    raw `y_true`/`y_pred` value, regardless of `sample_weight`: a sample
    with an invalid label is never forgiven merely because its weight is
    zero.

    `y_true`/`y_pred` may both be empty only when `labels` is given
    explicitly, in which case the result is a well-defined all-zero
    `len(labels) x len(labels)` matrix (`int64` without `sample_weight`,
    `float64` with it, including with `sample_weight=[]`) -- `labels=None`
    with empty input raises `ValueError` instead, since there is nothing to
    infer the class universe from.

    `sample_weight=None` (the default) uses the unweighted computation above
    unchanged, bit for bit: the returned matrix is always `int64` in that
    case. Given explicitly, `sample_weight` must be a `Sequence`/1-D
    `ndarray` the same length as `y_true`/`y_pred`, holding the same accepted
    numeric types as `y_true`'s companion `y_score` does in the ranking
    functions (Python/NumPy int/float, or `float16`/`float32`/`float64`), but
    non-negative -- a negative weight would make a matrix cell negative,
    breaking this function's own non-negativity guarantee (verified directly
    against a reference implementation that permits negative weights and
    does exactly that). Unlike the ranking functions' `sample_weight`, an
    all-zero `sample_weight` is legal here (see above) and unlike them there
    is no "at least one positive value" requirement at this function's own
    level -- the returned matrix is always `float64` whenever `sample_weight`
    is given at all, regardless of the weights' own dtype or values (even
    `sample_weight=[1, 1, ...]` or `sample_weight=[1.0, 1.0, ...]` gives a
    `float64` matrix, never `int64`) and regardless of whether any individual
    cell's value happens to be a whole number.

    A sample with `sample_weight == 0.0` contributes nothing to any cell and
    is removed from the *effective* set before class inference (`labels=
    None`) and before the matrix is built -- a class present only among
    zero-weight samples never appears as a row/column when `labels=None`
    (an explicit `labels` including that class still gives it a well-defined,
    all-zero row/column, since explicit `labels` never depends on weights at
    all). If every sample has zero weight and `labels=None`, there is no
    effective class to infer from, so this raises `ValueError` -- pass an
    explicit `labels` instead (which then legitimately gives an all-zero
    matrix, per the empty-input case above).

    Matrix cells are computed by grouping every effective sample landing in
    the same cell and summing that group's weights exactly once via
    `math.fsum` over the weights sorted first -- not a plain, order-dependent
    running sum (verified directly that both `np.bincount(..., weights=...)`
    and a hand-rolled sequential sum can give a different final cell value
    depending on the order same-cell samples happen to appear in, for an
    extreme weight ratio), so permuting the input (or the order of samples
    within one cell) never changes the result. A cell whose weights sum to
    more than `float64` can represent raises `ValueError` -- this does not,
    by itself, require every row/column/total sum to also be representable;
    a matrix with several individually-finite, huge cells is still a
    well-defined result here (`classification_metrics`/
    `classification_metrics_from_confusion_matrix` are where a
    non-representable row/column/total sum is actually rejected, since only
    they need to reduce the matrix further).

    The returned matrix is always shape `(len(labels), len(labels))`, a new
    array with no aliasing to `y_true`/`y_pred`, and read-only. Classes
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
        If `y_true`/`y_pred`/`labels`/`sample_weight` is not a `Sequence` or
        an `ndarray` (including `str`/`bytes`/`bytearray`, or an `ndarray`
        with a non-integer dtype), or contains a non-integral element
        (including `bool`/`np.bool_`/`float`/`str`/`None`), or if
        `sample_weight` has a `bool`/complex/object/wider-than-`float64`
        floating dtype, or an element that is not a Python/NumPy int or
        float.
    ValueError
        If `y_true` and `y_pred` have different lengths, if any of
        `y_true`/`y_pred`/`labels`/`sample_weight` is an `ndarray` that is
        not 1-D, if `labels` is empty or contains a duplicate, if a value
        observed in `y_true`/`y_pred` is not in an explicit `labels`, if
        `sample_weight`'s length differs from `y_true`/`y_pred`'s, if a
        `sample_weight` element is negative, NaN, `Inf`, or an integer not
        exactly representable as `float64`, if `labels=None` and either both
        inputs are empty or every effective (positive-weight) sample has
        been removed, if `len(labels) ** 2` is not representable as a dense
        array on this platform, or if a single cell's summed weight is not
        representable as a finite `float64`.
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

    if sample_weight is None:
        if labels is None and len(true_list) == 0:
            raise ValueError(
                "cannot infer labels from empty y_true/y_pred -- pass an explicit labels sequence"
            )
        resolved_labels = _resolve_labels(true_list, pred_list, labels)
        matrix = _build_confusion_matrix(true_list, pred_list, resolved_labels)
        return ConfusionMatrixResult(matrix=matrix, labels=resolved_labels)

    weights = _normalize_nonnegative_weight_sequence(
        sample_weight, len(true_list), allow_empty=(len(true_list) == 0)
    )

    if labels is not None:
        # Validates every raw y_true/y_pred value against the explicit labels, regardless of
        # weight -- an invalid label is never forgiven merely because its weight is zero.
        resolved_labels = _resolve_labels(true_list, pred_list, labels)
    else:
        effective_mask = weights > 0.0
        if not np.any(effective_mask):
            raise ValueError(
                "cannot infer labels from sample_weight with no positive total weight -- pass "
                "an explicit labels sequence"
            )
        keep = effective_mask.tolist()
        effective_true = [
            value for value, keep_value in zip(true_list, keep, strict=True) if keep_value
        ]
        effective_pred = [
            value for value, keep_value in zip(pred_list, keep, strict=True) if keep_value
        ]
        resolved_labels = tuple(sorted(set(effective_true) | set(effective_pred)))

    matrix = _build_weighted_confusion_matrix(true_list, pred_list, resolved_labels, weights)
    return ConfusionMatrixResult(matrix=matrix, labels=resolved_labels)


def classification_metrics(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_pred: Sequence[int] | npt.NDArray[np.integer],
    *,
    labels: Sequence[int] | npt.NDArray[np.integer] | None = None,
    average: Literal["micro", "macro", "weighted"] | None = None,
    zero_division: float | Literal["nan"] = 0.0,
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
) -> ClassificationMetrics:
    """Compute precision/recall/F1/support (and accuracy) directly from labels.

    Builds the confusion matrix internally (see `confusion_matrix` for the
    exact `y_true`/`y_pred`/`labels`/`sample_weight` contract, which applies
    identically here) and delegates to `classification_metrics_from_confusion_matrix`
    -- see that function for the full `average`/`zero_division` contract, which
    also applies identically here. Unlike `confusion_matrix`, this function
    never accepts empty input, even with an explicit `labels`: `zero_division`
    resolves an undefined value for one class among others, not the complete
    absence of any observation to evaluate at all -- so, unlike
    `confusion_matrix`, an all-zero `sample_weight` is never legal here either
    (there is no equivalent of `confusion_matrix`'s well-defined all-zero
    result): it always raises the same clear `ValueError` about
    `sample_weight`, after this function's own label validation (an invalid
    label is still never forgiven merely because every sample's weight is
    zero).

    `sample_weight=None` (the default) uses the unweighted computation above
    unchanged, bit for bit, giving `int64` `support`; given explicitly, both
    `ConfusionMatrixResult.matrix` (built internally) and the returned
    `support` are `float64`, per `confusion_matrix`'s dtype rule.

    Raises
    ------
    TypeError
        Same as `confusion_matrix`, plus a non-`bool`-rejecting type error
        for `zero_division`.
    ValueError
        Same as `confusion_matrix`, plus an empty `y_true`/`y_pred` (with or
        without explicit `labels`), an all-zero `sample_weight` (unlike
        `confusion_matrix`, always an error here, regardless of `labels`),
        an invalid `average`, or an invalid `zero_division`.
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

    if sample_weight is None:
        resolved_labels = _resolve_labels(true_list, pred_list, labels)
        matrix = _build_confusion_matrix(true_list, pred_list, resolved_labels)
        confusion = ConfusionMatrixResult(matrix=matrix, labels=resolved_labels)
        return _compute_classification_metrics(confusion, average, zero_division_value)

    weights = _normalize_nonnegative_weight_sequence(
        sample_weight, len(true_list), allow_empty=False
    )

    if labels is not None:
        # Validates every raw y_true/y_pred value against the explicit labels, regardless of
        # weight -- an invalid label is never forgiven merely because its weight is zero.
        explicit_resolved_labels = _resolve_labels(true_list, pred_list, labels)
    else:
        explicit_resolved_labels = None

    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must contain at least one positive value")

    if explicit_resolved_labels is not None:
        resolved_labels = explicit_resolved_labels
    else:
        effective_mask = weights > 0.0
        keep = effective_mask.tolist()
        effective_true = [
            value for value, keep_value in zip(true_list, keep, strict=True) if keep_value
        ]
        effective_pred = [
            value for value, keep_value in zip(pred_list, keep, strict=True) if keep_value
        ]
        resolved_labels = tuple(sorted(set(effective_true) | set(effective_pred)))

    matrix = _build_weighted_confusion_matrix(true_list, pred_list, resolved_labels, weights)
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
    `matrix` must be a square, non-empty `ndarray` whose side length equals
    `len(labels)`, and `labels` must be plain (non-`bool`) Python `int`s with
    no duplicates. `matrix` must be *exactly* `int64` or *exactly* `float64`
    -- `float16`/`float32`/`np.longdouble`, a `bool` matrix, or any integer
    dtype other than `int64` are all rejected, never silently cast; cast
    explicitly with `matrix.astype(...)` first if needed. An `int64` matrix
    must be non-negative; a `float64` matrix must be finite (no NaN/Inf) and
    non-negative. A `float64` matrix is always treated as weighted, giving a
    `float64` `support` -- even one holding only whole-number values (e.g.
    hand-built as `np.array([[2.0, 1.0], [3.0, 4.0]])`): dtype, not content,
    decides, mirroring `confusion_matrix`'s own rule for whether
    `sample_weight` was given.

    For each class `i`: `TP_i` is the diagonal entry, `FP_i` is that
    column's sum minus `TP_i`, `FN_i` is that row's sum minus `TP_i`, and
    `support_i` is that row's sum (`TP_i + FN_i`). For an `int64` matrix,
    every count involved (total, per-row, per-column) is verified to fit in
    `int64` before any of the following is computed -- a `ConfusionMatrixResult`
    built by hand from huge counts that would silently wrap around in raw
    `int64` arithmetic raises `ValueError` instead (see
    `_exact_nonnegative_int64_sum`). For a `float64` matrix, every row/column
    sum (and the flat total) is instead computed via `math.fsum` over that
    row/column/the whole matrix sorted first -- not a bare
    `matrix.sum(axis=...)`, which neither guards against overflow nor
    promises an order-independent result -- mapping any resulting
    `OverflowError` to `ValueError` (a matrix with individually-finite cells
    can still have a row/column/total sum that is not representable as a
    finite `float64`; `confusion_matrix` itself does not reject that, but
    this function does, since it must reduce the matrix further). `support`
    is that same canonical row-sum array; `average="weighted"`'s own
    denominator is a further, independent canonical sum of `support` itself
    (not the flat total), since `support` is the documented, public set of
    per-class weights for that average and can legitimately differ from the
    flat total by a rounding residual.

    `precision_i`, `recall_i`, and `f1_i` each have their *own* zero-check,
    computed directly from `TP_i`/`FP_i`/`FN_i` -- not from each other, and
    never by adding two already-`zero_division`-filled values together:

    - `precision_i = TP_i / (TP_i + FP_i)`, using `zero_division` only when
      `TP_i + FP_i == 0` (class `i` was never predicted at all). For a
      `float64` matrix this is `TP_i / column_sum_i` directly (`TP_i <=
      column_sum_i` always, so this never overflows).
    - `recall_i = TP_i / (TP_i + FN_i)`, using `zero_division` only when
      `TP_i + FN_i == 0` (class `i` never occurs in the true labels at all).
      For a `float64` matrix this is `TP_i / row_sum_i` directly, for the
      same reason.
    - `f1_i = 2 TP_i / (2 TP_i + FP_i + FN_i)`, using `zero_division` only
      when `2 TP_i + FP_i + FN_i == 0` (class `i` has no true positives, no
      false positives, and no false negatives -- i.e. it is completely
      absent from both `y_true` and `y_pred`). For a `float64` matrix, both
      `2 * TP_i` and `row_sum_i + column_sum_i` can independently overflow
      even when `precision_i`/`recall_i` do not (two independently-bounded
      sums added together) -- computed through a private helper that matches
      plain division bit for bit except at the entries that would actually
      overflow (verified directly: `TP_i = row_sum_i = column_sum_i =
      np.finfo(np.float64).max` gives exactly `1.0`, not a silently
      overflowed `0.0`).

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
    always `trace(matrix) / total` (the diagonal and total each computed
    via the same canonical summation as above for a `float64` matrix),
    independent of `average`. A `float64` matrix's legal, correctly-rounded
    underflow anywhere in this arithmetic (e.g. a tiny `TP_i` divided by a
    huge `row_sum_i`) never raises or warns, regardless of the caller's own
    `np.seterr`/`np.errstate` configuration.

    `average=None` returns per-class `precision`/`recall`/`f1` as read-only
    `float64` arrays aligned with `confusion.labels`, always including
    every declared class (even one with zero support). `average="micro"`
    sums `TP`/`FP`/`FN` across all classes before dividing (for this
    single-label multiclass case, the three micro values are always
    numerically equal to `accuracy` -- documented, not special-cased: this
    function does not shortcut to returning `accuracy` directly for
    `average="micro"`, it always goes through the same sum-then-divide
    computation; for a `float64` matrix this equality holds only up to
    `float64` rounding, not bit for bit, since the two are computed through
    genuinely different summation paths). `average="macro"` is the
    unweighted mean over every class in `confusion.labels`, including
    zero-support classes (each contributing its own `zero_division` value to
    the mean). `average="weighted"` is the mean weighted by `support`; for
    this same single-label multiclass case, weighted recall is always
    numerically equal to `accuracy` (also not special-cased, and also only
    up to rounding for a `float64` matrix).

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
        not an `ndarray` or not exactly `int64`/`float64`, if any of its
        `labels` is not a plain (non-`bool`) `int`, or `zero_division` is a
        `bool`.
    ValueError
        If `confusion`'s `matrix`/`labels` are inconsistent or invalid
        (wrong ndim, not square, empty, negative values, non-finite values
        for a `float64` matrix, mismatched lengths, duplicate labels, a
        total weight of zero, an `int64` total that exceeds what fits in
        `int64`, or a `float64` row/column/total sum that is not
        representable as a finite `float64`), or `average`/`zero_division`
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
    if confusion.matrix.dtype != np.int64:
        return _compute_weighted_classification_metrics(confusion, average, zero_division)

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


def _canonical_axis_sum(matrix: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    """Sum each row (`axis=1`) or column (`axis=0`) of a weighted (`float64`) confusion matrix
    via `math.fsum` over that row/column's values sorted first.

    Not `matrix.sum(axis=axis, dtype=np.float64)`: that neither guards against overflow (would
    silently return `inf` or warn, depending on the caller's own `np.seterr`) nor promises a
    summation order independent of the matrix's own memory layout. `math.fsum` over a sorted copy
    gives the same, correctly-rounded total regardless of layout, mirroring the same strategy
    already used to build the matrix itself (`_build_weighted_confusion_matrix`). An
    `OverflowError` (a row/column summing past `float64`'s range) is mapped to `ValueError`.
    """
    source = matrix if axis == 1 else matrix.T
    result = np.empty(source.shape[0], dtype=np.float64)
    try:
        for index in range(source.shape[0]):
            result[index] = math.fsum(sorted(source[index].tolist()))
    except OverflowError as exc:
        kind = "row" if axis == 1 else "column"
        raise ValueError(
            f"confusion matrix {kind} sum is not representable as a finite float64"
        ) from exc
    return result


def _canonical_flat_sum(values: npt.NDArray[np.float64], description: str) -> float:
    """`math.fsum` over `values`'s elements sorted first -- the same canonical, order-independent
    summation strategy as `_canonical_axis_sum`, applied to any 1-D or flattened `float64` array
    (the whole matrix, its diagonal, or `support`). An `OverflowError` is mapped to `ValueError`
    with `description` naming what could not be represented.
    """
    try:
        return math.fsum(sorted(values.reshape(-1).tolist()))
    except OverflowError as exc:
        raise ValueError(f"{description} is not representable as a finite float64") from exc


def _compute_weighted_classification_metrics(
    confusion: ConfusionMatrixResult,
    average: Literal["micro", "macro", "weighted"] | None,
    zero_division: float,
) -> ClassificationMetrics:
    """The weighted (`float64` matrix) counterpart of `_compute_classification_metrics`'s
    `int64` branch -- a fully separate code path, not a dtype branch threaded through the
    existing one, so the `int64` branch stays exactly as it was before `sample_weight` existed.

    `row_sum`/`col_sum`/the flat total/the diagonal total are each computed once via
    `_canonical_axis_sum`/`_canonical_flat_sum` (canonical `math.fsum`-based summation, mapping
    any `OverflowError` to `ValueError`) -- never a bare `matrix.sum(axis=...)`. The "at least one
    observation" gate uses the flat total (mirroring the `int64` branch's `total_samples == 0`
    check); `average="weighted"`'s own denominator is instead the canonical sum of the *returned*
    `support` array specifically (which is `row_sum`) -- support is the documented, public set of
    per-class weights for that average, and summing it again independently (rather than reusing
    the flat total) can legitimately differ from the flat total by a rounding residual, since the
    two are different sums over different-sized inputs; using the flat total there instead could
    silently disagree with the very weights the caller can see in `result.support`.

    `precision_i = TP_i / col_sum_i` and `recall_i = TP_i / row_sum_i` are computed directly
    (`TP_i <= col_sum_i`/`row_sum_i` always, by construction, so this never overflows -- only
    legitimately underflows, isolated via `np.errstate(under="ignore")`, mirroring the ranking
    core's identical fix). `f1_i = 2 * TP_i / (row_sum_i + col_sum_i)` can overflow even though
    `precision_i`/`recall_i` cannot (two independently-bounded sums added together) -- computed
    via `_f1_ratio`, which matches direct division bit for bit except at the entries that would
    actually overflow.
    """
    # Only ever called (via _compute_classification_metrics's dtype dispatch) when
    # confusion.matrix.dtype is already known, at runtime, to be float64 -- cast narrows the
    # static type accordingly, since ConfusionMatrixResult.matrix itself is a union.
    matrix = cast("npt.NDArray[np.float64]", confusion.matrix)
    labels = confusion.labels
    n_classes = matrix.shape[0]

    total_weight = _canonical_flat_sum(matrix, "confusion matrix total weight")
    if total_weight <= 0.0:
        raise ValueError(
            "classification_metrics_from_confusion_matrix requires positive total weight "
            "(confusion.matrix sums to zero)"
        )

    row_sum = _canonical_axis_sum(matrix, axis=1)
    col_sum = _canonical_axis_sum(matrix, axis=0)
    tp = np.diagonal(matrix).astype(np.float64, copy=True)
    fp = col_sum - tp
    fn = row_sum - tp
    if np.any(tp < 0) or np.any(fp < 0) or np.any(fn < 0):
        raise RuntimeError("internal error: TP/FP/FN must not be negative")
    support = row_sum.copy()
    support.flags.writeable = False

    diagonal_weight = _canonical_flat_sum(tp, "confusion matrix diagonal weight")
    with np.errstate(under="ignore"):
        accuracy = float(diagonal_weight / total_weight)

    with np.errstate(under="ignore"):
        precision_per_class = _safe_divide_array(tp, col_sum, zero_division)
        recall_per_class = _safe_divide_array(tp, row_sum, zero_division)
    f1_per_class = _f1_ratio_with_zero_division(tp, row_sum, col_sum, zero_division)

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
        tp_sum = _canonical_flat_sum(tp, "confusion matrix diagonal weight")
        fp_sum = _canonical_flat_sum(fp, "confusion matrix false-positive weight")
        fn_sum = _canonical_flat_sum(fn, "confusion matrix false-negative weight")
        with np.errstate(over="ignore"):
            pred_sum_total = tp_sum + fp_sum
            true_sum_total = tp_sum + fn_sum
        if not (math.isfinite(pred_sum_total) and math.isfinite(true_sum_total)):
            raise ValueError(
                "average='micro' cannot be computed: TP+FP or TP+FN is not representable as a "
                "finite float64"
            )
        with np.errstate(under="ignore"):
            precision = _safe_divide_scalar(tp_sum, pred_sum_total, zero_division)
            recall = _safe_divide_scalar(tp_sum, true_sum_total, zero_division)
        f1 = float(
            _f1_ratio_with_zero_division(
                np.array([tp_sum]),
                np.array([true_sum_total]),
                np.array([pred_sum_total]),
                zero_division,
            )[0]
        )
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
        with np.errstate(under="ignore"):
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
        weighted_total = _canonical_flat_sum(support, "confusion matrix support total")
        with np.errstate(under="ignore"):
            result = ClassificationMetrics(
                labels=labels,
                precision=_weighted_average(precision_per_class, support, weighted_total),
                recall=_weighted_average(recall_per_class, support, weighted_total),
                f1=_weighted_average(f1_per_class, support, weighted_total),
                support=support,
                accuracy=accuracy,
                average="weighted",
            )

    _check_weighted_metrics_postconditions(result, n_classes, total_weight)
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


def _stable_f1_ratio(
    tp: npt.NDArray[np.float64], row_sum: npt.NDArray[np.float64], col_sum: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute `2 * tp / (row_sum + col_sum)` elementwise without an intermediate
    `row_sum + col_sum` overflow (or `2 * tp` overflow).

    Only ever called (by `_f1_ratio`, on the subset of entries that actually overflow) where
    `row_sum + col_sum > 0` always holds by construction. Plain `2 * tp / (row_sum + col_sum)`
    silently returns `0.0` instead of the correct `1.0` when `tp == row_sum == col_sum ==
    np.finfo(np.float64).max` (both `2 * tp` and `row_sum + col_sum` overflow to `+inf`) --
    verified directly. Scaling every operand by `max(row_sum, col_sum)` first keeps every
    intermediate value within `float64`'s range whenever the true ratio is well-defined, which it
    always is here (mirrors `_stable_precision_ratio`'s identical strategy, extended to F1's
    three-term formula). Division here can legitimately underflow to `0.0` -- isolated via
    `np.errstate(under="ignore")`, for the same reason `_stable_precision_ratio` is.
    """
    scale = np.maximum(row_sum, col_sum)
    with np.errstate(under="ignore"):
        scaled_tp = tp / scale
        scaled_row_sum = row_sum / scale
        scaled_col_sum = col_sum / scale
        return (2.0 * scaled_tp) / (scaled_row_sum + scaled_col_sum)


def _f1_ratio(
    tp: npt.NDArray[np.float64], row_sum: npt.NDArray[np.float64], col_sum: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute `2 * tp / (row_sum + col_sum)` elementwise, matching plain `float64` arithmetic bit
    for bit whenever neither `2 * tp` nor `row_sum + col_sum` overflows -- only the (rare)
    overflowing entries are routed through `_stable_f1_ratio`'s scale-based formula instead, for
    the same reason `_precision_ratio` only reroutes its own overflowing entries (a single global
    fallback would change every entry's bit pattern, not just the overflowing one's).

    Both `2 * tp` and `row_sum + col_sum` are computed under `np.errstate(over="ignore")`: the
    overflow this deliberately allows (only to detect via `np.isinf` immediately after) would
    otherwise raise or warn. The ordinary division can legitimately underflow -- isolated via
    `np.errstate(under="ignore")`, mirroring `_precision_ratio`. `invalid` is deliberately left at
    the caller's own sensitivity: a NaN anywhere in `tp`/`row_sum`/`col_sum` would mean an
    internal error upstream, not a case this function should paper over.
    """
    with np.errstate(over="ignore"):
        doubled_tp = 2.0 * tp
        combined = row_sum + col_sum
    overflowed = np.isinf(doubled_tp) | np.isinf(combined)
    with np.errstate(under="ignore"):
        if not np.any(overflowed):
            return doubled_tp / combined

        result = np.empty_like(tp)
        ordinary = ~overflowed
        result[ordinary] = doubled_tp[ordinary] / combined[ordinary]
    result[overflowed] = _stable_f1_ratio(tp[overflowed], row_sum[overflowed], col_sum[overflowed])
    return result


def _f1_ratio_with_zero_division(
    tp: npt.NDArray[np.float64],
    row_sum: npt.NDArray[np.float64],
    col_sum: npt.NDArray[np.float64],
    zero_division: float,
) -> npt.NDArray[np.float64]:
    """`_f1_ratio`, but returning `zero_division` for any entry where `row_sum == col_sum == 0`
    (class `i` is completely absent from both `y_true` and `y_pred`) instead of computing a
    `0/0`-shaped ratio.

    The zero-check mask (`row_sum > 0` or `col_sum > 0`) never adds `row_sum` and `col_sum`
    together, unlike `_f1_ratio`'s own combined-overflow detection -- so building this mask can
    never itself overflow or warn, even before `_f1_ratio` is invoked on the (always nonzero)
    entries that need it.
    """
    result = np.full(tp.shape, zero_division, dtype=np.float64)
    nonzero_mask = (row_sum > 0.0) | (col_sum > 0.0)
    if np.any(nonzero_mask):
        result[nonzero_mask] = _f1_ratio(
            tp[nonzero_mask], row_sum[nonzero_mask], col_sum[nonzero_mask]
        )
    return result


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


def _build_weighted_confusion_matrix(
    true_list: list[int],
    pred_list: list[int],
    labels: tuple[int, ...],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The weighted counterpart of `_build_confusion_matrix`, used only when `sample_weight` is
    not `None`.

    Every value in `true_list`/`pred_list` is already guaranteed present in `index_of` by the
    time this is called: for explicit `labels`, every raw (unfiltered) value was already
    validated against them; for inferred `labels`, `labels` was built from exactly the effective
    (positive-weight) subset this function also filters to. A zero-weight sample contributes
    nothing and is filtered out before any indexing happens -- never mapped through `index_of` at
    all, so a class present only among zero-weight samples never needs an entry there.

    Builds the flat cell index only for the effective subset, then groups same-cell entries and
    sums each group's weights exactly once via `math.fsum` over that group's weights sorted
    ascending first -- not `np.bincount(flat, weights=weights)` or `np.add.at`, both verified
    directly to be order-dependent for extreme weight ratios (`np.bincount(flat, weights=
    [1e16, 1.0, 1.0])` gives a different final cell value than `np.bincount(flat, weights=
    [1.0, 1.0, 1e16])` for three samples landing in the same cell) -- which would make this
    module's existing "permuting the input never changes the result" expectation depend on
    summation order. `math.fsum(sorted(...))` gives the same, correctly-rounded total regardless
    of input order, mirroring `_compute_weighted_ranking_core`'s identical strategy.

    An `OverflowError` from `math.fsum` (a single cell's weights summing past `float64`'s range)
    is mapped to `ValueError`. No total/row/column sum is checked here -- see
    `_check_weighted_matrix_postconditions` and this function's callers for why that is
    deliberately deferred to `classification_metrics`/`classification_metrics_from_confusion_
    matrix` instead.
    """
    n_classes = len(labels)
    _check_allocation_representable(n_classes)
    index_of = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    effective_mask = weights > 0.0

    if np.any(effective_mask):
        effective_true = [
            t for t, keep in zip(true_list, effective_mask.tolist(), strict=True) if keep
        ]
        effective_pred = [
            p for p, keep in zip(pred_list, effective_mask.tolist(), strict=True) if keep
        ]
        effective_weight = weights[effective_mask]

        true_idx = np.fromiter(
            (index_of[value] for value in effective_true), dtype=np.intp, count=len(effective_true)
        )
        pred_idx = np.fromiter(
            (index_of[value] for value in effective_pred), dtype=np.intp, count=len(effective_pred)
        )
        flat = true_idx * n_classes + pred_idx

        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        sorted_weight = effective_weight[order]

        group_end_mask = np.concatenate((sorted_flat[1:] != sorted_flat[:-1], np.array([True])))
        group_end_indices = np.flatnonzero(group_end_mask)
        group_start_indices = np.empty_like(group_end_indices)
        group_start_indices[0] = 0
        group_start_indices[1:] = group_end_indices[:-1] + 1

        flat_matrix = matrix.reshape(-1)
        try:
            for group_index in range(group_end_indices.shape[0]):
                start = int(group_start_indices[group_index])
                end = int(group_end_indices[group_index]) + 1
                cell = int(sorted_flat[start])
                flat_matrix[cell] = math.fsum(sorted(sorted_weight[start:end].tolist()))
        except OverflowError as exc:
            raise ValueError(
                "sample_weight sum in a single confusion-matrix cell is not representable as a "
                "finite float64"
            ) from exc

    matrix.flags.writeable = False
    _check_weighted_matrix_postconditions(matrix, n_classes)
    return matrix


def _check_weighted_matrix_postconditions(matrix: np.ndarray, n_classes: int) -> None:
    """Postconditions for a weighted (`float64`) confusion matrix -- deliberately separate from
    `_check_matrix_postconditions` (`int64`-only) rather than adding dtype branches to it.

    Unlike the `int64` postconditions, this never checks that the matrix's total sums to the
    expected sample count: a `float64` matrix with several individually-finite, huge cells is
    still a well-defined, useful result on its own (e.g. for inspection or serialization) even
    when its total is not representable as a finite `float64` -- rejecting *that* is
    `classification_metrics`/`classification_metrics_from_confusion_matrix`'s job (see
    `_compute_weighted_classification_metrics`), not `confusion_matrix`'s.
    """
    if matrix.dtype != np.float64:
        raise RuntimeError(
            f"internal error: confusion matrix has dtype {matrix.dtype}, expected float64"
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(
            f"internal error: confusion matrix has shape {matrix.shape}, expected square"
        )
    if matrix.shape[0] != n_classes:
        raise RuntimeError(
            f"internal error: confusion matrix has {matrix.shape[0]} rows, expected {n_classes}"
        )
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("internal error: confusion matrix contains a non-finite value")
    if np.any(matrix < 0):
        raise RuntimeError("internal error: confusion matrix contains a negative value")
    if matrix.flags.writeable:
        raise RuntimeError("internal error: confusion matrix is writeable")


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

    if matrix.dtype == np.int64:
        if np.any(matrix < 0):
            raise ValueError("confusion.matrix must not contain negative counts")
    elif matrix.dtype == np.float64:
        # A float64 matrix is always treated as weighted, regardless of whether its values
        # happen to be whole numbers -- dtype, not content, decides (mirrors confusion_matrix's
        # own sample_weight-is-given-or-not dtype rule). Never silently cast any other dtype
        # (float16/float32/longdouble, or any integer dtype other than int64) -- exactly int64 or
        # exactly float64 are the only two accepted matrix dtypes.
        if not np.all(np.isfinite(matrix)):
            raise ValueError("confusion.matrix must contain only finite values (no NaN or Inf)")
        if np.any(matrix < 0):
            raise ValueError("confusion.matrix must not contain negative values")
    else:
        raise TypeError(f"confusion.matrix must have dtype int64 or float64, got {matrix.dtype}")

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


def _check_weighted_metrics_postconditions(
    result: ClassificationMetrics, n_classes: int, total_weight: float
) -> None:
    """The weighted (`float64` `support`) counterpart of `_check_metrics_postconditions`.

    Unlike that function, does not check `support`'s sum against an expected total: `support`
    (the canonical per-row sum) and `total_weight` (the canonical flat sum over the whole matrix)
    are computed via two independently-rounded `math.fsum` calls over different-shaped inputs, so
    -- unlike the `int64` case, where both are exact integers and must match precisely -- they
    can legitimately differ by a rounding residual here. `total_weight` is accepted as a parameter
    only to keep this function's signature symmetric with `_check_metrics_postconditions`; it is
    intentionally unused for any exact-match assertion.
    """
    del total_weight
    if not (isinstance(result.accuracy, float) and math.isfinite(result.accuracy)):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not a finite float")
    if not (0.0 <= result.accuracy <= 1.0):
        raise RuntimeError(f"internal error: accuracy {result.accuracy!r} is not in [0, 1]")

    if result.support.shape != (n_classes,) or result.support.dtype != np.float64:
        raise RuntimeError("internal error: support has unexpected shape/dtype")
    if result.support.flags.writeable:
        raise RuntimeError("internal error: support is writeable")
    if np.any(result.support < 0):
        raise RuntimeError("internal error: support contains a negative value")
    if not np.all(np.isfinite(result.support)):
        raise RuntimeError("internal error: support contains a non-finite value")

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
    `precision` equals the overall positive prevalence -- `P / len(y_true)` without
    `sample_weight`, or the weighted prevalence (total effective positive weight divided by
    total effective weight) when `sample_weight` is given (predicting everything positive). All
    three arrays are new, independent, read-only `float64` arrays -- never views of
    `y_true`/`y_score` or of each other.

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
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
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

    `sample_weight=None` (the default) uses the unweighted computation above unchanged, bit for
    bit. Given explicitly, `sample_weight` must be a `Sequence`/1-D `ndarray` the same length as
    `y_true`/`y_score`, holding the same accepted numeric types as `y_score` (Python/NumPy
    int/float, or `float16`/`float32`/`float64`), but non-negative and with at least one positive
    value -- NaN/Inf, negative values (which would otherwise make `true_positive_rate` a
    non-monotonic function of the threshold, verified directly against a reference
    implementation that permits them), and an integer not exactly representable as `float64` are
    all rejected the same way they are for `y_score`. A sample with `sample_weight == 0.0` is
    removed entirely before thresholds are built: a score that appears only at zero weight never
    produces a threshold. `TP(t)`/`FP(t)` become the *sum of weights* of the effective positive/
    negative samples with `score >= t` (instead of a plain count); `true_positive_rate`/
    `false_positive_rate` divide those sums by the total effective positive/negative weight.
    Requires positive total effective weight for both classes -- a class present only among
    zero-weight samples is treated the same as a class that is entirely absent. An extreme
    `sample_weight` dynamic range (e.g. one sample weighing `1e16` and another `1.0`) can make a
    later, individually-positive-weight group fail to change the cumulative `float64` sum at
    all; this is detected explicitly and raises `ValueError` rather than silently dropping that
    group's contribution to the curve. This is distinct from a rate that legitimately rounds to
    `0.0` through ordinary `float64` underflow (e.g. one weight many orders of magnitude smaller
    than another) -- that is a correctly-rounded result, not an error, and never raises or warns
    regardless of the caller's own `np.seterr`/`np.errstate` configuration.

    Raises
    ------
    TypeError
        If `y_true` is not a `Sequence`/1-D integer `ndarray`, if any of its elements is not
        integral (including `bool`/`np.bool_`), if `positive_label` is not a Python/NumPy
        integral value (including `bool`/`np.bool_`), if `y_score`/`sample_weight` is not a
        `Sequence`/1-D float-or-integer `ndarray`, if `y_score`/`sample_weight` has a
        `bool`/complex/object/wider-than-`float64` floating dtype, or if a `y_score`/
        `sample_weight` element is not a Python/NumPy int or float.
    ValueError
        If `y_true`/`y_score` is empty, if `y_true`/`y_score`/`sample_weight` lengths differ, if
        any `y_true`/`y_score`/`sample_weight` `ndarray` is not 1-D, if a `y_score`/
        `sample_weight` element is NaN/`Inf` (or an integer not exactly representable as
        `float64`), if a `sample_weight` element is negative or every element is zero, if
        `y_true` has no sample equal to `positive_label` (after removing zero-weight samples,
        when `sample_weight` is given), if `y_true` has no sample different from
        `positive_label` (likewise), or if `sample_weight`'s dynamic range is too large to
        preserve every threshold's contribution in `float64`.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    if sample_weight is None:
        core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=True)
        false_positive_rate = core.false_positives.astype(np.float64) / core.n_negative
        true_positive_rate = core.true_positives.astype(np.float64) / core.n_positive
        thresholds = core.thresholds
        result_positive_label = core.positive_label
    else:
        weighted_core = _compute_weighted_ranking_core(
            y_true, y_score, sample_weight, positive_label, require_negative=True
        )
        false_positive_rate, true_positive_rate = _compute_weighted_roc_rates(weighted_core)
        thresholds = weighted_core.thresholds
        result_positive_label = weighted_core.positive_label

    false_positive_rate.flags.writeable = False
    true_positive_rate.flags.writeable = False
    thresholds.flags.writeable = False

    result = RocCurve(
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        thresholds=thresholds,
        positive_label=result_positive_label,
    )
    _check_roc_curve_postconditions(result)
    return result


def precision_recall_curve(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
) -> PrecisionRecallCurve:
    """Compute a binary, one-vs-rest precision-recall curve at each distinct score threshold.

    See `roc_curve` for the shared one-vs-rest/`positive_label`/`y_score`/`sample_weight`/
    tie-aggregation contract, which applies identically here. Requires at least one (effective)
    positive sample; unlike `roc_curve`, a `y_true` with no (effective) negative sample is legal
    (precision is then `1.0` at every real threshold, since there is never a false positive to
    count) -- a deliberate departure from `sklearn.metrics.precision_recall_curve`'s own
    length/ordering conventions, made so that `precision`/`recall`/`thresholds` always share one
    length (`K + 1` for `K` distinct scores) and the same descending-threshold/ascending-recall
    pairing as `roc_curve`.

    `thresholds[0]` is `+inf`, giving the synthetic starting point
    `(precision, recall) = (1.0, 0.0)` with no corresponding real threshold. `recall` is
    non-decreasing and its final value is `1.0`; the final `precision` is the overall positive
    prevalence (predicting everything positive) -- `P / len(y_true)` without `sample_weight`, or
    the weighted prevalence when `sample_weight` is given. `precision` matches a direct
    `tp / (tp + fp)` bit for bit at every threshold where that division would not itself
    overflow; only a threshold where both the effective positive and negative weight are
    individually huge enough that `tp + fp` would overflow (near `float64`'s max) instead uses a
    scale-based division there, giving the mathematically correct ratio rather than silently
    underflowing to `0.0`.

    Raises
    ------
    TypeError
        Same as `roc_curve`.
    ValueError
        Same as `roc_curve`, except a `y_true` with no (effective) negative sample is accepted
        rather than rejected.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    if sample_weight is None:
        core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=False)
        precision, recall = _compute_precision_recall_arrays(core)
        thresholds = core.thresholds
        result_positive_label = core.positive_label
    else:
        weighted_core = _compute_weighted_ranking_core(
            y_true, y_score, sample_weight, positive_label, require_negative=False
        )
        precision, recall = _compute_weighted_precision_recall_arrays(weighted_core)
        thresholds = weighted_core.thresholds
        result_positive_label = weighted_core.positive_label

    precision.flags.writeable = False
    recall.flags.writeable = False
    thresholds.flags.writeable = False

    result = PrecisionRecallCurve(
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        positive_label=result_positive_label,
    )
    _check_pr_curve_postconditions(result)
    return result


def average_precision_score(
    y_true: Sequence[int] | npt.NDArray[np.integer],
    y_score: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    *,
    positive_label: int,
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
) -> float:
    """Compute binary, one-vs-rest average precision: a non-interpolated ranking-quality score.

    This is classification ranking average precision, not object-detection AP or mAP (which
    additionally require matching predictions to ground truth by IoU) -- see `roc_curve` for
    the shared one-vs-rest/`positive_label`/`y_score`/`sample_weight`/tie-aggregation contract,
    which applies identically here (same input and error contract as `precision_recall_curve`,
    including that a `y_true` with no effective negative sample is legal).

    Defined as the weighted mean of precision, using each recall increment as its weight:
    `AP = sum((recall[i] - recall[i - 1]) * precision[i] for i in 1..K)`, over the same `K + 1`
    grouped-threshold points `precision_recall_curve` would return (`precision[i]` is always
    taken from the *right* end of each recall increment, never `precision[i - 1]`) -- this is
    not linear interpolation and not the trapezoidal area under the PR curve, which is a
    distinct quantity that this function does not compute: depending on the shape of the curve
    and its ties, the trapezoidal area can be larger or smaller than this average precision,
    never consistently one or the other. `sample_weight` (see `roc_curve` for the full contract)
    changes `recall`/`precision` the same way it does in `precision_recall_curve`, so, unlike the
    unweighted case, two samples with different weights no longer contribute equally to this sum
    even when their score ranks them identically.

    A perfectly reversed ranking does not give `0.0` -- unlike ROC AUC, average precision has no
    symmetric complement relation (`average_precision_score` of the negated scores is not
    `1 - average_precision_score`). A `y_true` with no effective negative sample gives exactly
    `1.0` (returned directly, not computed through a `0/0`-adjacent division, and not by summing
    `precision_recall_curve`'s own recall increments, which -- for weighted input -- can differ
    from `1.0` by a rounding residual even when every recall increment's underlying ratio is
    exactly `1.0`); constant scores (no discriminative power) give exactly the positive
    prevalence (`P / len(y_true)` without `sample_weight`, the weighted prevalence with it) --
    these are the only two inputs with a closed-form result documented here; no other ranking's
    average precision is claimed to equal prevalence. A weighted recall increment or precision
    value that legitimately rounds to `0.0`/underflows during this sum never raises or warns,
    regardless of the caller's own `np.seterr`/`np.errstate` configuration.

    Raises
    ------
    TypeError
        Same as `roc_curve`.
    ValueError
        Same as `roc_curve`, except a `y_true` with no effective negative sample is accepted
        rather than rejected (same relaxation as `precision_recall_curve`).
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    if sample_weight is None:
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
    else:
        weighted_core = _compute_weighted_ranking_core(
            y_true, y_score, sample_weight, positive_label, require_negative=False
        )
        if weighted_core.total_negative_weight == 0.0:
            result = 1.0
        else:
            precision, recall = _compute_weighted_precision_recall_arrays(weighted_core)
            # A tiny recall increment times a small precision (e.g. a recall increment near
            # np.nextafter(0.0, 1.0)) can legitimately underflow to a correctly-rounded 0.0
            # contribution -- not an error that should depend on the caller's own np.seterr
            # configuration (verified directly: np.seterr(all="raise") previously turned this
            # legal underflow into an uncaught FloatingPointError).
            with np.errstate(under="ignore"):
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
    sample_weight: Sequence[float]
    | npt.NDArray[np.floating]
    | npt.NDArray[np.integer]
    | None = None,
) -> float:
    """Compute the area under the binary, one-vs-rest ROC curve.

    Uses the same false/true positive rate points `roc_curve` would return (see `roc_curve` for
    the full `positive_label`/`y_score`/`sample_weight`/tie-aggregation/error contract, which
    applies identically here), integrated with the trapezoidal rule -- equivalent to the
    probability that a uniformly random positive sample outranks a uniformly random negative
    sample, with a tied pair counted as one-half (weighted by `sample_weight` when given). Does
    not call `np.trapz` or `np.trapezoid`: neither name exists across the full range of NumPy
    versions this project supports (verified directly: `np.trapz` is gone in current NumPy,
    `np.trapezoid` does not exist on this project's NumPy floor), so the trapezoidal sum is
    computed directly from basic array arithmetic instead. Does not call the public `roc_curve`
    internally (even though it shares that function's inputs and error contract exactly) --
    both functions build their false/true positive rate arrays from the same private ranking
    core, so `roc_auc_score(..., sample_weight=w)` always equals
    `auc(*the equivalent roc_curve(..., sample_weight=w)'s rate arrays*)` bit for bit, without
    paying for `RocCurve`'s own allocation/postcondition overhead here.

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
    if sample_weight is None:
        core = _compute_ranking_core(y_true, y_score, positive_label, require_negative=True)
        false_positive_rate = core.false_positives.astype(np.float64) / core.n_negative
        true_positive_rate = core.true_positives.astype(np.float64) / core.n_positive
    else:
        weighted_core = _compute_weighted_ranking_core(
            y_true, y_score, sample_weight, positive_label, require_negative=True
        )
        false_positive_rate, true_positive_rate = _compute_weighted_roc_rates(weighted_core)

    area = _trapezoidal_area(false_positive_rate, true_positive_rate)
    _check_roc_auc_postconditions(area)
    return area


def auc(
    x: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
    y: Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer],
) -> float:
    """Compute the area under a curve given as `(x, y)` points, using the trapezoidal rule.

    A general-purpose geometric helper with no ranking semantics of its own -- unlike
    `roc_auc_score`/`average_precision_score`, it has no `positive_label`, no tie-aggregation,
    and no notion of "positive"/"negative" samples. `x` must be either non-decreasing or
    non-increasing throughout (duplicate `x` values are legal in either direction and
    contribute a zero-width segment; constant `x` is legal and gives exactly `0.0`); a
    non-increasing `x` gives the same positive geometric area as the equivalent non-decreasing
    order would -- this is not a signed integral, so the result never depends on which
    direction a monotonic `x` happens to be given in. `y` may be negative, and so may the
    result: there is no `[0, 1]` bound here the way there is for `roc_auc_score`/
    `average_precision_score`, whose `x`/`y` are always ratios in `[0, 1]` by construction.

    `x`/`y` accept the same containers/dtypes as `roc_curve`'s `y_score` (a `Sequence` of
    Python `int`/`float` or NumPy `float16`/`float32`/`float64` scalars, or a 1-D `ndarray` with
    an integer or `float16`/`float32`/`float64` dtype) -- a generator/iterator, a 2-D (including
    a column-vector-shaped `(n, 1)` array -- deliberately not silently raveled) or 0-D array, a
    `bool`/complex/object/wider-than-`float64` floating dtype, NaN/Inf, and an integer not
    exactly representable as `float64` are all rejected, for the same reasons `y_score` rejects
    them: consistency and no silent precision loss anywhere in this module, even though `x`/`y`
    here are curve coordinates, not ranking scores.

    This function computes the trapezoidal area under the precision-recall curve when called
    as `auc(curve.recall, curve.precision)` for a `curve` from `precision_recall_curve` -- this
    is a distinct quantity from `average_precision_score` (a non-interpolated, non-trapezoidal
    definition; see that function's docstring), and this module deliberately does not expose
    the trapezoidal PR-curve area under its own name: `auc(curve.recall, curve.precision)` is
    the complete, supported way to compute it.

    Ordinary calls -- `y` entirely non-negative or entirely non-positive, with no intermediate
    overflow/underflow -- use a fast, canonical `float64` summation (this is the path
    `roc_auc_score`'s always-non-negative TPR and `auc(curve.recall, curve.precision)`'s
    always-non-negative precision both take). This falls back to computing the exact
    trapezoidal sum as a rational number (`fractions.Fraction`, standard library, used only on
    these rare paths) over the already-normalized `float64` values, then converts only the final
    total to `float`, in three situations: (1) an intermediate segment width, height sum, or
    product would overflow `float64`; (2) a segment's own contribution would underflow in a way
    that could lose it before it has a chance to be summed with its neighbors; or (3) `y`
    contains both a positive and a negative value, since opposite-signed contributions can
    cancel in the final sum in a way no NumPy overflow/underflow/invalid signal would ever catch.
    This correctly preserves cancellation between huge intermediate contributions (e.g. two
    segments whose true areas are individually far larger than `float64`'s range but whose sum
    is exactly representable), an accumulated subnormal residual (e.g. several segments that
    each round to `0.0` on their own, but whose exact total is a representable positive
    subnormal `float64`), and a residual from cancellation between oppositely-signed
    contributions (e.g. `y=[1.0, 1e-20, -1.0]`, whose exact area is `1e-20` but whose plain
    `float64` segment contributions of `0.5` and `-0.5` would otherwise cancel to `0.0`) -- all
    cases a fast, per-segment `float64` computation could otherwise lose. A contribution whose
    exact value genuinely is closer to `0.0` than to the smallest positive subnormal `float64`
    still legitimately rounds to `0.0` -- the exact fallback decides this correctly rather than
    the fast path guessing. Only an input whose true, exact area is not representable as a
    finite `float64` raises `ValueError`. No NumPy warning is ever emitted for an accepted input
    regardless of the caller's own `np.seterr`/`np.errstate` configuration, and this never
    silently returns `inf`/`-inf`/`NaN`.

    Raises
    ------
    TypeError
        If `x`/`y` is not a `Sequence`/1-D `ndarray`, if an element/dtype is `bool`, complex,
        `object`, or a floating type wider than `float64`, or if an element is not a
        Python/NumPy int or float.
    ValueError
        If `x`/`y` is empty, if either `ndarray` is not 1-D, if an element is NaN/Inf (or an
        integer not exactly representable as `float64`), if `x` and `y` have different lengths,
        if fewer than 2 points are given, if `x` is neither non-decreasing nor non-increasing,
        or if the true area is not representable as a finite `float64`.
    RuntimeError
        If the computed result fails this function's own postconditions.
    """
    x_array = _normalize_score_sequence(x, "x")
    y_array = _normalize_score_sequence(y, "y")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            f"x and y must have the same length, got {x_array.shape[0]} and {y_array.shape[0]}"
        )
    if x_array.shape[0] < 2:
        raise ValueError(f"at least 2 points are needed to compute an area, got {x_array.shape[0]}")

    non_decreasing = np.all(x_array[1:] >= x_array[:-1])
    non_increasing = np.all(x_array[1:] <= x_array[:-1])
    if not (non_decreasing or non_increasing):
        raise ValueError("x must be monotonic (either non-decreasing or non-increasing)")

    result = _trapezoidal_area(x_array, y_array)
    _check_auc_postconditions(result)
    return result


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


def _sort_and_group_by_score(
    scores: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.intp], npt.NDArray[np.float64], npt.NDArray[np.intp], npt.NDArray[np.intp]
]:
    """Shared by the unweighted and weighted ranking cores: descending sort order, the
    resulting sorted scores, and each distinct-score group's start/end index within that
    sorted order.

    Returns `(order, sorted_scores, group_start_indices, group_end_indices)`: `order` is a
    descending permutation of `scores` (via a stable ascending sort, reversed -- cheaper than a
    dedicated descending sort, and the within-tie order it produces is irrelevant either way,
    since every sample sharing a score is aggregated into one group); `sorted_scores =
    scores[order]`; group `i` spans sorted positions `group_start_indices[i]` through
    `group_end_indices[i]` inclusive, for `K` distinct scores in strictly decreasing order.

    Grouping uses direct comparison (`sorted_scores[1:] != sorted_scores[:-1]`), not
    `np.diff(sorted_scores) != 0`: subtracting two finite scores near +-float64 max can overflow
    to +-inf and raise a spurious "overflow encountered in subtract" warning -- verified
    directly.
    """
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]

    distinct = sorted_scores[1:] != sorted_scores[:-1]
    group_end_mask = np.concatenate((distinct, np.array([True], dtype=bool)))
    group_end_indices = np.flatnonzero(group_end_mask)

    group_start_indices = np.empty_like(group_end_indices)
    group_start_indices[0] = 0
    group_start_indices[1:] = group_end_indices[:-1] + 1

    return order, sorted_scores, group_start_indices, group_end_indices


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

    order, sorted_scores, _group_start_indices, group_end_indices = _sort_and_group_by_score(scores)
    sorted_is_positive = is_positive[order]

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


class _WeightedRankingCore(NamedTuple):
    """The weighted counterpart of `_RankingCore`, used only when `sample_weight` is given.

    Field names are deliberately different from `_RankingCore`'s (`true_positive_weight`/
    `false_positive_weight`/`total_positive_weight`/`total_negative_weight`, rather than
    `true_positives`/`false_positives`/`n_positive`/`n_negative`) so that passing the wrong core
    to a helper expecting the other one fails a static/attribute-name check rather than silently
    type-checking as `int64` counts. `thresholds`/`true_positive_weight`/`false_positive_weight`
    all share length `K + 1` for `K` distinct scores among the *effective* (positive-weight)
    samples, with index 0 the `+inf` sentinel (`0.0`/`0.0`) and indices `1..K` the cumulative
    effective weight at each of the `K` distinct scores, in strictly decreasing order.
    `total_positive_weight`/`total_negative_weight` equal
    `true_positive_weight[-1]`/`false_positive_weight[-1]` respectively, kept as their own plain
    `float` fields since every caller needs them as scalars.
    """

    thresholds: npt.NDArray[np.float64]
    true_positive_weight: npt.NDArray[np.float64]
    false_positive_weight: npt.NDArray[np.float64]
    total_positive_weight: float
    total_negative_weight: float
    positive_label: int


def _compute_weighted_ranking_core(
    y_true: object,
    y_score: object,
    sample_weight: object,
    positive_label: object,
    *,
    require_negative: bool,
) -> _WeightedRankingCore:
    """The weighted counterpart of `_compute_ranking_core`, used only when `sample_weight` is
    not `None`.

    `y_true`/`y_score`/`sample_weight` are each fully validated first (same as the unweighted
    core, plus `sample_weight`'s own contract via `_normalize_sample_weight`) -- a bad `y_score`
    is never "forgiven" merely because its weight happens to be zero. Only after that does a
    sample with `sample_weight == 0.0` get removed from the effective set, before sorting/
    grouping -- a score that exists only among zero-weight samples never produces a threshold.
    "Effective positive/negative support" (at least one *effective* sample of that class) is
    then checked with a plain integer count, not a weight sum -- exact, with no floating-point
    edge case, since `sample_weight > 0.0` already defines the effective set.

    Within each distinct-score group, the positive and negative samples' weights are each summed
    once via `math.fsum` over that group's weights sorted ascending first -- not
    `np.cumsum(weight, dtype=np.float64)` over individual samples: that would make the tie
    contract's "permuting samples within a tie never changes the result" promise depend on
    `float64` summation order (verified directly: `np.cumsum([1e16, 1.0, 1.0])[-1] != np.cumsum(
    [1.0, 1.0, 1e16])[-1]`), whereas `math.fsum` over a canonically sorted sequence gives the
    same correctly-rounded total regardless of the input order. `math.fsum` is called once per
    class per group, not once per prefix, so every sample is still consumed exactly once overall
    (beyond the cost of sorting).

    The `K`-length per-group weight totals are then turned into the length-`K` cumulative arrays
    with a single `np.cumsum(..., dtype=np.float64)` under `np.errstate(over="raise",
    invalid="raise")`, converting any overflow into `ValueError`. A group's own total being
    `> 0.0` is then required to strictly increase the corresponding cumulative sum over its
    predecessor (`0.0` for the first group) -- an extreme `sample_weight` dynamic range (e.g.
    `[1e16, 1.0]` at two different scores) can otherwise make a later, individually-positive
    group's contribution vanish into the running total's rounding error without ever overflowing
    or underflowing on its own (verified directly), which this check catches explicitly instead
    of silently returning a curve that skips that group's effect entirely.
    """
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

    weights = _normalize_sample_weight(sample_weight, len(true_list))

    is_positive = np.fromiter(
        (label == positive_label_value for label in true_list),
        dtype=bool,
        count=len(true_list),
    )

    effective_mask = weights > 0.0
    effective_is_positive = is_positive[effective_mask]
    n_positive_effective = int(np.count_nonzero(effective_is_positive))
    n_negative_effective = effective_is_positive.shape[0] - n_positive_effective
    if n_positive_effective == 0:
        raise ValueError(
            f"sample_weight assigns zero total weight to positive_label={positive_label_value!r}"
        )
    if require_negative and n_negative_effective == 0:
        raise ValueError("sample_weight assigns zero total weight to negative samples")

    effective_scores = scores[effective_mask]
    effective_weight = weights[effective_mask]

    order, sorted_scores, group_start_indices, group_end_indices = _sort_and_group_by_score(
        effective_scores
    )
    sorted_is_positive = effective_is_positive[order]
    sorted_weight = effective_weight[order]

    n_groups = group_end_indices.shape[0]
    positive_weight_per_score = np.empty(n_groups, dtype=np.float64)
    negative_weight_per_score = np.empty(n_groups, dtype=np.float64)

    try:
        for group_index in range(n_groups):
            start = int(group_start_indices[group_index])
            end = int(group_end_indices[group_index]) + 1
            group_is_positive = sorted_is_positive[start:end]
            group_weight = sorted_weight[start:end]
            positive_weight_per_score[group_index] = math.fsum(
                sorted(group_weight[group_is_positive].tolist())
            )
            negative_weight_per_score[group_index] = math.fsum(
                sorted(group_weight[~group_is_positive].tolist())
            )
    except OverflowError as exc:
        raise ValueError(
            "sample_weight dynamic range is too large to preserve every positive-weight "
            "threshold in float64"
        ) from exc

    try:
        with np.errstate(over="raise", invalid="raise"):
            cumulative_positive = np.cumsum(positive_weight_per_score, dtype=np.float64)
            cumulative_negative = np.cumsum(negative_weight_per_score, dtype=np.float64)
    except FloatingPointError as exc:
        raise ValueError(
            "sample_weight dynamic range is too large to preserve every threshold in float64"
        ) from exc

    _check_no_absorbed_weight_group(positive_weight_per_score, cumulative_positive)
    _check_no_absorbed_weight_group(negative_weight_per_score, cumulative_negative)

    thresholds = np.empty(n_groups + 1, dtype=np.float64)
    thresholds[0] = np.inf
    thresholds[1:] = sorted_scores[group_end_indices]

    true_positive_weight = np.empty(n_groups + 1, dtype=np.float64)
    true_positive_weight[0] = 0.0
    true_positive_weight[1:] = cumulative_positive

    false_positive_weight = np.empty(n_groups + 1, dtype=np.float64)
    false_positive_weight[0] = 0.0
    false_positive_weight[1:] = cumulative_negative

    return _WeightedRankingCore(
        thresholds=thresholds,
        true_positive_weight=true_positive_weight,
        false_positive_weight=false_positive_weight,
        total_positive_weight=float(cumulative_positive[-1]),
        total_negative_weight=float(cumulative_negative[-1]),
        positive_label=positive_label_value,
    )


def _check_no_absorbed_weight_group(
    weight_per_score: npt.NDArray[np.float64], cumulative: npt.NDArray[np.float64]
) -> None:
    """Raise ValueError if any group with a strictly positive weight total failed to strictly
    increase the running cumulative sum over its predecessor.

    `weight_per_score[i] > 0.0` must always mean `cumulative[i] > cumulative[i - 1]` (`> 0.0` for
    `i == 0`) -- an extreme dynamic range between groups (e.g. group weights `[1e16, 1.0]`) can
    make `np.cumsum` silently absorb a later, individually-positive group's entire contribution
    without raising any overflow/underflow/invalid signal (verified directly:
    `np.cumsum([1e16, 1.0])` gives `[1e16, 1e16]`, not `[1e16, 1.0000000000000002e16]`). This is
    deliberately conservative: the weighted core's results are defined by plain, deterministic
    `float64` arithmetic, but no positive-weight group is allowed to be silently dropped from the
    resulting curve's cumulative support.
    """
    previous = np.empty_like(cumulative)
    previous[0] = 0.0
    previous[1:] = cumulative[:-1]
    absorbed = (weight_per_score > 0.0) & ~(cumulative > previous)
    if np.any(absorbed):
        raise ValueError(
            "sample_weight dynamic range is too large to preserve every positive-weight "
            "threshold in float64"
        )


def _stable_precision_ratio(
    true_positive: npt.NDArray[np.float64], false_positive: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute `true_positive / (true_positive + false_positive)` elementwise without an
    intermediate `true_positive + false_positive` overflow.

    Only ever called (by `_precision_ratio`, on the subset of entries that actually overflow)
    where `true_positive + false_positive > 0` always holds by construction (every group has at
    least one effective, positive-weight sample). Plain `tp / (tp + fp)` silently returns `0.0`
    instead of the correct `0.5` when `tp == fp == np.finfo(np.float64).max` -- `tp + fp`
    overflows to `+inf` under default NumPy semantics there, and `tp / inf == 0.0` -- verified
    directly. Scaling both operands by their elementwise maximum first keeps every intermediate
    value within `float64`'s range whenever the true ratio is well-defined, which it always is
    here. This scaled formula does *not* reproduce plain division's bit pattern for ordinary,
    non-overflowing values (verified directly: `tp=3.0, fp=2.0` gives `0.6` via plain division
    but `0.6000000000000001` via this formula) -- which is exactly why `_precision_ratio` only
    routes the overflowing entries here, rather than falling back for the whole array.

    Dividing by `scale` can legitimately underflow to `0.0` when one operand is many orders of
    magnitude smaller than the other (e.g. `true_positive=tiny, false_positive=1.0`) -- a
    correctly-rounded result, not an error, so both divisions run under
    `np.errstate(under="ignore")`: a caller's own `np.seterr(under="raise"/"warn")` must not leak
    into this function's result (verified directly: `np.seterr(all="raise")` previously turned
    this legal underflow into an uncaught `FloatingPointError`).
    """
    scale = np.maximum(true_positive, false_positive)
    with np.errstate(under="ignore"):
        scaled_true_positive = true_positive / scale
        scaled_false_positive = false_positive / scale
        return scaled_true_positive / (scaled_true_positive + scaled_false_positive)


def _precision_ratio(
    true_positive: npt.NDArray[np.float64], false_positive: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute `true_positive / (true_positive + false_positive)` elementwise, matching plain
    `float64` division bit for bit whenever `true_positive + false_positive` does not overflow --
    only the (rare) overflowing entries are routed through `_stable_precision_ratio`'s
    scale-based formula instead.

    A single global fallback the moment *any* entry overflows would change every entry's bit
    pattern, not just the overflowing one's (see `_stable_precision_ratio`'s docstring) -- that
    would silently break the `sample_weight=[1.0, ...]` (all-ones) bit-for-bit identity with
    `sample_weight=None` for every threshold in the curve, not only the ones that would have
    actually overflowed. `true_positive + false_positive` is computed under
    `np.errstate(over="ignore")`: the overflow this deliberately allows (only to detect via
    `np.isinf` immediately after) would otherwise raise or warn.

    The ordinary division below can legitimately underflow to `0.0` (e.g. `true_positive=tiny,
    false_positive=1.0` gives a correctly-rounded `0.0` ratio) -- computed under
    `np.errstate(under="ignore")` for the same reason `_stable_precision_ratio` is: a caller's own
    `np.seterr(under="raise"/"warn")` must not leak into this function's result. `over`/`invalid`
    are deliberately left at their default (raising) sensitivity here -- an overflow in the
    ordinary branch, or a NaN/inf anywhere in `true_positive`/`false_positive`, would mean this
    function's own `overflowed` mask is wrong, an internal error this should not paper over.
    """
    with np.errstate(over="ignore"):
        combined = true_positive + false_positive
    overflowed = np.isinf(combined)
    with np.errstate(under="ignore"):
        if not np.any(overflowed):
            return true_positive / combined

        result = np.empty_like(true_positive)
        ordinary = ~overflowed
        result[ordinary] = true_positive[ordinary] / combined[ordinary]
    result[overflowed] = _stable_precision_ratio(
        true_positive[overflowed], false_positive[overflowed]
    )
    return result


def _compute_weighted_roc_rates(
    core: _WeightedRankingCore,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the false/true positive rate arrays shared by `roc_curve` and `roc_auc_score` for
    the weighted case.

    `total_positive_weight`/`total_negative_weight` are fixed, already-overflow-checked scalars
    (see `_compute_weighted_ranking_core`), and `true_positive_weight`/`false_positive_weight`
    are always `<=` their respective totals by construction -- so this plain elementwise
    division never needs `_stable_precision_ratio`'s scaling (that helper exists only for
    `tp + fp`, an unbounded sum of two independently-growing quantities, not for a bounded
    quantity divided by a fixed total).

    Run under `np.errstate(under="ignore")`: a tiny `true_positive_weight`/`false_positive_weight`
    divided by a huge total is a legitimate, correctly-rounded `0.0` rate, not an error -- a
    caller's own `np.seterr(under="raise"/"warn")` must not leak into this result (verified
    directly). `over`/`invalid` are deliberately left at their default sensitivity: both totals
    are already-overflow-checked positive scalars and every numerator is `<=` its total by
    construction, so neither should ever occur here -- if one did, that would be a real internal
    error, not a legal edge case to silence.
    """
    with np.errstate(under="ignore"):
        false_positive_rate = core.false_positive_weight / core.total_negative_weight
        true_positive_rate = core.true_positive_weight / core.total_positive_weight
    return false_positive_rate, true_positive_rate


def _compute_weighted_precision_recall_arrays(
    core: _WeightedRankingCore,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the precision/recall arrays shared by `precision_recall_curve` and
    `average_precision_score` for the weighted case.

    Returns new, independent, still-writeable `float64` buffers (never a view of `core`'s own
    arrays), mirroring `_compute_precision_recall_arrays`. `precision[1:]` uses
    `_precision_ratio` rather than a direct `true_positive / (true_positive + false_positive)`,
    since both operands here can independently grow large (unlike `recall`, divided by the fixed
    `total_positive_weight`) -- but `_precision_ratio` still matches that direct division bit for
    bit whenever it would not have overflowed.

    `recall`'s division runs under `np.errstate(under="ignore")` for the same reason
    `_compute_weighted_roc_rates` does: a tiny `true_positive_weight` divided by a huge
    `total_positive_weight` is a legitimate, correctly-rounded `0.0`, not an error that should
    depend on the caller's own `np.seterr` configuration.
    """
    true_positive = core.true_positive_weight
    false_positive = core.false_positive_weight

    precision = np.empty_like(true_positive)
    precision[0] = 1.0
    precision[1:] = _precision_ratio(true_positive[1:], false_positive[1:])

    with np.errstate(under="ignore"):
        recall = true_positive / core.total_positive_weight
    return precision, recall


def _trapezoidal_area(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Sum of `|x[i+1]-x[i]| * (y[i]+y[i+1]) * 0.5` over every segment -- shared by `auc` and
    `roc_auc_score`.

    Assumes `x`/`y` are already validated (finite, matching length, at least 2 points); `x`
    does not need to be monotonic here (`roc_auc_score`'s `false_positive_rate` always is, by
    construction, so `abs()` is a no-op there -- verified directly that this shared formula
    reproduces `roc_auc_score`'s previous bit-for-bit result on representative curves, ties,
    constant scores, and 1000 random deterministic rankings).

    Routes to the exact fallback in two situations:

    - `y` has both a positive and a negative value (checked first, unconditionally): every
      segment's width is already non-negative (`abs()`), so a `y` that is entirely non-negative
      or entirely non-positive can only ever produce same-signed contributions -- nothing to
      cancel, so the fast path is safe (and is what `roc_auc_score`'s always-non-negative TPR,
      and `auc(curve.recall, curve.precision)`'s always-non-negative precision, both use). Once
      `y` has mixed signs, contributions of opposite sign can cancel in the final sum, and a
      `float64` `y[i] + y[i+1]` can round away a residual that survives in exact arithmetic --
      verified directly: `x=[0,1,2], y=[1.0, 1e-20, -1.0]` has exact area `1e-20`, but
      `1.0 + 1e-20 == 1.0` and `1e-20 - 1.0 == -1.0` in `float64`, so the fast path's two
      segment contributions round to `0.5` and `-0.5` and cancel to `0.0` -- silently, with no
      overflow, underflow, or invalid operation for `np.errstate` to catch. Checked globally
      across all of `y`, not pairwise between neighbors: even non-adjacent positive and negative
      contributions can cancel in the final sum.
    - `_trapezoidal_area_float64` raises `FloatingPointError` -- for an intermediate overflow,
      an invalid (NaN-producing) operation, or an intermediate underflow that could lose a
      segment's contribution before it has a chance to be summed with its neighbors (only
      possible here when `y` is single-signed, since mixed-sign `y` is already routed to the
      exact fallback above).

    Either way, `_trapezoidal_area_exact_fallback` recomputes the same sum exactly (via
    `fractions.Fraction`) on the already-normalized `float64` values, so none of these failure
    modes ever turns into a wrong or falsely-rejected result: cancelling huge intermediate
    contributions, an accumulated subnormal residual, and a mixed-sign cancellation residual are
    all preserved exactly (see that function's docstring). Not every underflow or cancellation
    changes the final result -- a contribution genuinely closer to `0.0` than to the smallest
    positive subnormal `float64` legitimately rounds to `0.0` either way; the exact fallback
    decides this correctly, rather than the fast path guessing wrong in either direction.

    For `roc_auc_score`'s own `[0, 1]`-bounded, always-non-negative-TPR domain, none of this
    ever triggers, so that fast path is always the one taken there, unchanged.
    """
    if np.any(y > 0.0) and np.any(y < 0.0):
        return _trapezoidal_area_exact_fallback(x, y)

    try:
        return _trapezoidal_area_float64(x, y)
    except FloatingPointError:
        return _trapezoidal_area_exact_fallback(x, y)


def _trapezoidal_area_float64(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """The canonical, fast `float64` trapezoidal sum -- raises `FloatingPointError` instead of
    silently returning a wrong result for an intermediate overflow *or* underflow.

    Only ever called by `_trapezoidal_area` when `y` is entirely non-negative or entirely
    non-positive (mixed-sign `y` is routed directly to the exact fallback instead, before this
    function is reached -- see `_trapezoidal_area`'s own docstring for why cancellation between
    opposite-signed contributions can silently lose a representable residual in a way
    `np.errstate` cannot detect). For single-signed `y`, every segment's contribution has the
    same sign, so no cancellation between segments is possible here -- only overflow, an
    invalid operation, or underflow within a single segment's own computation can occur.

    Wrapped in `np.errstate(over="raise", invalid="raise", under="raise")`. `over`/`invalid`:
    verified directly that NumPy's default ("warn") mode silently returns `inf` (with only a
    printed, easily-missed `RuntimeWarning`) for a genuinely finite, representable input (e.g.
    `x=[0, 1e250], y=[1e200, 1e200]`) -- this makes that failure loud and catchable instead.
    `under`: a segment whose own exact contribution rounds to `0.0` under plain `float64`
    arithmetic can still be one of several such segments whose *exact* contributions sum to a
    representable positive subnormal `float64` -- verified directly with four segments, each
    individually underflowing to `0.0`, whose exact total is `5e-324` (the smallest positive
    subnormal `float64`). Catching the underflow here and falling back to
    `_trapezoidal_area_exact_fallback`'s exact rational summation is what tells a segment that
    genuinely contributes `0.0` apart from a segment that only *appears* to, once summed with
    its neighbors in plain `float64`.
    """
    with np.errstate(over="raise", invalid="raise", under="raise"):
        return float(np.sum(np.abs(np.diff(x)) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64))


def _trapezoidal_area_exact_fallback(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> float:
    """Recompute the trapezoidal sum exactly (via `fractions.Fraction`), for every case where
    `_trapezoidal_area_float64`'s fast, canonical `float64` arithmetic cannot be trusted to give
    the correctly-rounded result: an overflow or invalid (NaN-producing) intermediate operation,
    an intermediate underflow that could lose a segment's own contribution before summation, or
    `y` containing both a positive and a negative value (routed here directly, without ever
    calling `_trapezoidal_area_float64` at all -- see `_trapezoidal_area`'s docstring).

    Every `float64` value converts to an exact `Fraction` (`Fraction.from_float`), so summing
    every segment's exact contribution and converting only the final total back to `float`
    -- rather than computing each segment (or the whole sum) with `float64` arithmetic of its
    own -- correctly preserves cancellation between huge intermediate contributions (verified
    directly: `x=[-M, -M/3, M/3, M], y=[M, M, -M, -M]` sums exactly to `0.0`, and a second,
    similarly-cancelling input sums exactly to `1.0`), a subnormal residual that a
    power-of-two-scaled `float64` fallback would otherwise underflow away to `0.0` (verified
    directly: a sum whose exact value is `5e-324`, the smallest positive subnormal `float64`,
    previously came back as exactly `0.0` from a naive halve-then-double `float64` fallback,
    because halving a value already at the edge of the subnormal range loses it entirely), and a
    residual from cancellation between *opposite-signed* contributions (verified directly:
    `x=[0,1,2], y=[1.0, 1e-20, -1.0]` has exact area `1e-20`, but plain `float64` rounds
    `1.0 + 1e-20` to `1.0` and `1e-20 - 1.0` to `-1.0`, so the two segments' `float64`
    contributions of `0.5` and `-0.5` cancel to `0.0` with no overflow, underflow, or invalid
    operation for `np.errstate` to catch at all).

    `float(Fraction)` gives the correctly-rounded nearest `float64` for the exact total, and
    raises `OverflowError` when the true magnitude is too large for any finite `float64` --
    converted here to the same `ValueError` `auc`/`roc_auc_score` already document, so `auc`'s
    contract (`ValueError` only when the true area is not representable as a finite `float64`)
    holds exactly, rather than firing merely because the fast, unscaled `float64` attempt
    happened to overflow. `math.isfinite` is checked too, as a last-resort defense (`float()`
    on a `Fraction` should never itself produce a non-finite result without raising, but this
    keeps the postcondition -- "finite or `ValueError`, never anything else" -- honest even if
    that assumption were ever wrong).

    `fractions.Fraction` is standard library, not a new runtime dependency, and is used only on
    these rare paths (overflow, invalid, underflow, or mixed-sign `y`) -- ordinary calls with
    single-signed, non-overflowing, non-underflowing data never construct a `Fraction`.
    """
    total = Fraction()
    for index in range(x.shape[0] - 1):
        x0 = Fraction.from_float(float(x[index]))
        x1 = Fraction.from_float(float(x[index + 1]))
        y0 = Fraction.from_float(float(y[index]))
        y1 = Fraction.from_float(float(y[index + 1]))
        width = abs(x1 - x0)
        height_sum = y0 + y1
        total += width * height_sum / 2

    try:
        result = float(total)
    except OverflowError as exc:
        raise ValueError("the trapezoidal area could not be computed as a finite float64") from exc
    if not math.isfinite(result):
        raise ValueError("the trapezoidal area could not be computed as a finite float64")
    return result


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


def _normalize_nonnegative_weight_sequence(
    value: object, expected_length: int, *, allow_empty: bool
) -> npt.NDArray[np.float64]:
    """Raise TypeError/ValueError unless `value` is a valid `sample_weight` container; return a
    new, independent, non-negative `float64` ndarray.

    Shares its numeric-value policy with `_normalize_score_sequence` (same accepted containers --
    a `Sequence` explicitly excluding `str`/`bytes`/`bytearray`, or a 1-D `ndarray` -- and the same
    per-element rules: Python/NumPy int or `float16`/`float32`/`float64`, rejecting `bool`/
    complex/object/wider-than-`float64`/NaN/Inf/an integer not exactly representable as `float64`,
    canonicalizing signed zero) by calling the same low-level element/array normalizers
    (`_normalize_score_scalar`, `_exact_integer_ndarray_to_float64`, `_canonicalize_signed_zero`),
    not by calling `_normalize_score_sequence` itself -- that function always rejects an empty
    container, which `sample_weight` must NOT always do: `confusion_matrix` needs an empty
    `sample_weight=[]` to be legal exactly when `y_true`/`y_pred` are also empty (mirroring its
    existing "empty input plus explicit labels" contract), something no caller of
    `_normalize_score_sequence` has ever needed. `allow_empty` controls exactly that one
    difference; every other rule is identical.

    Additionally (unlike `_normalize_score_sequence`, which has no such concept) rejects negative
    values unconditionally: a negative weight would make `true_positive_rate` non-monotonic in
    the ranking core, or a confusion-matrix cell negative, breaking an invariant this module
    otherwise guarantees for both -- verified directly against a reference implementation that
    permits negative weights for both. Does NOT check for a positive total -- zero is a fully
    legal value here (it carries its own, caller-specific "remove this sample" semantics), and
    whether at least one positive weight is required depends entirely on the caller (ranking
    metrics always require it, `confusion_matrix` never requires it directly,
    `classification_metrics` requires it only after its own label validation) -- each decides for
    itself, on the array this function returns.
    """
    name = "sample_weight"
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a Sequence of real numbers or a 1-D float/integer ndarray, "
            f"not {type(value).__name__}"
        )
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {value.shape}")
        if value.size == 0:
            if not allow_empty:
                raise ValueError(f"{name} must not be empty")
            weights = np.empty(0, dtype=np.float64)
        else:
            dtype = value.dtype
            if np.issubdtype(dtype, np.floating):
                if dtype not in _ALLOWED_SCORE_FLOAT_DTYPES:
                    raise TypeError(
                        f"{name} must have dtype float16, float32, or float64, got {dtype} -- a "
                        "wider floating dtype could narrow values in a platform-dependent way"
                    )
                weights = value.astype(np.float64, copy=True)
            elif np.issubdtype(dtype, np.integer):
                weights = _exact_integer_ndarray_to_float64(value, name)
            else:
                raise TypeError(f"{name} must have a floating or integer dtype, got {dtype}")
            if not np.all(np.isfinite(weights)):
                raise ValueError(f"{name} must contain only finite values (no NaN or Inf)")
            _canonicalize_signed_zero(weights)
    elif not isinstance(value, Sequence):
        raise TypeError(
            f"{name} must be a Sequence of real numbers (e.g. a list or tuple) or a 1-D "
            f"float/integer ndarray, not {type(value).__name__}"
        )
    elif len(value) == 0:
        if not allow_empty:
            raise ValueError(f"{name} must not be empty")
        weights = np.empty(0, dtype=np.float64)
    else:
        weights = np.empty(len(value), dtype=np.float64)
        for index, element in enumerate(value):
            weights[index] = _normalize_score_scalar(element, f"{name}[{index}]")
        _canonicalize_signed_zero(weights)

    if weights.shape[0] != expected_length:
        raise ValueError(
            f"{name} must have the same length as y_true/y_pred, expected {expected_length}, "
            f"got {weights.shape[0]}"
        )
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return weights


def _normalize_sample_weight(value: object, expected_length: int) -> npt.NDArray[np.float64]:
    """Raise TypeError/ValueError unless `value` is a valid ranking `sample_weight`; return
    `float64` ndarray.

    A strict wrapper around `_normalize_nonnegative_weight_sequence` (`allow_empty=False`, since
    ranking metrics -- unlike `confusion_matrix` -- never accept empty input at all) that adds
    the one rule specific to ranking metrics: at least one strictly positive value is required
    (an all-zero `sample_weight` assigns no effective weight to any sample at all, and a ranking
    curve/score has no analogue of `confusion_matrix`'s "empty input plus explicit labels gives a
    well-defined all-zero result").
    """
    weights = _normalize_nonnegative_weight_sequence(value, expected_length, allow_empty=False)

    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must contain at least one positive value")

    return weights


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


def _check_auc_postconditions(value: float) -> None:
    """Unlike `_check_roc_auc_postconditions`/`_check_average_precision_postconditions`, does
    not check a `[0, 1]` bound: `auc`'s `x`/`y` are arbitrary, so its result can legitimately be
    negative or greater than `1.0` -- only finiteness is guaranteed by construction here.
    """
    if not isinstance(value, float):
        raise RuntimeError(f"internal error: auc result is not a float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise RuntimeError(f"internal error: auc result {value!r} is not finite")
