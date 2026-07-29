import dataclasses
import math
import warnings
from collections.abc import Sequence
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import improcv as im
from improcv.evaluation import (
    ClassificationMetrics,
    ConfusionMatrixResult,
    PrecisionRecallCurve,
    RocCurve,
    auc,
    average_precision_score,
    classification_metrics,
    classification_metrics_from_confusion_matrix,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class _CustomSequence(Sequence):
    """A minimal, real `collections.abc.Sequence` that is neither list nor tuple."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


def _assert_all_close(a, b) -> None:
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


# --- the 8 manual scenarios from the audit, cross-checked against scikit-learn ---


def test_scenario_1_perfect_classification() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    cm = confusion_matrix(y_true, y_pred)
    assert_array_equal(cm.matrix, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert cm.labels == (0, 1, 2)

    metrics = classification_metrics(y_true, y_pred)
    _assert_all_close(metrics.precision, [1.0, 1.0, 1.0])
    _assert_all_close(metrics.recall, [1.0, 1.0, 1.0])
    _assert_all_close(metrics.f1, [1.0, 1.0, 1.0])
    assert_array_equal(metrics.support, [2, 2, 2])
    assert metrics.accuracy == 1.0


def test_scenario_2_all_wrong() -> None:
    y_true = [0, 1, 2]
    y_pred = [1, 2, 0]

    cm = confusion_matrix(y_true, y_pred)
    assert_array_equal(cm.matrix, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    metrics = classification_metrics(y_true, y_pred)
    _assert_all_close(metrics.precision, [0.0, 0.0, 0.0])
    _assert_all_close(metrics.recall, [0.0, 0.0, 0.0])
    _assert_all_close(metrics.f1, [0.0, 0.0, 0.0])
    assert_array_equal(metrics.support, [1, 1, 1])
    assert metrics.accuracy == 0.0


def test_scenario_3_class_never_predicted() -> None:
    y_true = [0, 1, 2]
    y_pred = [0, 0, 0]

    metrics = classification_metrics(y_true, y_pred)
    _assert_all_close(metrics.precision, [1 / 3, 0.0, 0.0])
    _assert_all_close(metrics.recall, [1.0, 0.0, 0.0])
    _assert_all_close(metrics.f1, [0.5, 0.0, 0.0])
    assert_array_equal(metrics.support, [1, 1, 1])
    assert metrics.accuracy == pytest.approx(1 / 3)


# --- F1 correctness regressions: F1 must come from TP/FP/FN directly, not from
# precision*recall, or zero_division wrongly overrides a well-defined F1=0 ---


@pytest.mark.parametrize("zero_division", [0.0, 1.0, "nan"])
def test_f1_is_zero_for_real_fp_fn_even_when_precision_and_recall_are_zero(
    zero_division,
) -> None:
    # every class has TP=0 with real (nonzero) FP and FN -- precision and recall are
    # both correctly 0 (not zero_division fills), and F1 must likewise be a correctly
    # defined 0, regardless of zero_division.
    metrics = classification_metrics([0, 1, 2], [1, 2, 0], zero_division=zero_division)

    assert_array_equal(metrics.precision, [0.0, 0.0, 0.0])
    assert_array_equal(metrics.recall, [0.0, 0.0, 0.0])
    assert_array_equal(metrics.f1, [0.0, 0.0, 0.0])


def test_f1_zero_division_1_does_not_leak_into_defined_f1_for_all_wrong() -> None:
    metrics = classification_metrics([0, 1, 2], [1, 2, 0], zero_division=1.0)
    assert isinstance(metrics.f1, np.ndarray)
    assert not np.any(metrics.f1 == 1.0)


def test_f1_zero_division_nan_does_not_leak_into_defined_f1_for_all_wrong() -> None:
    metrics = classification_metrics([0, 1, 2], [1, 2, 0], zero_division="nan")
    assert isinstance(metrics.f1, np.ndarray)
    assert not np.any(np.isnan(metrics.f1))


def test_f1_class_never_predicted_but_present_in_truth_zero_division_1() -> None:
    metrics = classification_metrics([0, 1, 2], [0, 0, 0], zero_division=1.0)
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    # classes 1 and 2: never predicted -> precision is undefined (zero_division)
    assert metrics.precision[1] == 1.0
    assert metrics.precision[2] == 1.0
    # but recall and F1 are both well-defined here, not affected by zero_division
    assert metrics.recall[1] == 0.0
    assert metrics.recall[2] == 0.0
    assert metrics.f1[1] == 0.0
    assert metrics.f1[2] == 0.0


def test_f1_class_never_predicted_but_present_in_truth_zero_division_nan() -> None:
    metrics = classification_metrics([0, 1, 2], [0, 0, 0], zero_division="nan")
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    assert math.isnan(metrics.precision[1])
    assert math.isnan(metrics.precision[2])
    assert metrics.recall[1] == 0.0
    assert metrics.recall[2] == 0.0
    assert metrics.f1[1] == 0.0
    assert metrics.f1[2] == 0.0


@pytest.mark.parametrize("zero_division", [0.0, 1.0, "nan"])
def test_f1_class_completely_absent_still_uses_zero_division(zero_division) -> None:
    # class 3: TP=FP=FN=0 (declared but absent from both y_true and y_pred) --
    # precision, recall, AND f1 are all genuinely undefined here, so all three
    # must use zero_division.
    metrics = classification_metrics(
        [0, 1, 2], [0, 1, 2], labels=[0, 1, 2, 3], zero_division=zero_division
    )
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    expected = math.nan if zero_division == "nan" else zero_division
    if zero_division == "nan":
        assert math.isnan(metrics.precision[3])
        assert math.isnan(metrics.recall[3])
        assert math.isnan(metrics.f1[3])
    else:
        assert metrics.precision[3] == expected
        assert metrics.recall[3] == expected
        assert metrics.f1[3] == expected


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize("zero_division", [0.0, 1.0, "nan"])
def test_f1_fix_holds_under_every_average_mode(average, zero_division) -> None:
    # all-wrong classification: F1 must never leak to 1.0/NaN under any average mode
    metrics = classification_metrics(
        [0, 1, 2], [1, 2, 0], average=average, zero_division=zero_division
    )
    assert isinstance(metrics.f1, float)
    assert metrics.f1 == 0.0


def test_scenario_4_class_absent_from_truth() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 2]

    metrics = classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    _assert_all_close(metrics.precision, [1.0, 0.5, 0.0])
    _assert_all_close(metrics.recall, [0.5, 0.5, 0.0])
    _assert_all_close(metrics.f1, [2 / 3, 0.5, 0.0])
    assert_array_equal(metrics.support, [2, 2, 0])
    assert metrics.accuracy == 0.5


def test_scenario_5_declared_class_absent_everywhere() -> None:
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]

    metrics = classification_metrics(y_true, y_pred, labels=[0, 1, 2, 3])
    _assert_all_close(metrics.precision, [1.0, 1.0, 1.0, 0.0])
    _assert_all_close(metrics.recall, [1.0, 1.0, 1.0, 0.0])
    _assert_all_close(metrics.f1, [1.0, 1.0, 1.0, 0.0])
    assert_array_equal(metrics.support, [1, 1, 1, 0])
    assert metrics.accuracy == 1.0

    # both precision and recall are simultaneously undefined (0/0) for class 3;
    # zero_division="nan" must make both NaN, not just one
    nan_metrics = classification_metrics(y_true, y_pred, labels=[0, 1, 2, 3], zero_division="nan")
    assert isinstance(nan_metrics.precision, np.ndarray)
    assert isinstance(nan_metrics.recall, np.ndarray)
    assert isinstance(nan_metrics.f1, np.ndarray)
    assert math.isnan(nan_metrics.precision[3])
    assert math.isnan(nan_metrics.recall[3])
    assert math.isnan(nan_metrics.f1[3])


def test_scenario_6_single_class() -> None:
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    cm = confusion_matrix(y_true, y_pred)
    assert_array_equal(cm.matrix, [[3]])

    for average in (None, "micro", "macro", "weighted"):
        metrics = classification_metrics(y_true, y_pred, average=average)
        _assert_all_close(metrics.precision, 1.0)
        _assert_all_close(metrics.recall, 1.0)
        _assert_all_close(metrics.f1, 1.0)
        assert metrics.accuracy == 1.0


def test_scenario_7_imbalance() -> None:
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10

    metrics_none = classification_metrics(y_true, y_pred, labels=[0, 1])
    _assert_all_close(metrics_none.precision, [0.9, 0.0])
    _assert_all_close(metrics_none.recall, [1.0, 0.0])
    _assert_all_close(metrics_none.f1, [2 * 0.9 * 1 / (0.9 + 1), 0.0])
    assert metrics_none.accuracy == pytest.approx(0.9)

    micro = classification_metrics(y_true, y_pred, labels=[0, 1], average="micro")
    assert micro.precision == pytest.approx(0.9)
    assert micro.recall == pytest.approx(0.9)
    assert micro.f1 == pytest.approx(0.9)

    macro = classification_metrics(y_true, y_pred, labels=[0, 1], average="macro")
    assert macro.precision == pytest.approx(0.45)
    assert macro.recall == pytest.approx(0.5)
    assert macro.f1 == pytest.approx((2 * 0.9 * 1 / 1.9) / 2)

    weighted = classification_metrics(y_true, y_pred, labels=[0, 1], average="weighted")
    assert weighted.precision == pytest.approx(0.81)
    assert weighted.recall == pytest.approx(0.9)
    assert weighted.f1 == pytest.approx((2 * 0.9 * 1 / 1.9) * 0.9)


def test_scenario_8_prediction_outside_explicit_labels_is_an_error() -> None:
    y_true = [0, 1, 0]
    y_pred = [0, 1, 2]

    with pytest.raises(ValueError, match=r"y_pred\[2\]"):
        confusion_matrix(y_true, y_pred, labels=[0, 1])


# --- micro == accuracy, weighted recall == accuracy (single-label multiclass identities) ---


@pytest.mark.parametrize(
    ("y_true", "y_pred", "labels"),
    [
        ([0] * 9 + [1], [0] * 10, [0, 1]),
        ([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [1, 2, 0], [0, 1, 2]),
        ([0, 0, 1, 1], [0, 1, 1, 2], [0, 1, 2]),
    ],
)
def test_micro_equals_accuracy(y_true, y_pred, labels) -> None:
    metrics = classification_metrics(y_true, y_pred, labels=labels, average="micro")
    accuracy = classification_metrics(y_true, y_pred, labels=labels).accuracy
    assert metrics.precision == pytest.approx(accuracy)
    assert metrics.recall == pytest.approx(accuracy)
    assert metrics.f1 == pytest.approx(accuracy)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "labels"),
    [
        ([0] * 9 + [1], [0] * 10, [0, 1]),
        ([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], [0, 1, 2]),
        ([0, 0, 1, 1], [0, 1, 1, 2], [0, 1, 2]),
    ],
)
def test_weighted_recall_equals_accuracy(y_true, y_pred, labels) -> None:
    metrics = classification_metrics(y_true, y_pred, labels=labels, average="weighted")
    accuracy = classification_metrics(y_true, y_pred, labels=labels).accuracy
    assert metrics.recall == pytest.approx(accuracy)


# --- confusion_matrix: orientation, labels resolution ---


def test_confusion_matrix_orientation_rows_true_columns_predicted() -> None:
    # 2 samples: true=0 predicted=1 (twice) -- must land at matrix[0][1], not matrix[1][0]
    cm = confusion_matrix([0, 0], [1, 1], labels=[0, 1])
    assert cm.matrix[0, 1] == 2
    assert cm.matrix[1, 0] == 0


def test_confusion_matrix_labels_none_gives_sorted_union() -> None:
    cm = confusion_matrix([2, 0, 1], [1, 0, 2])
    assert cm.labels == (0, 1, 2)


def test_confusion_matrix_explicit_labels_preserve_given_order() -> None:
    cm = confusion_matrix([0, 1], [0, 1], labels=[2, 1, 0])
    assert cm.labels == (2, 1, 0)


def test_confusion_matrix_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])


def test_confusion_matrix_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="empty"):
        confusion_matrix([0], [0], labels=[])


def test_confusion_matrix_negative_labels_are_legal() -> None:
    cm = confusion_matrix([-5, 3], [-5, 3])
    assert cm.labels == (-5, 3)
    assert_array_equal(cm.matrix, [[1, 0], [0, 1]])


def test_confusion_matrix_very_large_python_int_labels() -> None:
    huge = 10**30
    cm = confusion_matrix([huge, 1], [huge, 1])
    assert cm.labels == (1, huge)
    assert_array_equal(cm.matrix, [[1, 0], [0, 1]])


def test_confusion_matrix_sparse_labels_no_dense_max_allocation() -> None:
    cm = confusion_matrix([0, 1_000_000_000], [0, 1_000_000_000])
    assert cm.matrix.shape == (2, 2)
    assert_array_equal(cm.matrix, [[1, 0], [0, 1]])


def test_confusion_matrix_accepts_int_enum_labels() -> None:
    cm = confusion_matrix([_Color.RED, _Color.GREEN], [_Color.RED, _Color.GREEN])
    assert cm.labels == (0, 1)
    assert all(type(label) is int for label in cm.labels)


@pytest.mark.parametrize("container", [list, tuple, _CustomSequence])
def test_confusion_matrix_accepts_various_sequence_containers(container) -> None:
    cm = confusion_matrix(container([0, 1, 0]), container([0, 1, 1]))
    assert cm.labels == (0, 1)


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_confusion_matrix_accepts_1d_ndarray_of_any_integer_dtype(dtype) -> None:
    y_true = np.array([0, 1, 0], dtype=dtype)
    y_pred = np.array([0, 1, 1], dtype=dtype)
    cm = confusion_matrix(y_true, y_pred)
    assert cm.labels == (0, 1)


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda: np.array([True, False]),
        lambda: np.array([1.0, 2.0]),
        lambda: np.array([1, "a"], dtype=object),
    ],
)
def test_confusion_matrix_rejects_non_integer_ndarray_dtype(make_bad) -> None:
    with pytest.raises(TypeError):
        confusion_matrix(make_bad(), make_bad())


def test_confusion_matrix_rejects_generator() -> None:
    with pytest.raises(TypeError):
        confusion_matrix((x for x in [0, 1]), [0, 1])  # type: ignore[arg-type]


def test_confusion_matrix_rejects_iterator() -> None:
    with pytest.raises(TypeError):
        confusion_matrix(iter([0, 1]), [0, 1])  # type: ignore[arg-type]


def test_confusion_matrix_rejects_2d_ndarray() -> None:
    with pytest.raises(ValueError, match="1-D"):
        confusion_matrix(np.zeros((2, 2), dtype=np.int64), [0, 1])


def test_confusion_matrix_rejects_0d_ndarray() -> None:
    with pytest.raises(ValueError, match="1-D"):
        confusion_matrix(np.array(5), [0])


def test_confusion_matrix_rejects_str_bytes_bytearray() -> None:
    for bad in ("01", b"01", bytearray(b"01")):
        with pytest.raises(TypeError):
            confusion_matrix(bad, [0, 1])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_element", [True, 1.0, "a", None, 1 + 2j])
def test_confusion_matrix_rejects_non_integral_element(bad_element) -> None:
    with pytest.raises(TypeError, match=r"y_true\[1\]"):
        confusion_matrix([0, bad_element], [0, 0])


def test_confusion_matrix_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        confusion_matrix([0, 1, 2], [0, 1])


def test_confusion_matrix_empty_with_explicit_labels_returns_zero_matrix() -> None:
    cm = confusion_matrix([], [], labels=[0, 1])
    assert cm.matrix.shape == (2, 2)
    assert_array_equal(cm.matrix, [[0, 0], [0, 0]])
    assert not cm.matrix.flags.writeable


def test_confusion_matrix_empty_without_labels_raises() -> None:
    with pytest.raises(ValueError, match="infer"):
        confusion_matrix([], [], labels=None)


def test_confusion_matrix_does_not_mutate_inputs() -> None:
    y_true = [0, 1, 2]
    y_pred = [2, 1, 0]
    before_true, before_pred = list(y_true), list(y_pred)

    confusion_matrix(y_true, y_pred)

    assert y_true == before_true
    assert y_pred == before_pred


def test_confusion_matrix_result_is_read_only_and_independent() -> None:
    y_true = np.array([0, 1, 0])
    cm = confusion_matrix(y_true, [0, 1, 1])

    assert not cm.matrix.flags.writeable
    assert not np.shares_memory(cm.matrix, y_true)
    with pytest.raises(ValueError):
        cm.matrix[0, 0] = 99


def test_confusion_matrix_permutation_of_samples_is_invariant() -> None:
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 1, 0, 2]
    perm = [3, 0, 5, 1, 4, 2]
    y_true_perm = [y_true[i] for i in perm]
    y_pred_perm = [y_pred[i] for i in perm]

    cm1 = confusion_matrix(y_true, y_pred)
    cm2 = confusion_matrix(y_true_perm, y_pred_perm)
    assert cm1 == cm2


def test_confusion_matrix_permuting_labels_permutes_axes() -> None:
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]

    cm_natural = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_permuted = confusion_matrix(y_true, y_pred, labels=[2, 0, 1])

    assert cm_permuted.labels == (2, 0, 1)
    # diagonal must remain "correct predictions" under any label order
    assert_array_equal(np.diagonal(cm_natural.matrix), np.diagonal(cm_permuted.matrix))


# --- allocation representability, without a real huge allocation ---


def test_confusion_matrix_rejects_unrepresentable_class_count() -> None:
    huge_labels = list(range(1)) + [10**18]  # only 2 real classes; check the *count* path
    # exercise the guard directly via an absurd number of *distinct* labels instead of
    # constructing an actual huge list -- use the private guard to avoid any real allocation.
    from improcv.evaluation import _check_allocation_representable

    with pytest.raises(ValueError):
        _check_allocation_representable(2**40)

    # sanity: a small, real call still works normally
    cm = confusion_matrix(huge_labels, huge_labels)
    assert cm.matrix.shape == (2, 2)


# --- classification_metrics: averaging return types ---


def test_classification_metrics_average_none_returns_arrays() -> None:
    metrics = classification_metrics([0, 1], [0, 1], average=None)
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    assert metrics.precision.dtype == np.float64
    assert metrics.average is None


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_classification_metrics_aggregate_average_returns_floats(average) -> None:
    metrics = classification_metrics([0, 1], [0, 1], average=average)
    assert isinstance(metrics.precision, float)
    assert isinstance(metrics.recall, float)
    assert isinstance(metrics.f1, float)
    assert metrics.average == average


def test_classification_metrics_support_is_always_per_class_array() -> None:
    for average in (None, "micro", "macro", "weighted"):
        metrics = classification_metrics([0, 1, 1], [0, 1, 0], average=average)
        assert isinstance(metrics.support, np.ndarray)
        assert metrics.support.dtype == np.int64
        assert metrics.support.shape == (2,)


def test_classification_metrics_accuracy_always_float() -> None:
    for average in (None, "micro", "macro", "weighted"):
        metrics = classification_metrics([0, 1], [0, 1], average=average)
        assert isinstance(metrics.accuracy, float)


def test_classification_metrics_rejects_bad_average() -> None:
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 1], average="binary")  # type: ignore[arg-type]


# --- zero_division ---


@pytest.mark.parametrize("zero_division", [0.0, 1.0, "nan"])
def test_classification_metrics_accepts_valid_zero_division(zero_division) -> None:
    classification_metrics([0, 1, 2], [0, 0, 0], zero_division=zero_division)


def test_classification_metrics_rejects_bool_zero_division() -> None:
    with pytest.raises(TypeError):
        classification_metrics([0, 1], [0, 1], zero_division=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("zero_division", [0.5, "warn", None, 2])
def test_classification_metrics_rejects_invalid_zero_division(zero_division) -> None:
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 1], zero_division=zero_division)  # type: ignore[arg-type]


def test_classification_metrics_zero_division_1_used_for_undefined_precision() -> None:
    metrics = classification_metrics([0, 1, 2], [0, 0, 0], zero_division=1.0)
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert metrics.precision[1] == 1.0
    assert metrics.precision[2] == 1.0
    # recall is well-defined (not a division-by-zero case) here and must not be affected
    assert metrics.recall[1] == 0.0
    assert metrics.recall[2] == 0.0


def test_classification_metrics_nan_propagates_to_macro_not_masked() -> None:
    metrics = classification_metrics(
        [0, 1, 2], [0, 0, 0], labels=[0, 1, 2], zero_division="nan", average="macro"
    )
    assert math.isnan(metrics.precision)
    # recall has no undefined per-class values in this scenario, so macro recall is real
    assert not math.isnan(metrics.recall)


def test_classification_metrics_nan_propagates_through_weighted_even_at_zero_weight() -> None:
    # class 3 has zero support (zero weight) AND an undefined (0/0) precision/recall;
    # NaN * 0 == NaN, so the weighted aggregate must also be NaN, not skip that class.
    metrics = classification_metrics(
        [0, 1, 2], [0, 1, 2], labels=[0, 1, 2, 3], zero_division="nan", average="weighted"
    )
    assert math.isnan(metrics.precision)
    assert math.isnan(metrics.recall)
    assert math.isnan(metrics.f1)


def test_classification_metrics_no_numpy_warnings_raised(recwarn) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classification_metrics([0, 1, 2], [0, 0, 0], labels=[0, 1, 2, 3], zero_division="nan")
        classification_metrics([0, 1, 2], [0, 0, 0], zero_division=0.0)
        classification_metrics([0, 1, 2], [0, 0, 0], zero_division=1.0)


# --- empty data for classification_metrics ---


def test_classification_metrics_rejects_empty_data_even_with_labels() -> None:
    with pytest.raises(ValueError):
        classification_metrics([], [], labels=[0, 1])


def test_classification_metrics_from_confusion_matrix_rejects_zero_sum_matrix() -> None:
    zero_matrix = ConfusionMatrixResult(matrix=np.zeros((2, 2), dtype=np.int64), labels=(0, 1))
    with pytest.raises(ValueError):
        classification_metrics_from_confusion_matrix(zero_matrix)


# --- classification_metrics_from_confusion_matrix: type/validation ---


def test_from_confusion_matrix_rejects_bare_ndarray() -> None:
    with pytest.raises(TypeError, match="ConfusionMatrixResult"):
        classification_metrics_from_confusion_matrix(np.eye(2, dtype=np.int64))  # type: ignore[arg-type]


def test_from_confusion_matrix_consistent_with_direct_call() -> None:
    y_true, y_pred, labels = [0] * 9 + [1], [0] * 10, [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for average in (None, "micro", "macro", "weighted"):
        direct = classification_metrics(y_true, y_pred, labels=labels, average=average)
        from_cm = classification_metrics_from_confusion_matrix(cm, average=average)
        assert direct == from_cm


@pytest.mark.parametrize(
    "make_bad_matrix",
    [
        lambda: np.eye(2, dtype=np.float64),
        lambda: np.eye(2, dtype=bool),
        lambda: np.eye(2, dtype=np.int32),
        lambda: np.eye(2, dtype=np.uint64),
    ],
)
def test_from_confusion_matrix_rejects_wrong_matrix_dtype(make_bad_matrix) -> None:
    bad = ConfusionMatrixResult(matrix=make_bad_matrix(), labels=(0, 1))
    with pytest.raises(TypeError):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_rejects_non_square_matrix() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((2, 3), dtype=np.int64), labels=(0, 1))
    with pytest.raises(ValueError, match="square"):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_rejects_negative_counts() -> None:
    bad = ConfusionMatrixResult(matrix=np.array([[1, -1], [0, 1]], dtype=np.int64), labels=(0, 1))
    with pytest.raises(ValueError, match="negative"):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_rejects_labels_length_mismatch() -> None:
    bad = ConfusionMatrixResult(matrix=np.eye(2, dtype=np.int64), labels=(0, 1, 2))
    with pytest.raises(ValueError, match="labels"):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_rejects_duplicate_labels_field() -> None:
    bad = ConfusionMatrixResult(matrix=np.eye(2, dtype=np.int64), labels=(0, 0))
    with pytest.raises(ValueError, match="duplicate"):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_rejects_bool_label_in_labels_field() -> None:
    bad = ConfusionMatrixResult(matrix=np.eye(2, dtype=np.int64), labels=(True, False))
    with pytest.raises(TypeError):
        classification_metrics_from_confusion_matrix(bad)


def test_from_confusion_matrix_accepts_manually_aggregated_matrix() -> None:
    batch_a = confusion_matrix([0, 1], [0, 1], labels=[0, 1]).matrix
    batch_b = confusion_matrix([0, 1], [1, 1], labels=[0, 1]).matrix
    combined = ConfusionMatrixResult(matrix=(batch_a + batch_b), labels=(0, 1))

    metrics = classification_metrics_from_confusion_matrix(combined)
    assert_array_equal(metrics.support, [2, 2])


# --- int64 overflow regression: a manually constructed matrix whose true total
# exceeds int64 must be rejected, never silently wrapped to a negative support ---


def test_from_confusion_matrix_rejects_overflowing_manual_matrix() -> None:
    m = np.int64(2**62)
    matrix = np.array(
        [
            [0, m, m],
            [m, 0, m],
            [m, 0, 0],
        ],
        dtype=np.int64,
    )
    confusion = ConfusionMatrixResult(matrix=matrix, labels=(0, 1, 2))

    with pytest.raises(ValueError, match="int64"):
        classification_metrics_from_confusion_matrix(confusion)


def test_from_confusion_matrix_int64_max_boundary_is_legal() -> None:
    matrix = np.array([[np.iinfo(np.int64).max]], dtype=np.int64)
    confusion = ConfusionMatrixResult(matrix=matrix, labels=(0,))

    metrics = classification_metrics_from_confusion_matrix(confusion)

    assert_array_equal(metrics.support, [np.iinfo(np.int64).max])
    assert metrics.accuracy == 1.0
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    _assert_all_close(metrics.precision, [1.0])
    _assert_all_close(metrics.recall, [1.0])
    _assert_all_close(metrics.f1, [1.0])


# --- read-only / aliasing on classification_metrics results ---


def test_classification_metrics_arrays_are_read_only() -> None:
    metrics = classification_metrics([0, 1, 1], [0, 1, 0], average=None)
    assert isinstance(metrics.precision, np.ndarray)
    assert isinstance(metrics.recall, np.ndarray)
    assert isinstance(metrics.f1, np.ndarray)
    assert not metrics.precision.flags.writeable
    assert not metrics.recall.flags.writeable
    assert not metrics.f1.flags.writeable
    assert not metrics.support.flags.writeable
    for arr in (metrics.precision, metrics.recall, metrics.f1, metrics.support):
        with pytest.raises(ValueError):
            arr[0] = 99


def test_classification_metrics_no_aliasing_with_confusion_matrix() -> None:
    cm = confusion_matrix([0, 1, 1], [0, 1, 0])
    metrics = classification_metrics_from_confusion_matrix(cm)
    assert not np.shares_memory(metrics.support, cm.matrix)


# --- equality ---


def test_confusion_matrix_result_equality() -> None:
    a = confusion_matrix([0, 1], [0, 1])
    b = confusion_matrix([0, 1], [0, 1])
    c = confusion_matrix([0, 1], [1, 1])
    assert a == b
    assert a != c
    assert a != 5
    assert (a == 5) is False


def test_confusion_matrix_result_unhashable() -> None:
    a = confusion_matrix([0, 1], [0, 1])
    with pytest.raises(TypeError):
        hash(a)


def test_classification_metrics_equality_meaningful() -> None:
    a = classification_metrics([0, 1, 1], [0, 1, 0], average=None)
    b = classification_metrics([0, 1, 1], [0, 1, 0], average=None)
    c = classification_metrics([0, 1, 1], [0, 1, 1], average=None)
    assert a == b
    assert a != c
    assert a != "not a metrics object"


def test_classification_metrics_equality_with_nan() -> None:
    a = classification_metrics([0, 1, 2], [0, 1, 2], labels=[0, 1, 2, 3], zero_division="nan")
    b = classification_metrics([0, 1, 2], [0, 1, 2], labels=[0, 1, 2, 3], zero_division="nan")
    assert a == b  # both have NaN in the same position(s); must compare equal


def test_classification_metrics_unhashable() -> None:
    a = classification_metrics([0, 1], [0, 1])
    with pytest.raises(TypeError):
        hash(a)


def test_classification_metrics_aggregate_equality() -> None:
    a = classification_metrics([0, 1, 1], [0, 1, 0], average="macro")
    b = classification_metrics([0, 1, 1], [0, 1, 0], average="macro")
    assert a == b
    assert isinstance(a.precision, float)


# --- top-level exports ---


def test_evaluation_exports_from_top_level_package() -> None:
    assert im.confusion_matrix is confusion_matrix
    assert im.classification_metrics is classification_metrics
    assert (
        im.classification_metrics_from_confusion_matrix
        is classification_metrics_from_confusion_matrix
    )
    assert im.ConfusionMatrixResult is ConfusionMatrixResult
    assert im.roc_curve is roc_curve
    assert im.precision_recall_curve is precision_recall_curve
    assert im.roc_auc_score is roc_auc_score
    assert im.RocCurve is RocCurve
    assert im.PrecisionRecallCurve is PrecisionRecallCurve
    assert im.average_precision_score is average_precision_score
    assert im.auc is auc


# =====================================================================================
# Binary one-vs-rest ranking curves: roc_curve / precision_recall_curve / roc_auc_score /
# average_precision_score
# =====================================================================================

_RANKING_FUNCTIONS = (roc_curve, precision_recall_curve, roc_auc_score, average_precision_score)


def _mann_whitney_auc(y_true, y_score, positive_label) -> float:
    """Independent, non-vectorized AUC oracle: fraction of (positive, negative) pairs where
    the positive outranks the negative, with a tied pair counted as one-half."""
    positives = [s for t, s in zip(y_true, y_score, strict=True) if t == positive_label]
    negatives = [s for t, s in zip(y_true, y_score, strict=True) if t != positive_label]
    total = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                total += 1.0
            elif p == n:
                total += 0.5
    return total / (len(positives) * len(negatives))


class _CustomScoreSequence(Sequence):
    """A minimal, real `collections.abc.Sequence` that is neither list nor tuple."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


# --- y_true validation (shared across all three functions via the private ranking core) ---


@pytest.mark.parametrize(
    ("make_y_true", "expected_exception"),
    [
        (lambda: (x for x in [0, 0, 1, 1]), TypeError),
        (lambda: iter([0, 0, 1, 1]), TypeError),
        (lambda: "0011", TypeError),
        (lambda: b"0011", TypeError),
        (lambda: [False, False, True, True], TypeError),
        (lambda: [0.0, 0.0, 1.0, 1.0], TypeError),
        (lambda: np.array(1), ValueError),
        (lambda: np.array([[0, 1], [0, 1]]), ValueError),
        (lambda: [], ValueError),
    ],
)
def test_ranking_functions_reject_bad_y_true(make_y_true, expected_exception) -> None:
    y_score = [0.1, 0.4, 0.35, 0.8]
    for func in _RANKING_FUNCTIONS:
        with pytest.raises(expected_exception):
            func(make_y_true(), y_score, positive_label=1)


@pytest.mark.parametrize("container", [list, tuple, _CustomScoreSequence])
def test_ranking_functions_accept_various_y_true_sequence_containers(container) -> None:
    y_true = container([0, 1, 0, 1])
    y_score = [0.1, 0.9, 0.2, 0.8]
    for func in _RANKING_FUNCTIONS:
        func(y_true, y_score, positive_label=1)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32])
def test_ranking_functions_accept_1d_integer_ndarray_y_true(dtype) -> None:
    y_true = np.array([0, 1, 0, 1], dtype=dtype)
    y_score = [0.1, 0.9, 0.2, 0.8]
    for func in _RANKING_FUNCTIONS:
        func(y_true, y_score, positive_label=1)


def test_ranking_functions_accept_int_enum_labels() -> None:
    y_true = [_Color.RED, _Color.GREEN, _Color.RED, _Color.GREEN]
    y_score = [0.1, 0.9, 0.2, 0.8]
    for func in _RANKING_FUNCTIONS:
        func(y_true, y_score, positive_label=_Color.GREEN)


def test_ranking_functions_accept_negative_and_huge_integer_labels() -> None:
    y_true = [-5, 10**20, -5, 10**20]
    y_score = [0.1, 0.9, 0.2, 0.8]
    for func in _RANKING_FUNCTIONS:
        func(y_true, y_score, positive_label=10**20)


def test_ranking_functions_accept_many_distinct_negative_labels() -> None:
    y_true = [5, 7, 2, 1, 5, 7]
    y_score = [0.9, 0.1, 0.2, 0.8, 0.3, 0.05]
    roc = roc_curve(y_true, y_score, positive_label=5)
    assert roc.true_positive_rate[-1] == 1.0
    assert roc.false_positive_rate[-1] == 1.0


def test_positive_label_absent_from_y_true_raises() -> None:
    for func in _RANKING_FUNCTIONS:
        with pytest.raises(ValueError):
            func([0, 0, 2, 2], [0.1, 0.2, 0.3, 0.4], positive_label=1)


def test_ranking_functions_require_at_least_one_positive() -> None:
    for func in _RANKING_FUNCTIONS:
        with pytest.raises(ValueError):
            func([0, 0, 0], [0.1, 0.2, 0.3], positive_label=1)


def test_roc_and_auc_require_at_least_one_negative() -> None:
    with pytest.raises(ValueError):
        roc_curve([1, 1, 1], [0.1, 0.2, 0.3], positive_label=1)
    with pytest.raises(ValueError):
        roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3], positive_label=1)


def test_precision_recall_curve_allows_no_negative_samples() -> None:
    pr = precision_recall_curve([1, 1, 1], [0.1, 0.2, 0.3], positive_label=1)
    assert np.all(pr.precision == 1.0)
    assert_array_equal(pr.recall, [0.0, 1 / 3, 2 / 3, 1.0])


# --- positive_label validation ---


@pytest.mark.parametrize(
    "positive_label",
    [True, np.bool_(True), 1.0, "1", None],
)
def test_ranking_functions_reject_bad_positive_label(positive_label) -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    for func in _RANKING_FUNCTIONS:
        with pytest.raises(TypeError):
            func(y_true, y_score, positive_label=positive_label)


def test_ranking_functions_accept_negative_positive_label() -> None:
    y_true = [-1, -1, 3, 3]
    y_score = [0.1, 0.9, 0.2, 0.8]
    for func in _RANKING_FUNCTIONS:
        func(y_true, y_score, positive_label=-1)


# --- y_score validation ---


@pytest.mark.parametrize(
    ("make_y_score", "expected_exception"),
    [
        (lambda: (s for s in [0.1, 0.4, 0.35, 0.8]), TypeError),
        (lambda: iter([0.1, 0.4, 0.35, 0.8]), TypeError),
        (lambda: "abcd", TypeError),
        (lambda: b"abcd", TypeError),
        (lambda: np.array([True, False, True, False]), TypeError),
        (lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j]), TypeError),
        (lambda: np.array([0.1, 0.4, 0.35, 0.8], dtype=object), TypeError),
        (lambda: np.array(1.0), ValueError),
        (lambda: np.array([[0.1, 0.4], [0.35, 0.8]]), ValueError),
        (lambda: [], ValueError),
        (lambda: [0.1, 0.4, float("nan"), 0.8], ValueError),
        (lambda: [0.1, 0.4, float("inf"), 0.8], ValueError),
        (lambda: [0.1, 0.4, float("-inf"), 0.8], ValueError),
        (lambda: [0.1, 0.4, Decimal("0.35"), 0.8], TypeError),
        (lambda: [0.1, 0.4, Fraction(35, 100), 0.8], TypeError),
        (lambda: [0.1, 0.4, None, 0.8], TypeError),
        (lambda: [0.1, 0.4, 1 + 2j, 0.8], TypeError),
        (lambda: [0.1, 0.4, True, 0.8], TypeError),
        (lambda: [0.1, 0.4, 0.35], ValueError),
    ],
)
def test_ranking_functions_reject_bad_y_score(make_y_score, expected_exception) -> None:
    y_true = [0, 0, 1, 1]
    for func in _RANKING_FUNCTIONS:
        with pytest.raises(expected_exception):
            func(y_true, make_y_score(), positive_label=1)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_y_score_accepts_floating_ndarray_dtypes(dtype) -> None:
    y_true = [0, 1, 0, 1]
    y_score = np.array([0.1, 0.9, 0.2, 0.8], dtype=dtype)
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.dtype == np.float64


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_y_score_accepts_integer_ndarray_dtypes(dtype) -> None:
    y_true = [0, 1, 0, 1]
    y_score = np.array([1, 9, 2, 8], dtype=dtype)
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.dtype == np.float64


def test_y_score_accepts_python_and_numpy_scalar_elements() -> None:
    y_true = [0, 1, 0, 1]
    y_score = [1, np.int64(9), 2.0, np.float32(8.0)]
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.shape[0] == 5


def test_y_score_values_outside_unit_interval_are_legal() -> None:
    y_true = [0, 1, 0, 1]
    y_score = [-100.0, 50.0, -3.0, 1000.0]
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.true_positive_rate[-1] == 1.0


def test_y_score_signed_zero_creates_one_threshold() -> None:
    roc = roc_curve([0, 1], [0.0, -0.0], positive_label=1)
    assert roc.thresholds.shape[0] == 2
    assert_array_equal(roc.false_positive_rate, [0.0, 1.0])
    assert_array_equal(roc.true_positive_rate, [0.0, 1.0])


def test_y_score_subnormal_values_are_distinct() -> None:
    roc = roc_curve([0, 1], [5e-320, 5e-321], positive_label=1)
    assert roc.thresholds.shape[0] == 3


def test_y_score_extreme_finite_float64_no_warning() -> None:
    y_true = [0, 1, 0, 1]
    y_score = [
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        roc = roc_curve(y_true, y_score, positive_label=1)
        auc = roc_auc_score(y_true, y_score, positive_label=1)
        precision_recall_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.shape[0] == 3
    assert auc == 0.0


@pytest.mark.parametrize("container", [list, tuple, _CustomScoreSequence])
def test_y_score_accepts_various_sequence_containers(container) -> None:
    y_true = [0, 1, 0, 1]
    y_score = container([0.1, 0.9, 0.2, 0.8])
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.shape[0] == 5


@pytest.mark.parametrize(("value", "legal"), [(2**53, True), (2**53 + 1, False), (2**53 + 2, True)])
def test_y_score_exact_integer_representability_boundary_sequence(value, legal) -> None:
    y_true = [0, 1, 0]
    y_score = [0.0, value, 1.0]
    if legal:
        roc = roc_curve(y_true, y_score, positive_label=1)
        assert roc.thresholds.shape[0] == 4
    else:
        with pytest.raises(ValueError):
            roc_curve(y_true, y_score, positive_label=1)


@pytest.mark.parametrize(("value", "legal"), [(2**53, True), (2**53 + 1, False), (2**53 + 2, True)])
def test_y_score_exact_integer_representability_boundary_int64_ndarray(value, legal) -> None:
    y_true = [0, 1, 0]
    y_score = np.array([0, value, 1], dtype=np.int64)
    if legal:
        roc = roc_curve(y_true, y_score, positive_label=1)
        assert roc.thresholds.shape[0] == 4
    else:
        with pytest.raises(ValueError):
            roc_curve(y_true, y_score, positive_label=1)


def test_y_score_exact_integer_representability_boundary_uint64_ndarray() -> None:
    y_true = [0, 1, 0]
    roc = roc_curve(y_true, np.array([0, 2**53 + 2, 1], dtype=np.uint64), positive_label=1)
    assert roc.thresholds.shape[0] == 4
    with pytest.raises(ValueError):
        roc_curve(y_true, np.array([0, 2**53 + 1, 1], dtype=np.uint64), positive_label=1)


def test_y_score_rejects_wider_than_float64_floating_dtype_when_present() -> None:
    if np.dtype(np.longdouble) == np.dtype(np.float64):
        pytest.skip("this platform's longdouble is identical to float64 -- nothing wider exists")
    y_true = [0, 1, 0, 1]
    y_score = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.longdouble)
    with pytest.raises(TypeError):
        roc_curve(y_true, y_score, positive_label=1)


@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_ranking_functions_reject_python_integer_outside_float64_range(value: int) -> None:
    for function in _RANKING_FUNCTIONS:
        with pytest.raises(ValueError, match="exactly representable as float64"):
            function([0, 1, 0], [0, value, 1], positive_label=1)


def test_ranking_functions_reject_python_integer_outside_float64_range_has_overflow_cause() -> None:
    with pytest.raises(ValueError) as excinfo:
        roc_curve([0, 1, 0], [0, 10**400, 1], positive_label=1)
    assert isinstance(excinfo.value.__cause__, OverflowError)


def test_y_score_large_exactly_representable_int_beyond_2_53_is_still_legal() -> None:
    # 2**60 is far beyond 2**53 (where float64 precision loss starts) but is itself an exact
    # power of two, hence exactly representable -- the overflow fix must not have turned the
    # "not exactly representable" contract into a blanket range limit.
    value = 2**60
    assert int(float(value)) == value
    for function in _RANKING_FUNCTIONS:
        function([0, 1, 0], [0, value, 1], positive_label=1)


def test_y_score_rejects_wider_numpy_floating_scalar_in_sequence() -> None:
    if np.dtype(np.longdouble) == np.dtype(np.float64):
        pytest.skip("this platform's longdouble is identical to float64 -- nothing wider exists")

    value = np.longdouble("0.5")
    for function in _RANKING_FUNCTIONS:
        with pytest.raises(TypeError, match="float64|longdouble"):
            function([0, 1], [np.longdouble("0.1"), value], positive_label=1)  # type: ignore[arg-type]


def test_y_score_wider_numpy_floating_scalar_would_create_a_false_tie_if_unchecked() -> None:
    if np.dtype(np.longdouble) == np.dtype(np.float64):
        pytest.skip("this platform's longdouble is identical to float64 -- nothing wider exists")

    lower = np.longdouble(1.0)
    higher = np.nextafter(lower, np.longdouble(2.0), dtype=np.longdouble)
    assert lower != higher
    assert float(lower) == float(higher)

    for function in _RANKING_FUNCTIONS:
        with pytest.raises(TypeError):
            function([0, 1], [lower, higher], positive_label=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_y_score_accepts_numpy_floating_scalars_of_allowed_dtype_in_sequence(dtype) -> None:
    y_true = [0, 1, 0, 1]
    y_score = [dtype(0.1), dtype(0.9), dtype(0.2), dtype(0.8)]
    for function in _RANKING_FUNCTIONS:
        function(y_true, y_score, positive_label=1)


def test_roc_curve_signed_zero_is_permutation_deterministic() -> None:
    first = roc_curve([0, 1], [0.0, -0.0], positive_label=1)
    second = roc_curve([1, 0], [-0.0, 0.0], positive_label=1)
    assert first == second
    assert first.thresholds.tobytes() == second.thresholds.tobytes()
    assert not np.signbit(first.thresholds[1])
    assert not np.signbit(second.thresholds[1])


def test_precision_recall_curve_signed_zero_is_permutation_deterministic() -> None:
    first = precision_recall_curve([0, 1], [0.0, -0.0], positive_label=1)
    second = precision_recall_curve([1, 0], [-0.0, 0.0], positive_label=1)
    assert first == second
    assert first.thresholds.tobytes() == second.thresholds.tobytes()
    assert not np.signbit(first.thresholds[1])
    assert not np.signbit(second.thresholds[1])


def test_roc_auc_score_signed_zero_is_permutation_deterministic() -> None:
    first = roc_auc_score([0, 1], [0.0, -0.0], positive_label=1)
    second = roc_auc_score([1, 0], [-0.0, 0.0], positive_label=1)
    assert first == second


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_y_score_ndarray_signed_zero_thresholds_are_always_positive_zero(dtype) -> None:
    y_score = np.array([0.0, -0.0], dtype=dtype)
    roc = roc_curve([0, 1], y_score, positive_label=1)
    assert not np.signbit(roc.thresholds[1])
    pr = precision_recall_curve([0, 1], y_score, positive_label=1)
    assert not np.signbit(pr.thresholds[1])


def test_ranking_functions_do_not_mutate_or_alias_inputs() -> None:
    y_true = [0, 0, 1, 1]
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    y_true_copy = list(y_true)
    y_score_copy = y_score.copy()

    roc = roc_curve(y_true, y_score, positive_label=1)
    precision_recall_curve(y_true, y_score, positive_label=1)
    roc_auc_score(y_true, y_score, positive_label=1)

    assert y_true == y_true_copy
    assert_array_equal(y_score, y_score_copy)
    assert not np.shares_memory(roc.thresholds, y_score)
    assert not np.shares_memory(roc.false_positive_rate, y_score)
    assert not np.shares_memory(roc.true_positive_rate, y_score)


# --- ties, sorting, threshold semantics ---


def test_number_of_real_thresholds_matches_distinct_scores() -> None:
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5]
    roc = roc_curve(y_true, y_score, positive_label=1)
    assert roc.thresholds.shape[0] == len(set(y_score)) + 1


def test_threshold_semantics_score_greater_equal_threshold() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    roc = roc_curve(y_true, y_score, positive_label=1)
    for threshold, expected_fpr, expected_tpr in zip(
        roc.thresholds, roc.false_positive_rate, roc.true_positive_rate, strict=True
    ):
        predicted_positive = [s >= threshold for s in y_score]
        tp = sum(1 for p, t in zip(predicted_positive, y_true, strict=True) if p and t == 1)
        fp = sum(1 for p, t in zip(predicted_positive, y_true, strict=True) if p and t != 1)
        assert tp / 2 == expected_tpr
        assert fp / 2 == expected_fpr


def test_roc_permutation_invariance() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.2, 0.5, 0.5, 0.5, 0.9]
    baseline_roc = roc_curve(y_true, y_score, positive_label=1)
    baseline_pr = precision_recall_curve(y_true, y_score, positive_label=1)
    baseline_auc = roc_auc_score(y_true, y_score, positive_label=1)

    rng = np.random.default_rng(0)
    indices = np.arange(len(y_true))
    for _ in range(20):
        rng.shuffle(indices)
        permuted_true = [y_true[i] for i in indices]
        permuted_score = [y_score[i] for i in indices]
        assert roc_curve(permuted_true, permuted_score, positive_label=1) == baseline_roc
        assert (
            precision_recall_curve(permuted_true, permuted_score, positive_label=1) == baseline_pr
        )
        assert roc_auc_score(permuted_true, permuted_score, positive_label=1) == baseline_auc


# --- ROC curve ---


def test_roc_curve_manual_expected_arrays() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    np.testing.assert_allclose(roc.false_positive_rate, [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(roc.true_positive_rate, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(roc.thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])
    assert roc.positive_label == 1


def test_roc_curve_perfect_separation() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], positive_label=1)
    np.testing.assert_allclose(roc.false_positive_rate, [0.0, 0.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(roc.true_positive_rate, [0.0, 0.5, 1.0, 1.0, 1.0])


def test_roc_curve_one_positive_one_negative() -> None:
    roc = roc_curve([0, 1], [0.3, 0.7], positive_label=1)
    np.testing.assert_allclose(roc.false_positive_rate, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(roc.true_positive_rate, [0.0, 1.0, 1.0])


def test_roc_curve_mixed_ties() -> None:
    roc = roc_curve([0, 1, 0, 1, 1, 0], [0.5, 0.5, 0.5, 0.9, 0.9, 0.1], positive_label=1)
    assert roc.thresholds.shape[0] == 4
    np.testing.assert_allclose(roc.false_positive_rate, [0.0, 0.0, 2 / 3, 1.0])
    np.testing.assert_allclose(roc.true_positive_rate, [0.0, 2 / 3, 1.0, 1.0])


def test_roc_curve_endpoints_and_monotonicity() -> None:
    roc = roc_curve([0, 0, 1, 1, 0, 1], [0.05, 0.4, 0.3, 0.9, 0.2, 0.6], positive_label=1)
    assert roc.thresholds[0] == np.inf
    assert roc.false_positive_rate[0] == 0.0
    assert roc.true_positive_rate[0] == 0.0
    assert roc.false_positive_rate[-1] == 1.0
    assert roc.true_positive_rate[-1] == 1.0
    rest = roc.thresholds[1:]
    assert np.all(rest[1:] < rest[:-1])
    assert np.all(roc.false_positive_rate[1:] >= roc.false_positive_rate[:-1])
    assert np.all(roc.true_positive_rate[1:] >= roc.true_positive_rate[:-1])
    n = roc.thresholds.shape[0]
    assert roc.false_positive_rate.shape == (n,)
    assert roc.true_positive_rate.shape == (n,)


def test_roc_curve_single_real_threshold_has_exactly_two_points() -> None:
    roc = roc_curve([0, 1], [0.5, 0.5], positive_label=1)
    assert roc.thresholds.shape[0] == 2
    np.testing.assert_allclose(roc.false_positive_rate, [0.0, 1.0])
    np.testing.assert_allclose(roc.true_positive_rate, [0.0, 1.0])


# --- precision-recall curve ---


def test_precision_recall_curve_manual_expected_arrays() -> None:
    pr = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    np.testing.assert_allclose(pr.precision, [1.0, 1.0, 0.5, 2.0 / 3.0, 0.5])
    np.testing.assert_allclose(pr.recall, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(pr.thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])
    assert pr.positive_label == 1


def test_precision_recall_curve_perfect_separation() -> None:
    pr = precision_recall_curve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], positive_label=1)
    np.testing.assert_allclose(pr.precision, [1.0, 1.0, 1.0, 2.0 / 3.0, 0.5])
    np.testing.assert_allclose(pr.recall, [0.0, 0.5, 1.0, 1.0, 1.0])


def test_precision_recall_curve_mixed_ties() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.9, 0.9, 0.1]
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    assert pr.thresholds.shape[0] == 4
    np.testing.assert_allclose(pr.precision, [1.0, 1.0, 0.6, 0.5])
    np.testing.assert_allclose(pr.recall, [0.0, 2 / 3, 1.0, 1.0])


def test_precision_recall_curve_endpoints_and_monotonicity() -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.05, 0.4, 0.3, 0.9, 0.2, 0.6]
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    assert pr.thresholds[0] == np.inf
    assert pr.precision[0] == 1.0
    assert pr.recall[0] == 0.0
    assert pr.recall[-1] == 1.0
    assert pr.precision[-1] == pytest.approx(3 / 6)
    assert np.all(pr.recall[1:] >= pr.recall[:-1])
    rest = pr.thresholds[1:]
    assert np.all(rest[1:] < rest[:-1])
    n = pr.thresholds.shape[0]
    assert pr.precision.shape == (n,)
    assert pr.recall.shape == (n,)


def test_precision_recall_curve_single_real_threshold_has_exactly_two_points() -> None:
    pr = precision_recall_curve([0, 1], [0.5, 0.5], positive_label=1)
    assert pr.thresholds.shape[0] == 2
    np.testing.assert_allclose(pr.precision, [1.0, 0.5])
    np.testing.assert_allclose(pr.recall, [0.0, 1.0])


# --- ROC AUC ---


def test_roc_auc_score_manual_expected_value() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1) == pytest.approx(
        0.75
    )


def test_roc_auc_perfect_separation_is_one() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], positive_label=1) == 1.0


def test_roc_auc_reverse_separation_is_zero() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], positive_label=1) == 0.0


def test_roc_auc_constant_score_is_one_half() -> None:
    assert roc_auc_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], positive_label=1) == 0.5


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([0, 1, 0, 1, 1, 0], [0.2, 0.5, 0.1, 0.9, 0.5, 0.5]),
        ([1, 1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1, 0.9]),
        ([0, 1], [0.5, 0.5]),
        ([0, 0, 0, 1], [0.3, 0.3, 0.9, 0.3]),
    ],
)
def test_roc_auc_matches_independent_mann_whitney_oracle(y_true, y_score) -> None:
    expected = _mann_whitney_auc(y_true, y_score, positive_label=1)
    actual = roc_auc_score(y_true, y_score, positive_label=1)
    assert actual == pytest.approx(expected)


def test_roc_auc_monotonic_transform_invariance() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.5, 0.1, 0.9, 0.5, 0.5]
    baseline = roc_auc_score(y_true, y_score, positive_label=1)
    transformed_score = [math.exp(s) for s in y_score]
    transformed = roc_auc_score(y_true, transformed_score, positive_label=1)
    assert transformed == pytest.approx(baseline)


def test_roc_auc_complement_relation_with_ties() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.5, 0.1, 0.9, 0.5, 0.5]
    auc = roc_auc_score(y_true, y_score, positive_label=1)
    negated_score = [-s for s in y_score]
    complement_auc = roc_auc_score(y_true, negated_score, positive_label=1)
    assert auc + complement_auc == pytest.approx(1.0)


def test_roc_auc_score_result_is_plain_float() -> None:
    result = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    assert type(result) is float


def test_roc_auc_score_bounds() -> None:
    for y_true, y_score in (
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([0, 1], [0.5, 0.5]),
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]),
    ):
        result = roc_auc_score(y_true, y_score, positive_label=1)
        assert 0.0 <= result <= 1.0


def test_roc_auc_score_does_not_use_trapz_or_trapezoid() -> None:
    import ast
    import inspect

    from improcv import evaluation

    tree = ast.parse(inspect.getsource(evaluation))
    forbidden = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("trapz", "trapezoid")
    }
    assert forbidden == set()


def test_evaluation_module_does_not_import_scipy_or_sklearn() -> None:
    import ast
    import inspect

    from improcv import evaluation

    tree = ast.parse(inspect.getsource(evaluation))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "scipy" not in imported
    assert "sklearn" not in imported


def test_roc_auc_score_bit_identical_to_pre_refactor_arithmetic_standard_example() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    core_fpr = np.array([0.0, 0.0, 0.5, 0.5, 1.0])
    core_tpr = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
    pre_refactor = float(
        np.sum(np.diff(core_fpr) * (core_tpr[:-1] + core_tpr[1:]) * 0.5, dtype=np.float64)
    )
    assert roc_auc_score(y_true, y_score, positive_label=1) == pre_refactor


def test_roc_auc_score_bit_identical_to_pre_refactor_arithmetic_mixed_ties() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.9, 0.9, 0.1]
    roc = roc_curve(y_true, y_score, positive_label=1)
    pre_refactor = float(
        np.sum(
            np.diff(roc.false_positive_rate)
            * (roc.true_positive_rate[:-1] + roc.true_positive_rate[1:])
            * 0.5,
            dtype=np.float64,
        )
    )
    assert roc_auc_score(y_true, y_score, positive_label=1) == pre_refactor


def test_roc_auc_score_bit_identical_to_pre_refactor_arithmetic_constant_scores() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.5, 0.5, 0.5, 0.5]
    roc = roc_curve(y_true, y_score, positive_label=1)
    pre_refactor = float(
        np.sum(
            np.diff(roc.false_positive_rate)
            * (roc.true_positive_rate[:-1] + roc.true_positive_rate[1:])
            * 0.5,
            dtype=np.float64,
        )
    )
    assert roc_auc_score(y_true, y_score, positive_label=1) == pre_refactor


def test_roc_auc_score_bit_identical_to_pre_refactor_arithmetic_extreme_scores() -> None:
    y_true = [0, 1, 0, 1]
    y_score = [
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
    ]
    roc = roc_curve(y_true, y_score, positive_label=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pre_refactor = float(
            np.sum(
                np.diff(roc.false_positive_rate)
                * (roc.true_positive_rate[:-1] + roc.true_positive_rate[1:])
                * 0.5,
                dtype=np.float64,
            )
        )
        result = roc_auc_score(y_true, y_score, positive_label=1)
    assert result == pre_refactor


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_roc_auc_score_bit_identical_to_pre_refactor_arithmetic_random_rankings(seed) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 30))
    y_true = rng.integers(0, 2, size=n).tolist()
    if 0 not in y_true:
        y_true[0] = 0
    if 1 not in y_true:
        y_true[1] = 1
    y_score = rng.random(n).tolist()
    roc = roc_curve(y_true, y_score, positive_label=1)
    pre_refactor = float(
        np.sum(
            np.diff(roc.false_positive_rate)
            * (roc.true_positive_rate[:-1] + roc.true_positive_rate[1:])
            * 0.5,
            dtype=np.float64,
        )
    )
    assert roc_auc_score(y_true, y_score, positive_label=1) == pre_refactor


# --- generic auc(x, y) ---


def _manual_trapezoid_oracle(x, y) -> float:
    """Independent oracle: plain Python loop over segment pairs, no NumPy trapezoid helpers."""
    total = 0.0
    for i in range(len(x) - 1):
        width = abs(x[i + 1] - x[i])
        height_sum = y[i] + y[i + 1]
        total += width * height_sum * 0.5
    return total


def test_auc_triangle_manual_area() -> None:
    # Triangle with base 2, height 2: area = 0.5 * base * height = 2.0
    assert auc([0.0, 1.0, 2.0], [0.0, 2.0, 0.0]) == 2.0


def test_auc_rectangle_manual_area() -> None:
    assert auc([0.0, 1.0, 2.0], [3.0, 3.0, 3.0]) == 6.0


def test_auc_exactly_two_points() -> None:
    assert auc([0.0, 1.0], [2.0, 4.0]) == 3.0


def test_auc_repeated_x_increasing_and_decreasing() -> None:
    assert auc([0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 5.0, 5.0]) == pytest.approx(
        _manual_trapezoid_oracle([0.0, 1.0, 1.0, 2.0], [1.0, 1.0, 5.0, 5.0])
    )
    assert auc([2.0, 1.0, 1.0, 0.0], [5.0, 5.0, 1.0, 1.0]) == pytest.approx(
        _manual_trapezoid_oracle([2.0, 1.0, 1.0, 0.0], [5.0, 5.0, 1.0, 1.0])
    )


def test_auc_constant_x_is_zero() -> None:
    assert auc([2.0, 2.0, 2.0], [1.0, 5.0, 9.0]) == 0.0


def test_auc_decreasing_x_gives_same_positive_area_as_increasing() -> None:
    x = [0.0, 0.3, 0.7, 1.0]
    y = [0.2, 0.5, 0.4, 0.9]
    increasing = auc(x, y)
    decreasing = auc(list(reversed(x)), list(reversed(y)))
    assert increasing == pytest.approx(decreasing)
    assert increasing > 0.0


def test_auc_rejects_non_monotonic_x() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        auc([0.0, 1.0, 0.5], [1.0, 2.0, 3.0])


def test_auc_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        auc([], [])


def test_auc_rejects_one_point() -> None:
    with pytest.raises(ValueError):
        auc([1.0], [2.0])


def test_auc_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        auc([0.0, 1.0, 2.0], [0.0, 1.0])


@pytest.mark.parametrize(
    ("make_bad", "expected_exception"),
    [
        (lambda: (v for v in [0.0, 1.0, 2.0]), TypeError),
        (lambda: iter([0.0, 1.0, 2.0]), TypeError),
        (lambda: "012", TypeError),
        (lambda: b"012", TypeError),
        (lambda: np.array(1.0), ValueError),
        (lambda: np.array([[0.0, 1.0], [2.0, 3.0]]), ValueError),
        (lambda: np.array([[1.0], [2.0], [3.0]]), ValueError),
        (lambda: np.array([True, False, True]), TypeError),
        (lambda: np.array([1 + 2j, 3 + 4j, 5 + 6j]), TypeError),
        (lambda: np.array([0.0, 1.0, 2.0], dtype=object), TypeError),
        (lambda: [0.0, float("nan"), 2.0], ValueError),
        (lambda: [0.0, float("inf"), 2.0], ValueError),
        (lambda: [0.0, 2**53 + 1, 2.0], ValueError),
    ],
)
def test_auc_rejects_bad_x_and_y(make_bad, expected_exception) -> None:
    good = [0.0, 1.0, 2.0]
    with pytest.raises(expected_exception):
        auc(make_bad(), good)
    with pytest.raises(expected_exception):
        auc(good, make_bad())


def test_auc_rejects_column_vector_shaped_ndarray() -> None:
    # (n, 1) is deliberately NOT silently raveled, unlike sklearn's column_or_1d.
    with pytest.raises(ValueError, match="1-D"):
        auc(np.array([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_auc_accepts_supported_floating_dtypes(dtype) -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=dtype)
    y = np.array([0.0, 1.0, 0.0], dtype=dtype)
    assert auc(x, y) == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32])
def test_auc_accepts_integer_dtypes(dtype) -> None:
    x = np.array([0, 1, 2], dtype=dtype)
    y = np.array([0, 2, 0], dtype=dtype)
    assert auc(x, y) == 2.0


def test_auc_rejects_wider_than_float64_floating_dtype_when_present() -> None:
    if np.dtype(np.longdouble) == np.dtype(np.float64):
        pytest.skip("this platform's longdouble is identical to float64 -- nothing wider exists")
    with pytest.raises(TypeError):
        auc(np.array([0.0, 1.0, 2.0], dtype=np.longdouble), [0.0, 1.0, 0.0])


def test_auc_signed_zero_x_is_treated_as_duplicate() -> None:
    assert auc([0.0, -0.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("container", [list, tuple, _CustomScoreSequence])
def test_auc_accepts_various_sequence_containers(container) -> None:
    x = container([0.0, 1.0, 2.0])
    y = container([0.0, 2.0, 0.0])
    assert auc(x, y) == 2.0


def test_auc_does_not_mutate_or_alias_inputs() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    x_copy = x.copy()
    y_copy = y.copy()
    auc(x, y)
    assert_array_equal(x, x_copy)
    assert_array_equal(y, y_copy)


def test_auc_result_is_plain_float() -> None:
    assert type(auc([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])) is float


def test_auc_result_can_be_negative() -> None:
    result = auc([0.0, 1.0, 2.0], [1.0, -1.0, -3.0])
    assert result < 0.0
    assert result == -2.0


def test_auc_result_can_exceed_unit_interval() -> None:
    result = auc([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    assert result == 40.0


# --- generic auc(x, y): overflow safety (mandatory cases from the approved corrections) ---


def test_auc_avoids_intermediate_height_sum_overflow() -> None:
    M = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc([0.0, 1.0], [M, M])
    assert result == M


def test_auc_avoids_intermediate_width_overflow_for_zero_area() -> None:
    M = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc([-M, M], [0.0, 0.0])
    assert result == 0.0


def test_auc_constant_x_with_extreme_finite_y_is_zero() -> None:
    M = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc([0.0, 0.0], [M, M])
    assert result == 0.0


def test_auc_width_overflow_with_tiny_y_gives_finite_nonzero_result() -> None:
    M = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc([-M, M], [1e-300, 1e-300])
    assert result > 0.0
    assert math.isfinite(result)


def test_auc_mixed_sign_y_gives_negative_result() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc([0.0, 1.0, 2.0], [1.0, -1.0, -3.0])
    assert result == -2.0


def test_auc_genuinely_non_representable_area_raises_value_error() -> None:
    M = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite float64"):
        auc([-M, M], [M, M])


def test_auc_extreme_arithmetic_never_warns() -> None:
    M = np.finfo(np.float64).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auc([0.0, 1.0], [M, M])
        auc([-M, M], [0.0, 0.0])
        auc([0.0, 0.0], [M, M])
        auc([-M, M], [1e-300, 1e-300])
        auc([0.0, 1.0, 2.0], [1.0, -1.0, -3.0])
        with pytest.raises(ValueError):
            auc([-M, M], [M, M])


def test_auc_matches_independent_manual_oracle_for_ordinary_data() -> None:
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(2, 15))
        x = np.sort(rng.uniform(-10, 10, n)).tolist()
        y = rng.uniform(-10, 10, n).tolist()
        expected = _manual_trapezoid_oracle(x, y)
        assert auc(x, y) == pytest.approx(expected)


# --- generic auc(x, y): exact fallback (regression -- the naive 0.5/2.0-scaled fallback
# previously raised a false ValueError for cancelling huge contributions, and separately
# underflowed a genuine subnormal residual to 0.0) ---


def _fraction_trapezoid_oracle(x, y) -> Fraction:
    """Independent oracle used only in tests: exact rational trapezoidal sum via
    `fractions.Fraction`, mirroring the production exact fallback's algorithm but written
    completely independently (no shared helper call) so it cannot share a bug with it."""
    total = Fraction()
    for i in range(len(x) - 1):
        x0 = Fraction.from_float(float(x[i]))
        x1 = Fraction.from_float(float(x[i + 1]))
        y0 = Fraction.from_float(float(y[i]))
        y1 = Fraction.from_float(float(y[i + 1]))
        total += abs(x1 - x0) * (y0 + y1) / 2
    return total


def test_auc_exact_fallback_handles_huge_cancellation_to_zero() -> None:
    M = np.finfo(np.float64).max
    x = [-M, -M / 3, M / 3, M]
    y = [M, M, -M, -M]
    assert _fraction_trapezoid_oracle(x, y) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc(x, y)
    assert result == 0.0


def test_auc_exact_fallback_handles_huge_cancellation_to_one() -> None:
    M = np.finfo(np.float64).max
    x = [-M, -M / 2, -1.0, 0.0, 1.0, M / 2, M]
    y = [M, M, 0.0, 1.0, 0.0, -M, -M]
    assert _fraction_trapezoid_oracle(x, y) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc(x, y)
    assert result == 1.0


def test_auc_exact_fallback_preserves_subnormal_residual() -> None:
    M = np.finfo(np.float64).max
    tiny = np.nextafter(0.0, 1.0)
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    y = [0.0, M, M, 0.0, -M, -M, 0.0, tiny, tiny]
    expected = float(_fraction_trapezoid_oracle(x, y))
    assert expected == 1e-323
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc(x, y)
    assert result == 1e-323
    assert result != 0.0


def test_auc_exact_fallback_cancellation_reversed_order_gives_same_area() -> None:
    M = np.finfo(np.float64).max
    x = [-M, -M / 3, M / 3, M]
    y = [M, M, -M, -M]
    forward = auc(x, y)
    reversed_result = auc(list(reversed(x)), list(reversed(y)))
    assert forward == 0.0
    assert reversed_result == pytest.approx(forward)


def test_auc_exact_fallback_huge_positive_and_negative_gives_finite_negative_result() -> None:
    M = np.finfo(np.float64).max
    # The "cancel to one" example with every y negated -- exact rational total is -1.0.
    x = [-M, -M / 2, -1.0, 0.0, 1.0, M / 2, M]
    y = [-M, -M, 0.0, -1.0, 0.0, M, M]
    expected = float(_fraction_trapezoid_oracle(x, y))
    assert expected == -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = auc(x, y)
    assert result == expected
    assert result < 0.0
    assert math.isfinite(result)


def test_auc_exact_fallback_still_raises_when_truly_not_representable() -> None:
    M = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite float64"):
        auc([-M, M], [M, M])
    # A different construction (single huge segment, not a tie/cancellation case) whose exact
    # area -- width M times height-sum 2M, halved -- genuinely exceeds float64's range.
    with pytest.raises(ValueError, match="finite float64"):
        auc([0.0, M], [M, M])


@pytest.mark.parametrize(
    ("x", "y"),
    [
        # Single overflowing segment, finite representable result (exactly M).
        ([np.finfo(np.float64).max, 0.0], [1.0, 1.0]),
        # Cancelling contributions with a different split than the named regression above.
        (
            [
                -np.finfo(np.float64).max,
                -np.finfo(np.float64).max / 4,
                np.finfo(np.float64).max / 4,
                np.finfo(np.float64).max,
            ],
            [
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                -np.finfo(np.float64).max,
                -np.finfo(np.float64).max,
            ],
        ),
        # Overflowing product from the other operand's side (large y, small integer width).
        ([0.0, 2.0], [np.finfo(np.float64).max / 2, np.finfo(np.float64).max / 2]),
    ],
)
def test_auc_exact_fallback_matches_independent_fraction_oracle(x, y) -> None:
    expected = float(_fraction_trapezoid_oracle(x, y))
    result = auc(x, y)
    assert result == expected


def test_auc_exact_fallback_does_not_mutate_or_alias_inputs() -> None:
    M = np.finfo(np.float64).max
    x = np.array([-M, -M / 3, M / 3, M], dtype=np.float64)
    y = np.array([M, M, -M, -M], dtype=np.float64)
    x_copy = x.copy()
    y_copy = y.copy()
    auc(x, y)
    assert_array_equal(x, x_copy)
    assert_array_equal(y, y_copy)


def test_auc_exact_fallback_result_is_plain_float() -> None:
    M = np.finfo(np.float64).max
    result = auc([-M, -M / 3, M / 3, M], [M, M, -M, -M])
    assert type(result) is float


def test_roc_auc_score_never_enters_exact_fallback(monkeypatch) -> None:
    from improcv import evaluation

    def _fail_if_called(x, y):
        raise AssertionError("roc_auc_score must never reach the exact fallback")

    monkeypatch.setattr(evaluation, "_trapezoidal_area_exact_fallback", _fail_if_called)

    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1) == 0.75
    y_true = [0, 1, 0, 1]
    y_score = [
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
    ]
    roc_auc_score(y_true, y_score, positive_label=1)


# --- generic auc(x, y): trapezoidal PR AUC composition (distinct from average_precision_score) ---


def test_auc_trapezoidal_pr_curve_area_example_trapezoid_larger_than_ap() -> None:
    y_true = [0, 1, 0]
    y_score = [3, 3, 2]
    curve = precision_recall_curve(y_true, y_score, positive_label=1)
    ap = average_precision_score(y_true, y_score, positive_label=1)
    trapezoid = auc(curve.recall, curve.precision)
    assert ap == 1 / 2
    assert trapezoid == 3 / 4
    assert trapezoid > ap


def test_auc_trapezoidal_pr_curve_area_example_trapezoid_smaller_than_ap() -> None:
    y_true = [0, 0, 1]
    y_score = [3, 3, 2]
    curve = precision_recall_curve(y_true, y_score, positive_label=1)
    ap = average_precision_score(y_true, y_score, positive_label=1)
    trapezoid = auc(curve.recall, curve.precision)
    assert ap == 1 / 3
    assert trapezoid == 1 / 6
    assert trapezoid < ap


# --- average precision ---


def _grouped_threshold_ap_oracle(y_true, y_score, positive_label) -> float:
    """Independent oracle: group by distinct score, then sum recall-increment-weighted
    precision using cumulative recall (never a single division applied after summing raw
    count-weighted increments) -- deliberately not the algebraically-equivalent-but-
    differently-rounded transformed form the approved audit correction rejected."""
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda item: item[0], reverse=True)
    n_positive = sum(1 for _, label in pairs if label == positive_label)

    groups: list[tuple[float, int, int]] = []
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        tp = 0
        fp = 0
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == positive_label:
                tp += 1
            else:
                fp += 1
            index += 1
        groups.append((score, tp, fp))

    cumulative_tp = 0
    cumulative_fp = 0
    previous_recall = 0.0
    ap = 0.0
    for _, tp, fp in groups:
        cumulative_tp += tp
        cumulative_fp += fp
        recall = cumulative_tp / n_positive
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def test_average_precision_standard_example() -> None:
    result = average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    assert result == pytest.approx(5 / 6)


def test_average_precision_perfect_ranking_is_one() -> None:
    result = average_precision_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], positive_label=1)
    assert result == 1.0


def test_average_precision_reverse_ranking_is_not_zero() -> None:
    # 5/12 exactly in real-number arithmetic, but the curve-based float64 computation does not
    # land on the same bit pattern as the literal Python division `5 / 12` -- pytest.approx
    # is deliberately used here (unlike the exact-equality regression below, which was built
    # specifically so the curve-based result lands on an exact bit pattern).
    result = average_precision_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], positive_label=1)
    assert result == pytest.approx(5 / 12)
    assert result != 0.0


def test_average_precision_all_positive_is_exactly_one() -> None:
    result = average_precision_score([1, 1, 1], [0.1, 0.2, 0.3], positive_label=1)
    assert result == 1.0


def test_average_precision_allows_no_negative_samples() -> None:
    result = average_precision_score([1, 1, 1], [0.1, 0.2, 0.3], positive_label=1)
    assert result == 1.0


def test_average_precision_constant_score_is_prevalence() -> None:
    result = average_precision_score([0, 0, 1, 1, 1], [0.5] * 5, positive_label=1)
    assert result == pytest.approx(3 / 5)


def test_average_precision_one_positive_no_ties() -> None:
    # A single positive at rank 2 (1-indexed, descending by score) among distinct scores:
    # precision at that rank is 1/2, and it is the only nonzero recall-increment term.
    result = average_precision_score([0, 1, 0], [0.9, 0.8, 0.5], positive_label=1)
    assert result == pytest.approx(1 / 2)


def test_average_precision_many_positives() -> None:
    y_true = [0, 1, 0, 1, 1, 0, 1]
    y_score = [0.05, 0.9, 0.1, 0.8, 0.6, 0.2, 0.4]
    result = average_precision_score(y_true, y_score, positive_label=1)
    expected = _grouped_threshold_ap_oracle(y_true, y_score, 1)
    assert result == pytest.approx(expected)


def test_average_precision_many_distinct_negative_labels() -> None:
    y_true = [5, 7, 2, 1, 5, 7]
    y_score = [0.9, 0.1, 0.2, 0.8, 0.3, 0.05]
    result = average_precision_score(y_true, y_score, positive_label=5)
    expected = _grouped_threshold_ap_oracle(y_true, y_score, 5)
    assert result == pytest.approx(expected)


def test_average_precision_mixed_ties() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.5, 0.5, 0.5, 0.9, 0.9, 0.1]
    result = average_precision_score(y_true, y_score, positive_label=1)
    expected = _grouped_threshold_ap_oracle(y_true, y_score, 1)
    assert result == pytest.approx(expected)


def test_average_precision_all_scores_tied_is_prevalence() -> None:
    result = average_precision_score([0, 1, 0, 1, 1], [0.5] * 5, positive_label=1)
    assert result == pytest.approx(3 / 5)


def test_average_precision_signed_zero_ties() -> None:
    first = average_precision_score([0, 1], [0.0, -0.0], positive_label=1)
    second = average_precision_score([1, 0], [-0.0, 0.0], positive_label=1)
    assert first == second


def test_average_precision_extreme_finite_score_no_warning() -> None:
    y_true = [0, 1, 0, 1]
    y_score = [
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
        np.finfo(np.float64).max,
        -np.finfo(np.float64).max,
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = average_precision_score(y_true, y_score, positive_label=1)
    assert 0.0 <= result <= 1.0


def test_average_precision_permutation_invariance() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.2, 0.5, 0.5, 0.5, 0.9]
    baseline = average_precision_score(y_true, y_score, positive_label=1)

    rng = np.random.default_rng(0)
    indices = np.arange(len(y_true))
    for _ in range(20):
        rng.shuffle(indices)
        permuted_true = [y_true[i] for i in indices]
        permuted_score = [y_score[i] for i in indices]
        assert average_precision_score(permuted_true, permuted_score, positive_label=1) == baseline


def test_average_precision_tie_permutation_invariance() -> None:
    # All samples in the tie group at score 0.5 are permuted independently of the rest.
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.9, 0.5, 0.5, 0.5, 0.5, 0.1]
    baseline = average_precision_score(y_true, y_score, positive_label=1)

    tie_true = y_true[1:5]
    rng = np.random.default_rng(1)
    indices = np.arange(4)
    for _ in range(10):
        rng.shuffle(indices)
        permuted = [y_true[0]] + [tie_true[i] for i in indices] + [y_true[5]]
        assert average_precision_score(permuted, y_score, positive_label=1) == baseline


def test_average_precision_strictly_increasing_transform_invariance() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.5, 0.1, 0.9, 0.5, 0.5]
    baseline = average_precision_score(y_true, y_score, positive_label=1)
    transformed_score = [math.exp(s) for s in y_score]
    transformed = average_precision_score(y_true, transformed_score, positive_label=1)
    assert transformed == pytest.approx(baseline)


def test_average_precision_score_result_is_plain_float() -> None:
    result = average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    assert type(result) is float


def test_average_precision_score_bounds() -> None:
    for y_true, y_score in (
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([0, 1], [0.5, 0.5]),
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]),
    ):
        result = average_precision_score(y_true, y_score, positive_label=1)
        assert 0.0 <= result <= 1.0


def test_average_precision_equals_public_curve_arithmetic() -> None:
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.2, 0.5, 0.1, 0.9, 0.5, 0.5]
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    expected = float(np.sum(np.diff(pr.recall) * pr.precision[1:], dtype=np.float64))
    result = average_precision_score(y_true, y_score, positive_label=1)
    assert result == expected


def test_average_precision_uses_public_curve_arithmetic_order() -> None:
    # Constructed so the algebraically-equivalent "single division after summing raw
    # count-weighted increments" form rounds to a different float64 bit pattern than the
    # public curve-based definition -- verified directly (see the approved audit correction).
    y_true = [1, 0, 0, 0, 0, 0, 1, 1, 0]
    y_score = [2, 2, 2, 2, 2, 2, 1, 1, 1]

    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    expected = float(np.sum(np.diff(pr.recall) * pr.precision[1:], dtype=np.float64))

    result = average_precision_score(y_true, y_score, positive_label=1)

    assert expected == 5 / 18
    assert result == expected

    # The rejected, algebraically-equivalent transformed form gives a different float64 bit
    # pattern on this exact input -- demonstrating why the public definition's arithmetic
    # order is a real, load-bearing part of the contract, not an implementation detail.
    true_positive = np.array([0, 1, 3], dtype=np.float64)
    false_positive = np.array([0, 5, 6], dtype=np.float64)
    group_increment = np.array([1, 2], dtype=np.float64)
    precision = true_positive[1:] / (true_positive[1:] + false_positive[1:])
    transformed = float(np.sum(group_increment * precision, dtype=np.float64) / 3)
    assert transformed != expected
    assert transformed == 0.27777777777777773


def test_average_precision_vs_trapezoidal_pr_auc_trapezoid_larger() -> None:
    y_true = [0, 1, 0]
    y_score = [3, 3, 2]
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    ap = average_precision_score(y_true, y_score, positive_label=1)
    trapezoid = float(
        np.sum(
            np.diff(pr.recall) * (pr.precision[:-1] + pr.precision[1:]) * 0.5,
            dtype=np.float64,
        )
    )
    assert ap == pytest.approx(1 / 2)
    assert trapezoid == pytest.approx(3 / 4)
    assert trapezoid > ap


def test_average_precision_vs_trapezoidal_pr_auc_trapezoid_smaller() -> None:
    y_true = [0, 0, 1]
    y_score = [3, 3, 2]
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    ap = average_precision_score(y_true, y_score, positive_label=1)
    trapezoid = float(
        np.sum(
            np.diff(pr.recall) * (pr.precision[:-1] + pr.precision[1:]) * 0.5,
            dtype=np.float64,
        )
    )
    assert ap == pytest.approx(1 / 3)
    assert trapezoid == pytest.approx(1 / 6)
    assert trapezoid < ap


# --- result types: RocCurve / PrecisionRecallCurve ---


def test_roc_curve_equality() -> None:
    a = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    b = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    c = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.36, 0.8], positive_label=1)
    assert a == b
    assert a != c
    assert a != "not a curve"
    assert (a == 5) is False


def test_roc_curve_inequality_per_field() -> None:
    baseline = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    other_fpr = RocCurve(
        false_positive_rate=np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        true_positive_rate=baseline.true_positive_rate,
        thresholds=baseline.thresholds,
        positive_label=baseline.positive_label,
    )
    other_tpr = RocCurve(
        false_positive_rate=baseline.false_positive_rate,
        true_positive_rate=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
        thresholds=baseline.thresholds,
        positive_label=baseline.positive_label,
    )
    other_thresholds = RocCurve(
        false_positive_rate=baseline.false_positive_rate,
        true_positive_rate=baseline.true_positive_rate,
        thresholds=np.array([np.inf, 0.9, 0.4, 0.35, 0.1]),
        positive_label=baseline.positive_label,
    )
    other_label = RocCurve(
        false_positive_rate=baseline.false_positive_rate,
        true_positive_rate=baseline.true_positive_rate,
        thresholds=baseline.thresholds,
        positive_label=999,
    )
    assert baseline != other_fpr
    assert baseline != other_tpr
    assert baseline != other_thresholds
    assert baseline != other_label


def test_roc_curve_unhashable() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    with pytest.raises(TypeError):
        hash(roc)


def test_roc_curve_arrays_are_read_only_and_independent() -> None:
    y_true = [0, 0, 1, 1]
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    roc = roc_curve(y_true, y_score, positive_label=1)
    for arr in (roc.false_positive_rate, roc.true_positive_rate, roc.thresholds):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.ndim == 1
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 99.0
    assert not np.shares_memory(roc.false_positive_rate, roc.true_positive_rate)
    assert not np.shares_memory(roc.false_positive_rate, roc.thresholds)
    assert not np.shares_memory(roc.true_positive_rate, roc.thresholds)
    assert not np.shares_memory(roc.thresholds, y_score)


def test_roc_curve_asdict_and_repr() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    as_dict = dataclasses.asdict(roc)
    assert set(as_dict) == {
        "false_positive_rate",
        "true_positive_rate",
        "thresholds",
        "positive_label",
    }
    assert "RocCurve" in repr(roc)


def test_roc_curve_no_nan_and_inf_only_at_index_zero() -> None:
    roc = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    assert not np.any(np.isnan(roc.false_positive_rate))
    assert not np.any(np.isnan(roc.true_positive_rate))
    assert not np.any(np.isnan(roc.thresholds))
    assert np.isinf(roc.thresholds[0])
    assert not np.any(np.isinf(roc.thresholds[1:]))


def test_precision_recall_curve_equality() -> None:
    a = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    b = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    c = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.36, 0.8], positive_label=1)
    assert a == b
    assert a != c
    assert a != "not a curve"
    assert (a == 5) is False


def test_precision_recall_curve_inequality_per_field() -> None:
    baseline = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    other_precision = PrecisionRecallCurve(
        precision=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        recall=baseline.recall,
        thresholds=baseline.thresholds,
        positive_label=baseline.positive_label,
    )
    other_recall = PrecisionRecallCurve(
        precision=baseline.precision,
        recall=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
        thresholds=baseline.thresholds,
        positive_label=baseline.positive_label,
    )
    other_thresholds = PrecisionRecallCurve(
        precision=baseline.precision,
        recall=baseline.recall,
        thresholds=np.array([np.inf, 0.9, 0.4, 0.35, 0.1]),
        positive_label=baseline.positive_label,
    )
    other_label = PrecisionRecallCurve(
        precision=baseline.precision,
        recall=baseline.recall,
        thresholds=baseline.thresholds,
        positive_label=999,
    )
    assert baseline != other_precision
    assert baseline != other_recall
    assert baseline != other_thresholds
    assert baseline != other_label


def test_precision_recall_curve_unhashable() -> None:
    pr = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    with pytest.raises(TypeError):
        hash(pr)


def test_precision_recall_curve_arrays_are_read_only_and_independent() -> None:
    y_true = [0, 0, 1, 1]
    y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float64)
    pr = precision_recall_curve(y_true, y_score, positive_label=1)
    for arr in (pr.precision, pr.recall, pr.thresholds):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.ndim == 1
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 99.0
    assert not np.shares_memory(pr.precision, pr.recall)
    assert not np.shares_memory(pr.precision, pr.thresholds)
    assert not np.shares_memory(pr.recall, pr.thresholds)
    assert not np.shares_memory(pr.thresholds, y_score)


def test_precision_recall_curve_asdict_and_repr() -> None:
    pr = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    as_dict = dataclasses.asdict(pr)
    assert set(as_dict) == {"precision", "recall", "thresholds", "positive_label"}
    assert "PrecisionRecallCurve" in repr(pr)


def test_precision_recall_curve_no_nan_and_inf_only_at_index_zero() -> None:
    pr = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], positive_label=1)
    assert not np.any(np.isnan(pr.precision))
    assert not np.any(np.isnan(pr.recall))
    assert not np.any(np.isnan(pr.thresholds))
    assert np.isinf(pr.thresholds[0])
    assert not np.any(np.isinf(pr.thresholds[1:]))
    assert im.ClassificationMetrics is ClassificationMetrics
