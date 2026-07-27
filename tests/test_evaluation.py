import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import improcv as im
from improcv.evaluation import (
    ClassificationMetrics,
    ConfusionMatrixResult,
    classification_metrics,
    classification_metrics_from_confusion_matrix,
    confusion_matrix,
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
    assert im.ClassificationMetrics is ClassificationMetrics
