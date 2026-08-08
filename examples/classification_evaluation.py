"""Classification predictions -> confusion matrix + multiclass ranking evaluation + curves.

Uses a small, fully explicit, hand-written multiclass example (three classes,
nine samples, no files, no model, no scikit-learn) to show
`improcv.confusion_matrix`/`improcv.classification_metrics` for label-based
evaluation, `improcv.multiclass_roc_auc_score`/
`improcv.multiclass_average_precision_score` for score-based ranking
evaluation across every supported `average` mode, and
`improcv.multiclass_roc_curve`/`improcv.multiclass_precision_recall_curve`
for the per-class curves themselves.

`labels` is deliberately given in a non-sorted order (`[20, 10, 30]`) to show
that column order in `y_score` -- and row/column order in the confusion
matrix -- is controlled entirely by the `labels` you pass, not by sorting.
`multiclass_roc_curve`/`multiclass_precision_recall_curve` preserve that same
order in `result.curves`/`result.labels`, with no `average` parameter and no
macro/weighted/micro aggregate curve (see each function's own docstring).

Run with:

    uv run python examples/classification_evaluation.py
"""

from __future__ import annotations

import math

import numpy as np

import improcv as im


def main() -> None:
    labels = [20, 10, 30]  # intentionally not sorted

    y_true = [20, 10, 30, 20, 10, 30, 20, 10, 30]
    y_pred = [20, 10, 30, 20, 30, 30, 20, 10, 10]  # two misclassifications: indices 4 and 8

    # Columns follow `labels`' own order: column 0 is label 20's one-vs-rest
    # score, column 1 is label 10's, column 2 is label 30's. Rows do not sum
    # to 1.0 -- improcv's multiclass ranking functions never require a
    # probability simplex, unlike scikit-learn's multiclass roc_auc_score.
    y_score = np.array(
        [
            [0.75, 0.40, 0.35],
            [0.30, 0.70, 0.35],
            [0.20, 0.30, 0.80],
            [0.60, 0.50, 0.55],
            [0.55, 0.45, 0.40],  # true label is 10, but column 1 is not the highest score here
            [0.25, 0.35, 0.65],
            [0.70, 0.30, 0.45],
            [0.30, 0.65, 0.30],
            [0.20, 0.30, 0.60],
        ]
    )
    assert y_score.shape == (len(y_true), len(labels))

    # rows = true labels, columns = predicted labels
    cm = im.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    metrics = im.classification_metrics(y_true=y_true, y_pred=y_pred, labels=labels, average=None)
    macro_metrics = im.classification_metrics(
        y_true=y_true, y_pred=y_pred, labels=labels, average="macro"
    )

    assert cm.matrix.shape == (len(labels), len(labels))
    assert cm.labels == tuple(labels)
    assert metrics.support.shape == (len(labels),)
    assert math.isfinite(macro_metrics.accuracy)
    assert math.isfinite(macro_metrics.f1)

    # improcv's multiclass ranking functions build each class's score from
    # the existing binary one-vs-rest core, one `labels`-ordered column at a
    # time; "micro" additionally requires all columns to share one common,
    # comparable scale, since it flattens them into a single ranking problem
    # (a stricter, explicit requirement -- not a limitation borrowed from
    # any other library).
    auc_per_class = im.multiclass_roc_auc_score(y_true, y_score, labels=labels, average=None)
    auc_macro = im.multiclass_roc_auc_score(y_true, y_score, labels=labels, average="macro")
    auc_weighted = im.multiclass_roc_auc_score(y_true, y_score, labels=labels, average="weighted")
    auc_micro = im.multiclass_roc_auc_score(y_true, y_score, labels=labels, average="micro")

    ap_per_class = im.multiclass_average_precision_score(
        y_true, y_score, labels=labels, average=None
    )
    ap_macro = im.multiclass_average_precision_score(
        y_true, y_score, labels=labels, average="macro"
    )
    ap_weighted = im.multiclass_average_precision_score(
        y_true, y_score, labels=labels, average="weighted"
    )
    ap_micro = im.multiclass_average_precision_score(
        y_true, y_score, labels=labels, average="micro"
    )

    # average=None always returns a per-class array, never a scalar --
    # narrowed explicitly here (rather than relying on a static overload)
    # since both fields are used as arrays below.
    assert isinstance(auc_per_class, np.ndarray)
    assert isinstance(ap_per_class, np.ndarray)
    for per_class_array in (auc_per_class, ap_per_class):
        assert per_class_array.shape == (len(labels),)
        assert all(math.isfinite(value) for value in per_class_array)
    for scalar in (auc_macro, auc_weighted, auc_micro, ap_macro, ap_weighted, ap_micro):
        assert math.isfinite(scalar)

    # `multiclass_roc_curve`/`multiclass_precision_recall_curve` return one binary curve per
    # label, in exactly `labels`' own order -- no `average`, no macro/weighted/micro aggregate
    # curve. `result.labels` is derived from the nested curves' own `positive_label`, never a
    # separately stored field.
    roc_curves = im.multiclass_roc_curve(y_true, y_score, labels=labels)
    pr_curves = im.multiclass_precision_recall_curve(y_true, y_score, labels=labels)

    assert roc_curves.labels == tuple(labels)
    assert pr_curves.labels == tuple(labels)
    assert len(roc_curves.curves) == len(labels)
    assert len(pr_curves.curves) == len(labels)
    for curve, label in zip(roc_curves.curves, labels, strict=True):
        assert curve.positive_label == label

    # ROC relationship: the trapezoidal area under each per-class ROC curve is bit-for-bit
    # identical to that class's own multiclass_roc_auc_score(..., average=None) entry.
    roc_auc_from_curves = [
        im.auc(curve.false_positive_rate, curve.true_positive_rate) for curve in roc_curves.curves
    ]
    for from_curve, from_score in zip(roc_auc_from_curves, auc_per_class, strict=True):
        assert from_curve == from_score

    # PR relationship: the trapezoidal area under a per-class PR curve is NOT average precision --
    # they are different definitions (non-interpolated average precision vs. trapezoidal area),
    # never guaranteed equal or guaranteed different, so this only computes both, it does not
    # assert they diverge on this particular dataset.
    pr_trapezoidal_area = [im.auc(curve.recall, curve.precision) for curve in pr_curves.curves]

    print(f"labels: {labels}")
    print("confusion matrix (rows=true, columns=predicted):")
    print(cm.matrix)
    print(f"accuracy: {macro_metrics.accuracy:.3f}")
    print(f"macro F1: {macro_metrics.f1:.3f}")
    print(f"per-class support: {metrics.support.tolist()}")
    print(f"ROC AUC per class: {[round(float(v), 3) for v in auc_per_class]}")
    print(f"ROC AUC macro/weighted/micro: {auc_macro:.3f}/{auc_weighted:.3f}/{auc_micro:.3f}")
    print(f"AP per class: {[round(float(v), 3) for v in ap_per_class]}")
    print(f"AP macro/weighted/micro: {ap_macro:.3f}/{ap_weighted:.3f}/{ap_micro:.3f}")
    print(f"ROC curve count: {len(roc_curves.curves)}, labels: {roc_curves.labels}")
    print(f"ROC AUC from curves (auc(FPR, TPR)): {[round(v, 3) for v in roc_auc_from_curves]}")
    print(f"PR curve count: {len(pr_curves.curves)}, labels: {pr_curves.labels}")
    print(
        "PR trapezoidal area (auc(recall, precision)): "
        f"{[round(v, 3) for v in pr_trapezoidal_area]}"
    )
    print(
        f"AP per class (a different quantity from PR trapezoidal area): "
        f"{[round(float(v), 3) for v in ap_per_class]}"
    )


if __name__ == "__main__":
    main()
