"""Tests for src/metrics.py on tiny predictions whose answers are worked out by hand."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import cluster_order, confusion_matrix, per_class_accuracy, top_confusions, topk_accuracy

# 3 classes, 6 images. Rows of TOP5 are the ranked predictions (only 3 classes, so top-3).
LABELS = np.array([0, 0, 1, 1, 2, 2])
TOPK = np.array([
    [0, 1, 2],  # right
    [1, 0, 2],  # wrong at top-1, right at top-2
    [1, 2, 0],  # right
    [0, 1, 2],  # wrong, true class 2nd
    [1, 0, 2],  # wrong, true class 3rd
    [1, 2, 0],  # wrong, true class 2nd
])


def test_confusion_matrix_counts_true_rows_predicted_columns():
    cm = confusion_matrix(LABELS, TOPK[:, 0], num_classes=3)
    assert cm.tolist() == [
        [1, 1, 0],
        [1, 1, 0],
        [0, 2, 0],
    ]
    assert cm.sum() == len(LABELS)


def test_topk_accuracy():
    assert topk_accuracy(LABELS, TOPK, 1) == pytest.approx(2 / 6)
    assert topk_accuracy(LABELS, TOPK, 2) == pytest.approx(5 / 6)
    assert topk_accuracy(LABELS, TOPK, 3) == pytest.approx(1.0)


def test_per_class_accuracy_is_diagonal_over_row_sum():
    cm = confusion_matrix(LABELS, TOPK[:, 0], num_classes=3)
    assert per_class_accuracy(cm).tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_per_class_accuracy_of_an_empty_class_is_nan():
    cm = np.array([[2, 0], [0, 0]])
    acc = per_class_accuracy(cm)
    assert acc[0] == 1.0 and np.isnan(acc[1])


def test_top_confusions_ranks_off_diagonal_cells_by_share_of_true_class():
    cm = np.array([
        [6, 3, 1],   # 30% of class 0 goes to class 1
        [0, 10, 0],
        [5, 0, 5],   # 50% of class 2 goes to class 0
    ])
    pairs = top_confusions(cm, k=3)
    assert [(t, p) for t, p, _, _ in pairs] == [(2, 0), (0, 1), (0, 2)]
    assert [count for _, _, count, _ in pairs] == [5, 3, 1]
    assert [share for _, _, _, share in pairs] == pytest.approx([0.5, 0.3, 0.1])


def test_top_confusions_never_returns_the_diagonal_or_empty_cells():
    cm = np.diag([5, 5, 5])
    assert top_confusions(cm, k=5) == []


def test_cluster_order_puts_mutually_confused_classes_next_to_each_other():
    # Classes {0, 3} confuse each other, and so do {1, 4}; class 2 is confused with nothing.
    cm = np.diag([50] * 5)
    cm[0, 3] = cm[3, 0] = 20
    cm[1, 4] = cm[4, 1] = 20
    order = list(cluster_order(cm))
    assert sorted(order) == [0, 1, 2, 3, 4]
    assert abs(order.index(0) - order.index(3)) == 1
    assert abs(order.index(1) - order.index(4)) == 1
