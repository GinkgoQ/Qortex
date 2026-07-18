from __future__ import annotations

import pytest

from qortex.neuroai.public_detection import evaluate_single_image


def test_evaluate_single_image_matches_only_same_class_once() -> None:
    ground_truth = [
        {"annotation_id": 10, "category_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"annotation_id": 11, "category_id": 2, "bbox_xyxy": [20.0, 20.0, 30.0, 30.0]},
    ]
    predictions = [
        {"category_id": 1, "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"category_id": 1, "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
        {"category_id": 1, "score": 0.7, "bbox_xyxy": [20.0, 20.0, 30.0, 30.0]},
    ]

    metrics, evaluated = evaluate_single_image(predictions, ground_truth, iou_threshold=0.5)

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["mean_matched_iou"] == pytest.approx(1.0)
    assert [item["match"] for item in evaluated] == [
        "true_positive", "false_positive", "false_positive",
    ]


def test_evaluate_single_image_handles_no_predictions() -> None:
    ground_truth = [
        {"annotation_id": 10, "category_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
    ]

    metrics, evaluated = evaluate_single_image([], ground_truth, iou_threshold=0.5)

    assert evaluated == []
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_negatives"] == 1
    assert metrics["mean_matched_iou"] is None
