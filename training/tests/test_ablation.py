import pytest
from PIL import Image

from training.ablation import (
    WHOLE_IMAGE_FOLDER, build_whole_image_folder, image_level_metrics, pipeline_image_prediction, single_species,
    summarize, summary_markdown, whole_image_prediction,
)
from training.common import SPLITS, taxonomy
from training.evaluation import Detection, FramePrediction, GroundTruth


THRESHOLDS = {"detectorScoreThreshold": 0.25, "iouThreshold": 0.5, "classifierScoreThreshold": 0.5, "minMargin": 0.1}


def logits_for(target, strength=8.0):
    return tuple(strength if index == target else 0.0 for index in range(10))


def frame(detections):
    truth = (GroundTruth((0.1, 0.1, 0.3, 0.3), 2, logits_for(2)),)
    return FramePrediction("a.jpg", "g", "test", truth, tuple(detections), len(detections))


def test_single_species_ignores_mixed_frames():
    assert single_species({"objects": [{"classId": 3}, {"classId": 3}]}) == 3
    assert single_species({"objects": [{"classId": 3}, {"classId": 4}]}) is None


def test_pipeline_prediction_uses_most_confident_accepted_crop():
    weak = Detection((0.1, 0.1, 0.2, 0.2), 0.9, 0, logits_for(4, 3.0))
    strong = Detection((0.6, 0.6, 0.2, 0.2), 0.8, 1, logits_for(2, 9.0))
    assert pipeline_image_prediction(frame([weak, strong]), THRESHOLDS) == 2
    unsure = Detection((0.1, 0.1, 0.2, 0.2), 0.9, 0, (0.0,) * 10)
    assert pipeline_image_prediction(frame([unsure]), THRESHOLDS) is None
    assert pipeline_image_prediction(frame([]), THRESHOLDS) is None


def test_whole_image_prediction_applies_classifier_gate():
    assert whole_image_prediction(logits_for(5), THRESHOLDS) == 5
    assert whole_image_prediction((0.0,) * 10, THRESHOLDS) is None


def test_image_metrics_count_unanswered_as_errors():
    metrics = image_level_metrics([(0, 0), (1, 1), (2, None), (3, 4)])
    assert metrics["images"] == 4 and metrics["accuracy"] == 0.5 and metrics["unanswered"] == 1
    with pytest.raises(ValueError):
        image_level_metrics([])


def test_summary_reports_deltas_against_full_model():
    full = image_level_metrics([(0, 0), (1, 1)])
    no_cbam = image_level_metrics([(0, 0), (1, 0)])
    rows = summarize({"full": full, "tanpa-cbam": no_cbam})
    assert [row["variant"] for row in rows] == ["full", "tanpa-cbam"]
    assert rows[1]["deltaAccuracy"] == -0.5
    assert "tanpa CBAM" in summary_markdown(rows)
    with pytest.raises(ValueError):
        summarize({"tanpa-cbam": no_cbam})
    with pytest.raises(ValueError):
        summarize({"full": full, "tanpa-yolo": image_level_metrics([(0, 0)])})


def test_whole_image_folder_reuses_frozen_splits(tmp_path):
    records = []
    for split in SPLITS:
        for class_id in range(10):
            records.append({"split": split, "sha256": f"{len(records):064x}", "objects": [{"classId": class_id}]})
    records.append({"split": "train", "sha256": "f" * 64, "objects": [{"classId": 0}, {"classId": 1}]})
    for index, row in enumerate(records):
        path = tmp_path / "detector" / "images" / row["split"] / f"{index:06d}_{row['sha256'][:12]}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (900, 600), (10, 120, 30)).save(path)
    counts = build_whole_image_folder(records, tmp_path)
    slugs = [label["slug"] for label in taxonomy()["labels"]]
    assert all(counts[split][slug] == 1 for split in SPLITS for slug in slugs)
    written = list((tmp_path / WHOLE_IMAGE_FOLDER).rglob("*.jpg"))
    assert len(written) == 30
    with Image.open(written[0]) as image:
        assert max(image.size) == 448
    with pytest.raises(ValueError):
        build_whole_image_folder(records, tmp_path)
