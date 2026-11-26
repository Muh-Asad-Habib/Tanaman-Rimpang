import pytest
from PIL import Image

from training.evaluation import Detection, FramePrediction, GroundTruth
from training.robustness import PERTURBATIONS, UNRECORDED, capture_strata, degradation_table, perturb


THRESHOLDS = {"detectorScoreThreshold": 0.25, "iouThreshold": 0.5, "classifierScoreThreshold": 0.5, "minMargin": 0.1}
BOX = (0.1, 0.1, 0.4, 0.4)


def logits_for(target):
    return tuple(8.0 if index == target else 0.0 for index in range(10))


def frame(path, target, predicted):
    truth = (GroundTruth(BOX, target, logits_for(target)),)
    detections = (Detection(BOX, 0.9, 0, logits_for(predicted)),)
    return FramePrediction(path, f"g-{path}", "test", truth, detections, 1)


def test_perturbations_are_deterministic_and_change_pixels():
    image = Image.new("RGB", (64, 48), (120, 90, 60))
    for name in PERTURBATIONS:
        first, second = perturb(image, name, "a.jpg"), perturb(image, name, "a.jpg")
        assert first.mode == "RGB" and first.size == image.size
        assert first.tobytes() == second.tobytes()
    assert perturb(image, "gelap").getpixel((0, 0))[0] < 120
    assert perturb(image, "noise", "a.jpg").tobytes() != image.tobytes()
    with pytest.raises(ValueError):
        perturb(image, "hujan")


def test_capture_strata_split_metrics_by_condition():
    frames = [frame("a.jpg", 0, 0), frame("b.jpg", 1, 1), frame("c.jpg", 2, 3)]
    records = [
        {"path": "a.jpg", "capture": {"lighting": "terang"}},
        {"path": "b.jpg", "capture": {"lighting": "terang"}},
        {"path": "c.jpg"},
    ]
    result = capture_strata(frames, records, THRESHOLDS)
    lighting = result["strata"]["lighting"]
    assert lighting["terang"]["acceptedCorrectObjectCoverage"] == 1.0
    assert lighting[UNRECORDED]["acceptedCorrectObjectCoverage"] == 0.0
    assert result["strata"]["background"] == {UNRECORDED: result["strata"]["background"][UNRECORDED]}
    assert result["recordedFrames"] == 2


def test_degradation_table_reports_deltas():
    from training.evaluation import evaluate_frames
    clean = evaluate_frames([frame("a.jpg", 0, 0), frame("b.jpg", 1, 1)], THRESHOLDS, include_cases=False)
    dark = evaluate_frames([frame("a.jpg", 0, 0), frame("b.jpg", 1, 2)], THRESHOLDS, include_cases=False)
    rows = degradation_table(clean, {"gelap": dark})
    assert rows["bersih"]["acceptedCorrectObjectCoverage"] == 1.0
    assert rows["gelap"]["deltaAcceptedCorrectObjectCoverage"] == -0.5
    assert rows["gelap"]["deltaMacroF1"] < 0
