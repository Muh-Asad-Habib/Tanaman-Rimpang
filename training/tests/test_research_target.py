from training.evaluation import RESEARCH_TARGET, research_target


def metrics(crop, precision, coverage, macro):
    return {
        "classifierGroundTruthCrops": {"unconditional": {"correctObjectCoverage": crop}},
        "endToEnd": {"acceptedPrecision": precision, "acceptedCorrectObjectCoverage": coverage, "macroF1": macro},
    }


def test_research_target_uses_phase_one_minimum():
    assert RESEARCH_TARGET["minimum"] == 0.85
    result = research_target(metrics(0.98, 0.888, 0.742, 0.80))
    measures = result["measures"]
    assert measures["classifierAccuracyGroundTruthCrops"]["achieved"] is True
    assert measures["endToEndAcceptedPrecision"]["achieved"] is True
    assert measures["endToEndCorrectObjectCoverage"]["achieved"] is False
    assert result["allAchieved"] is False


def test_research_target_handles_unmeasured_and_boundary():
    result = research_target(metrics(0.85, None, 0.85, 0.85))
    assert result["measures"]["classifierAccuracyGroundTruthCrops"]["achieved"] is True
    assert result["measures"]["endToEndAcceptedPrecision"] == {"value": None, "achieved": None}
    assert result["allAchieved"] is False
    assert research_target(metrics(0.9, 0.9, 0.9, 0.9))["allAchieved"] is True
