import pandas as pd
import pytest

from deid_pipeline.risk_assessor import RiskAssessor


def test_reports_rare_diagnosis_values():
    dataframe = pd.DataFrame({
        "diagnosis": ["common"] * 6 + ["rare"],
    })

    report = RiskAssessor().assess(
        dataframe,
        sensitive_columns=("diagnosis",),
        rare_threshold=5,
        external_linkage_possible=False,
    )

    assert report.total_records == 7
    assert any("Rare values" in warning for warning in report.warnings)


def test_does_not_warn_for_values_meeting_threshold():
    dataframe = pd.DataFrame({"diagnosis": ["A"] * 5 + ["B"] * 5})

    report = RiskAssessor().assess(
        dataframe,
        sensitive_columns=("diagnosis",),
        rare_threshold=5,
        external_linkage_possible=False,
    )

    assert not any("Rare values" in warning for warning in report.warnings)


def test_warns_about_external_linkage():
    dataframe = pd.DataFrame({"diagnosis": ["A"] * 5})

    report = RiskAssessor().assess(
        dataframe,
        external_linkage_possible=True,
    )

    assert any("External dataset linkage" in warning for warning in report.warnings)


def test_linkage_warning_can_be_disabled():
    dataframe = pd.DataFrame({"diagnosis": ["A"] * 5})

    report = RiskAssessor().assess(
        dataframe,
        external_linkage_possible=False,
    )

    assert not any("External dataset linkage" in warning for warning in report.warnings)


def test_missing_sensitive_column_is_ignored():
    report = RiskAssessor().assess(
        pd.DataFrame({"age_band": ["40-59"] * 5}),
        sensitive_columns=("diagnosis",),
        external_linkage_possible=False,
    )

    assert report.total_records == 5
    assert report.warnings == []


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        RiskAssessor().assess([{"diagnosis": "A"}])


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5])
def test_rejects_invalid_rare_threshold(threshold):
    with pytest.raises(ValueError, match="rare_threshold"):
        RiskAssessor().assess(
            pd.DataFrame({"diagnosis": ["A"]}),
            rare_threshold=threshold,
        )


def test_rejects_string_instead_of_column_sequence():
    with pytest.raises(TypeError, match="sequence"):
        RiskAssessor().assess(
            pd.DataFrame({"diagnosis": ["A"]}),
            sensitive_columns="diagnosis",
        )


def test_empty_dataset_is_reported():
    report = RiskAssessor().assess(pd.DataFrame())

    assert report.total_records == 0
    assert any("No records" in warning for warning in report.warnings)
