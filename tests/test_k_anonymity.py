import pandas as pd
import pytest

from deid_pipeline.k_anonymity import KAnonymityChecker
from deid_pipeline.models import QuasiIdentifierSet, RiskReport


@pytest.fixture
def checker():
    return KAnonymityChecker(QuasiIdentifierSet(k=5))


def make_group(age, zip3, gender, count):
    return [
        {"age_band": age, "zip3": zip3, "gender": gender}
        for _ in range(count)
    ]


def test_group_size_is_counted_correctly(checker):
    data = pd.DataFrame(
        make_group("60-79", "564", "F", 5)
        + make_group("20-39", "123", "M", 2)
    )

    result = checker.check(data)

    assert result["_equivalence_class_size"].tolist() == [5] * 5 + [2] * 2
    assert result["_k_anonymous"].tolist() == [True] * 5 + [False] * 2


def test_exactly_k_records_pass(checker):
    data = pd.DataFrame(make_group("40-59", "564", "F", 5))

    assert checker.check(data)["_k_anonymous"].all()


def test_groups_below_k_are_reported(checker):
    data = pd.DataFrame(
        make_group("90+", "564", "F", 2)
        + make_group("40-59", "123", "M", 5)
    )

    failures = checker.failed_groups(data)

    assert failures == [
        {"age_band": "90+", "zip3": "564", "gender": "F", "count": 2}
    ]


def test_generalization_broadens_age_and_zip(checker):
    data = pd.DataFrame(
        [
            {"age_band": 93, "zip3": "564", "gender": "F"},
            {"age_band": 45, "zip3": "123", "gender": "M"},
            {"age_band": 25, "zip3": "987", "gender": "F"},
        ]
    )

    result = checker.generalize(data)

    assert result["age_band"].tolist() == ["80+", "40-59", "20-39"]
    assert result["zip3"].tolist() == ["500", "100", "900"]


def test_generalization_preserves_90_plus(checker):
    data = pd.DataFrame(
        [{"age_band": "90+", "zip3": "564", "gender": "F"}]
    )

    result = checker.generalize(data)

    assert result.loc[0, "age_band"] == "90+"


def test_suppression_removes_only_unsafe_records(checker):
    data = pd.DataFrame(
        make_group("40-59", "123", "F", 5)
        + make_group("90+", "564", "F", 2)
    )

    safe, report = checker.suppress(data)

    assert len(safe) == 5
    assert report.total_records == 7
    assert report.suppressed_records == 2
    assert len(report.failed_groups) == 1


def test_suppression_accepts_existing_report(checker):
    data = pd.DataFrame(make_group("40-59", "123", "F", 5))
    report = RiskReport(total_records=5)

    safe, returned_report = checker.suppress(data, report)

    assert returned_report is report
    assert len(safe) == 5
    assert report.suppressed_records == 0


def test_missing_quasi_identifier_column_is_rejected(checker):
    data = pd.DataFrame({"age_band": ["40-59"], "zip3": ["123"]})

    with pytest.raises(ValueError, match="gender"):
        checker.check(data)


def test_non_dataframe_input_is_rejected(checker):
    with pytest.raises(TypeError):
        checker.check([])


def test_empty_dataframe_is_supported(checker):
    data = pd.DataFrame(columns=["age_band", "zip3", "gender"])

    result = checker.check(data)

    assert result.empty
    assert "_k_anonymous" in result.columns
    assert checker.failed_groups(data) == []
