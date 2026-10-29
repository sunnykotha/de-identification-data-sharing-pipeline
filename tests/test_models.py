from dataclasses import FrozenInstanceError

import pytest

from deid_pipeline.models import (
    IdentifierField,
    PseudonymMap,
    QuasiIdentifierSet,
    Record,
    RiskReport,
)


def test_all_18_safe_harbor_identifier_categories_are_defined():
    assert len(IdentifierField) == 18


def test_record_accepts_valid_patient_data():
    record = Record(
        patient_id="patient-1",
        fields={"name": "Example Patient"},
        free_text_notes=["Follow-up visit"],
        quasi_identifiers={"age_band": "90+", "zip3": "564", "gender": "F"},
    )

    assert record.patient_id == "patient-1"
    assert record.fields["name"] == "Example Patient"
    assert len(record.free_text_notes) == 1


@pytest.mark.parametrize("patient_id", ["", "   "])
def test_record_rejects_empty_patient_id(patient_id):
    with pytest.raises(ValueError):
        Record(patient_id=patient_id)


def test_record_rejects_invalid_notes():
    with pytest.raises(TypeError):
        Record(patient_id="p1", free_text_notes="not a list")

    with pytest.raises(TypeError):
        Record(patient_id="p1", free_text_notes=[123])


def test_quasi_identifier_defaults():
    identifiers = QuasiIdentifierSet()

    assert identifiers.fields == ("age_band", "zip3", "gender")
    assert identifiers.k == 5


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_quasi_identifier_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        QuasiIdentifierSet(k=k)


def test_quasi_identifier_rejects_empty_or_duplicate_fields():
    with pytest.raises(ValueError):
        QuasiIdentifierSet(fields=())

    with pytest.raises(ValueError):
        QuasiIdentifierSet(fields=("age_band", "age_band"))


def test_quasi_identifier_set_is_immutable():
    identifiers = QuasiIdentifierSet()

    with pytest.raises(FrozenInstanceError):
        identifiers.k = 2


def test_pseudonym_map_accepts_consistent_mapping():
    mapping = PseudonymMap({"patient-1": "PSEUDO-ABC"})

    assert mapping.patient_to_pseudonym["patient-1"] == "PSEUDO-ABC"


def test_pseudonym_map_rejects_empty_values():
    with pytest.raises(ValueError):
        PseudonymMap({"": "PSEUDO-ABC"})

    with pytest.raises(ValueError):
        PseudonymMap({"patient-1": ""})


def test_risk_report_defaults():
    report = RiskReport(total_records=10)

    assert report.total_records == 10
    assert report.removed_identifiers == {}
    assert report.failed_groups == []
    assert report.suppressed_records == 0
    assert report.warnings == []


def test_risk_report_rejects_invalid_counts():
    with pytest.raises(ValueError):
        RiskReport(total_records=-1)

    with pytest.raises(ValueError):
        RiskReport(total_records=2, suppressed_records=3)
