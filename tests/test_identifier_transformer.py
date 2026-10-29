import pytest

from deid_pipeline.identifier_transformer import (
    FIELD_RULES,
    REMOVED_VALUE,
    IdentifierTransformer,
)
from deid_pipeline.models import IdentifierField, Record


@pytest.fixture
def transformer():
    return IdentifierTransformer()


def test_transformer_covers_all_18_categories(transformer):
    assert set(transformer.categories) == set(IdentifierField)
    assert len(transformer.categories) == 18


@pytest.mark.parametrize(
    "field_name",
    sorted({
        name
        for field_names in FIELD_RULES.values()
        for name in field_names
    }),
)
def test_every_registered_identifier_field_is_removed(
    transformer, field_name
):
    result = transformer.transform_mapping({field_name: "sensitive-value"})

    assert result[field_name] == REMOVED_VALUE


def test_transformer_handles_field_name_case_and_surrounding_spaces(
    transformer,
):
    result = transformer.transform_mapping({
        " Patient_Name ": "Jane Doe",
        "EMAIL": "jane@example.com",
    })

    assert result[" Patient_Name "] == REMOVED_VALUE
    assert result["EMAIL"] == REMOVED_VALUE


def test_transformer_preserves_unrelated_clinical_fields(transformer):
    source = {
        "diagnosis": "hypertension",
        "procedure_code": "99213",
        "blood_pressure": "120/80",
    }

    assert transformer.transform_mapping(source) == source


def test_transform_mapping_does_not_mutate_input(transformer):
    source = {"name": "Jane Doe", "diagnosis": "asthma"}

    transformer.transform_mapping(source)

    assert source["name"] == "Jane Doe"


def test_transformer_updates_record_fields_and_quasi_identifiers(
    transformer,
):
    record = Record(
        patient_id="internal-123",
        fields={
            "name": "Jane Doe",
            "medical_record_number": "MRN-123",
            "diagnosis": "asthma",
        },
        quasi_identifiers={"zip_code": "56401", "gender": "F"},
    )

    result = transformer.transform(record)

    assert result is record
    assert record.fields["name"] == REMOVED_VALUE
    assert record.fields["medical_record_number"] == REMOVED_VALUE
    assert record.fields["diagnosis"] == "asthma"
    assert record.quasi_identifiers["zip_code"] == REMOVED_VALUE
    assert record.quasi_identifiers["gender"] == "F"


def test_identifier_field_categories_are_explicit():
    assert IdentifierField.NAMES in FIELD_RULES
    assert IdentifierField.FULL_FACE_PHOTOGRAPHS in FIELD_RULES
    assert IdentifierField.OTHER_UNIQUE_IDENTIFIERS in FIELD_RULES


def test_nested_identifiers_are_removed_recursively():
    from deid_pipeline.identifier_transformer import IdentifierTransformer

    transformer = IdentifierTransformer()
    source = {
        'clinical': {
            'patient_id': 'patient-123',
            'diagnosis': 'asthma',
        },
        'events': [
            {'record_id': 'record-456', 'note': 'follow-up'},
            {'details': {'subject_identifier': 'subject-789'}},
        ],
    }

    transformed = transformer.transform_mapping(source)

    assert transformed['clinical']['patient_id'] == '[REMOVED]'
    assert transformed['clinical']['diagnosis'] == 'asthma'
    assert transformed['events'][0]['record_id'] == '[REMOVED]'
    assert transformed['events'][0]['note'] == 'follow-up'
    assert transformed['events'][1]['details']['subject_identifier'] == '[REMOVED]'
    assert source['clinical']['patient_id'] == 'patient-123'
