import pytest

from deid_pipeline.identifier_transformer import IdentifierTransformer, REMOVED_VALUE
from deid_pipeline.free_text_scrubber import FreeTextScrubber


@pytest.mark.parametrize("key", ["patient_identifier_value", "patientName"])
def test_identifier_key_variants_are_removed(key):
    transformer = IdentifierTransformer()
    result = transformer.transform_mapping({key: "SENSITIVE-123", "diagnosis": "asthma"})

    assert result[key] == REMOVED_VALUE
    assert result["diagnosis"] == "asthma"


@pytest.mark.parametrize("label, value", [("Patient ID", "ABC123"), ("Record ID", "XYZ456")])
def test_free_text_scrubs_patient_and_record_ids(label, value):
    scrubber = FreeTextScrubber(use_ner=False)
    result = scrubber.scrub(f"{label}: {value}")

    assert value not in result
    assert "[REDACTED:" in result
