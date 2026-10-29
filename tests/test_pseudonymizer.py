import pytest

from deid_pipeline.models import PseudonymMap
from deid_pipeline.pseudonymizer import Pseudonymizer


KEY_A = b"a" * 32
KEY_B = b"b" * 32


def test_same_patient_and_key_produce_same_pseudonym():
    pseudonymizer = Pseudonymizer(KEY_A)

    assert pseudonymizer.pseudonymize("patient-123") == (
        pseudonymizer.pseudonymize("patient-123")
    )


def test_different_patients_produce_different_pseudonyms():
    pseudonymizer = Pseudonymizer(KEY_A)

    assert pseudonymizer.pseudonymize("patient-123") != (
        pseudonymizer.pseudonymize("patient-456")
    )


def test_different_keys_produce_different_pseudonyms():
    first = Pseudonymizer(KEY_A)
    second = Pseudonymizer(KEY_B)

    assert first.pseudonymize("patient-123") != (
        second.pseudonymize("patient-123")
    )


def test_pseudonym_does_not_contain_original_patient_id():
    patient_id = "patient-123"
    pseudonym = Pseudonymizer(KEY_A).pseudonymize(patient_id)

    assert patient_id not in pseudonym
    assert pseudonym.startswith("PT-")
    assert len(pseudonym) == 67


@pytest.mark.parametrize("patient_id", ["", "   ", None])
def test_empty_or_invalid_patient_id_is_rejected(patient_id):
    with pytest.raises(ValueError):
        Pseudonymizer(KEY_A).pseudonymize(patient_id)


@pytest.mark.parametrize("key", [b"", b"short", "a" * 32, None])
def test_invalid_secret_key_is_rejected(key):
    with pytest.raises((TypeError, ValueError)):
        Pseudonymizer(key)


def test_short_key_is_rejected():
    with pytest.raises(ValueError):
        Pseudonymizer(b"x" * 31)


def test_exactly_32_byte_key_is_accepted():
    pseudonymizer = Pseudonymizer(b"x" * 32)

    assert pseudonymizer.pseudonymize("patient-1").startswith("PT-")


def test_pseudonymize_many_returns_mapping():
    pseudonymizer = Pseudonymizer(KEY_A)

    result = pseudonymizer.pseudonymize_many(
        ["patient-1", "patient-2", "patient-1"]
    )

    assert isinstance(result, PseudonymMap)
    assert len(result.patient_to_pseudonym) == 2
    assert (
        result.patient_to_pseudonym["patient-1"]
        == pseudonymizer.pseudonymize("patient-1")
    )


def test_pseudonym_mapping_is_consistent_across_calls():
    pseudonymizer = Pseudonymizer(KEY_A)

    first = pseudonymizer.pseudonymize_many(["patient-1"])
    second = pseudonymizer.pseudonymize_many(["patient-1"])

    assert (
        first.patient_to_pseudonym["patient-1"]
        == second.patient_to_pseudonym["patient-1"]
    )
