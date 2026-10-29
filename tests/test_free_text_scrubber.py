import pytest
import spacy

from deid_pipeline.free_text_scrubber import FreeTextScrubber
from deid_pipeline.models import Record


@pytest.fixture
def scrubber():
    return FreeTextScrubber(use_ner=False)


def test_scrubs_email_address(scrubber):
    result = scrubber.scrub_with_report(
        "Contact jane@example.com for follow-up."
    )

    assert "jane@example.com" not in result.text
    assert "[REDACTED:email_addresses]" in result.text
    assert "email_addresses" in result.detected_categories


def test_scrubs_phone_number(scrubber):
    result = scrubber.scrub("Call patient at (212) 555-0198.")

    assert "(212) 555-0198" not in result
    assert "[REDACTED:phone_numbers]" in result


@pytest.mark.parametrize(
    "note",
    [
        "DOB: 2025-04-18",
        "Seen on 04/18/2025",
        "Appointment on April 18, 2025",
    ],
)
def test_scrubs_dates_in_multiple_formats(scrubber, note):
    result = scrubber.scrub(note)

    assert "2025-04-18" not in result
    assert "04/18/2025" not in result
    assert "April 18, 2025" not in result
    assert "[REDACTED:dates]" in result


def test_scrubs_social_security_number(scrubber):
    result = scrubber.scrub("SSN: 123-45-6789")

    assert "123-45-6789" not in result
    assert "[REDACTED:social_security_numbers]" in result


def test_scrubs_medical_record_number(scrubber):
    result = scrubber.scrub("MRN: ABC12345")

    assert "ABC12345" not in result
    assert "[REDACTED:medical_record_numbers]" in result


def test_scrubs_ip_address(scrubber):
    result = scrubber.scrub("Source IP: 192.168.1.10")

    assert "192.168.1.10" not in result
    assert "[REDACTED:ip_addresses]" in result


def test_scrubs_url(scrubber):
    result = scrubber.scrub("Profile: https://example.com/patient")

    assert "https://example.com/patient" not in result
    assert "[REDACTED:web_urls]" in result


def test_preserves_unrelated_clinical_text(scrubber):
    note = "Patient reports mild headache and nausea."
    assert scrubber.scrub(note) == note


def test_empty_note_is_valid(scrubber):
    assert scrubber.scrub("") == ""


def test_non_string_note_is_rejected(scrubber):
    with pytest.raises(TypeError):
        scrubber.scrub(None)


def test_scrubs_notes_in_record(scrubber):
    record = Record(
        patient_id="internal-1",
        free_text_notes=[
            "Contact jane@example.com.",
            "Patient reports mild nausea.",
        ],
    )

    result = scrubber.transform_record(record)

    assert result is record
    assert "jane@example.com" not in record.free_text_notes[0]
    assert record.free_text_notes[1] == "Patient reports mild nausea."


def test_named_entity_recognition_scrubs_person_names():
    nlp = spacy.load("en_core_web_sm")
    scrubber = FreeTextScrubber(nlp=nlp)

    result = scrubber.scrub("Barack Obama attended the appointment.")

    assert "Barack Obama" not in result
    assert "[REDACTED:names]" in result


def test_named_entity_recognition_scrubs_geographic_entities():
    nlp = spacy.load("en_core_web_sm")
    scrubber = FreeTextScrubber(nlp=nlp)

    result = scrubber.scrub("The patient lives in London.")

    assert "London" not in result
    assert "[REDACTED:geographic_data]" in result


@pytest.mark.parametrize("note, category, sensitive_value", [
    ("Member ID: PLAN-12345", "health_plan_beneficiary_numbers", "PLAN-12345"),
    ("Account number: ACCT-12345", "account_numbers", "ACCT-12345"),
    ("License number: LIC-12345", "certificate_license_numbers", "LIC-12345"),
    ("VIN: 1HGCM82633A004352", "vehicle_identifiers", "1HGCM82633A004352"),
    ("Device serial: DEV-12345", "device_identifiers", "DEV-12345"),
    ("Fingerprint ID: BIO-12345", "biometric_identifiers", "BIO-12345"),
    ("Unique identifier: UID-12345", "other_unique_identifiers", "UID-12345"),
    ("Fax: 212-555-0198", "fax_numbers", "212-555-0198"),
])
def test_scrubs_labeled_identifier_formats(scrubber, note, category, sensitive_value):
    result = scrubber.scrub_with_report(note)

    assert sensitive_value not in result.text
    assert f"[REDACTED:{category}]" in result.text
    assert category in result.detected_categories

@pytest.mark.parametrize(
    "note",
    [
        "Admitted on 18 April 2025.",
        "Appointment date: 04.18.2025.",
        "Follow-up on April 18th, 2025.",
        "Recorded on 2025/04/18.",
    ],
)
def test_scrubs_additional_free_text_date_formats(scrubber, note):
    result = scrubber.scrub(note)

    assert "[REDACTED:dates]" in result
    assert "2025" not in result
