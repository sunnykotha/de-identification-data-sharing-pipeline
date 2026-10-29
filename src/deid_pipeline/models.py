from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IdentifierField(str, Enum):
    NAMES = "names"
    GEOGRAPHIC_DATA = "geographic_data"
    DATES = "dates"
    PHONE_NUMBERS = "phone_numbers"
    FAX_NUMBERS = "fax_numbers"
    EMAIL_ADDRESSES = "email_addresses"
    SOCIAL_SECURITY_NUMBERS = "social_security_numbers"
    MEDICAL_RECORD_NUMBERS = "medical_record_numbers"
    HEALTH_PLAN_BENEFICIARY_NUMBERS = "health_plan_beneficiary_numbers"
    ACCOUNT_NUMBERS = "account_numbers"
    CERTIFICATE_LICENSE_NUMBERS = "certificate_license_numbers"
    VEHICLE_IDENTIFIERS = "vehicle_identifiers"
    DEVICE_IDENTIFIERS = "device_identifiers"
    WEB_URLS = "web_urls"
    IP_ADDRESSES = "ip_addresses"
    BIOMETRIC_IDENTIFIERS = "biometric_identifiers"
    FULL_FACE_PHOTOGRAPHS = "full_face_photographs"
    OTHER_UNIQUE_IDENTIFIERS = "other_unique_identifiers"


@dataclass
class Record:
    patient_id: str
    fields: dict[str, Any] = field(default_factory=dict)
    free_text_notes: list[str] = field(default_factory=list)
    quasi_identifiers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patient_id, str) or not self.patient_id.strip():
            raise ValueError("patient_id must be a non-empty string")

        for name in ("fields", "quasi_identifiers"):
            if not isinstance(getattr(self, name), dict):
                raise TypeError(f"{name} must be a dictionary")

        if not isinstance(self.free_text_notes, list):
            raise TypeError("free_text_notes must be a list")

        if not all(isinstance(note, str) for note in self.free_text_notes):
            raise TypeError("Every free-text note must be a string")


@dataclass(frozen=True)
class QuasiIdentifierSet:
    fields: tuple[str, ...] = ("age_band", "zip3", "gender")
    k: int = 5

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("At least one quasi-identifier field is required")

        if len(set(self.fields)) != len(self.fields):
            raise ValueError("Quasi-identifier fields must be unique")

        if not all(isinstance(name, str) and name.strip() for name in self.fields):
            raise ValueError("Quasi-identifier field names must be non-empty strings")

        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be a positive integer")


@dataclass
class PseudonymMap:
    patient_to_pseudonym: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patient_to_pseudonym, dict):
            raise TypeError("patient_to_pseudonym must be a dictionary")

        if any(
            not isinstance(patient_id, str)
            or not patient_id.strip()
            or not isinstance(pseudonym, str)
            or not pseudonym.strip()
            for patient_id, pseudonym in self.patient_to_pseudonym.items()
        ):
            raise ValueError("Patient IDs and pseudonyms must be non-empty strings")


@dataclass
class RiskReport:
    total_records: int = 0
    removed_identifiers: dict[str, int] = field(default_factory=dict)
    failed_groups: list[dict[str, Any]] = field(default_factory=list)
    suppressed_records: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_records, bool)
            or not isinstance(self.total_records, int)
            or self.total_records < 0
        ):
            raise ValueError("total_records must be a non-negative integer")

        if (
            isinstance(self.suppressed_records, bool)
            or not isinstance(self.suppressed_records, int)
            or self.suppressed_records < 0
        ):
            raise ValueError("suppressed_records must be a non-negative integer")

        if self.suppressed_records > self.total_records:
            raise ValueError("suppressed_records cannot exceed total_records")
