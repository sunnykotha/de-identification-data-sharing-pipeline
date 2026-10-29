from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from deid_pipeline.models import IdentifierField, Record


# Canonical field names are explicit to avoid accidentally deleting
# unrelated clinical fields.
FIELD_RULES: dict[IdentifierField, tuple[str, ...]] = {
    IdentifierField.NAMES: (
        "name", "patient_name", "first_name", "middle_name",
        "last_name", "full_name", "maiden_name",
    ),
    IdentifierField.GEOGRAPHIC_DATA: (
        "address", "street_address", "city", "county", "state",
        "zip", "zip_code", "postal_code", "latitude", "longitude",
    ),
    IdentifierField.DATES: (
        "date", "birth_date", "date_of_birth", "admission_date",
        "discharge_date", "service_date", "death_date",
    ),
    IdentifierField.PHONE_NUMBERS: (
        "phone", "phone_number", "mobile", "telephone",
    ),
    IdentifierField.FAX_NUMBERS: ("fax", "fax_number"),
    IdentifierField.EMAIL_ADDRESSES: ("email", "email_address"),
    IdentifierField.SOCIAL_SECURITY_NUMBERS: (
        "ssn", "social_security_number",
    ),
    IdentifierField.MEDICAL_RECORD_NUMBERS: (
        "mrn", "medical_record_number",
    ),
    IdentifierField.HEALTH_PLAN_BENEFICIARY_NUMBERS: (
        "health_plan_id", "beneficiary_number", "member_number",
        "insurance_member_id",
    ),
    IdentifierField.ACCOUNT_NUMBERS: (
        "account_number", "bank_account_number",
    ),
    IdentifierField.CERTIFICATE_LICENSE_NUMBERS: (
        "license_number", "certificate_number", "professional_license",
    ),
    IdentifierField.VEHICLE_IDENTIFIERS: (
        "vehicle_id", "vin", "license_plate",
    ),
    IdentifierField.DEVICE_IDENTIFIERS: (
        "device_id", "serial_number", "medical_device_id",
    ),
    IdentifierField.WEB_URLS: ("url", "website", "web_url"),
    IdentifierField.IP_ADDRESSES: ("ip_address", "ip"),
    IdentifierField.BIOMETRIC_IDENTIFIERS: (
        "fingerprint", "voiceprint", "biometric_id",
    ),
    IdentifierField.FULL_FACE_PHOTOGRAPHS: (
        "photo", "photograph", "face_image", "full_face_photo",
    ),
    IdentifierField.OTHER_UNIQUE_IDENTIFIERS: (
        "unique_identifier", "other_identifier", "patient_id", "patientid", "patient_identifier", "patient_identifier_value", "record_identifier_value", "subject_identifier_value", "subject_id", "subject_identifier", "record_id", "record_identifier",
    ),
}

# Direct identifiers are removed. Categories needing special treatment
# are also removed in this initial transformer and handled in later steps.
REMOVED_VALUE = "[REMOVED]"


class IdentifierTransformer:
    """Remove known identifying fields from a patient record."""

    def __init__(self) -> None:
        self._field_to_category: dict[str, IdentifierField] = {}
        for category, field_names in FIELD_RULES.items():
            for field_name in field_names:
                normalized = field_name.casefold()
                existing = self._field_to_category.get(normalized)
                if existing is not None and existing != category:
                    raise ValueError(
                        f"Field {field_name!r} belongs to multiple categories"
                    )
                self._field_to_category[normalized] = category

    @property
    def categories(self) -> tuple[IdentifierField, ...]:
        return tuple(FIELD_RULES)

    def transform(self, record: Record) -> Record:
        """Modify and return the supplied record, removing known identifiers."""
        self._transform_mapping(record.fields)
        self._transform_mapping(record.quasi_identifiers)
        return record

    def _transform_mapping(self, values: dict[str, Any]) -> None:
        for key in list(values):
            if isinstance(key, str):
                normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key.strip()).casefold()
                if normalized in self._field_to_category:
                    values[key] = REMOVED_VALUE
                    continue

            self._transform_nested_value(values[key])

    def _transform_nested_value(self, value: Any) -> None:
        if isinstance(value, dict):
            self._transform_mapping(value)
        elif isinstance(value, list):
            for item in value:
                self._transform_nested_value(item)

    def transform_mapping(
        self, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return a transformed copy without modifying the caller's mapping."""
        transformed = deepcopy(dict(values))
        self._transform_mapping(transformed)
        return transformed
