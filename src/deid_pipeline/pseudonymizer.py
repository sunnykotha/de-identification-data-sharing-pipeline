import hashlib
import hmac

from deid_pipeline.models import PseudonymMap


class Pseudonymizer:
    """Generate deterministic, keyed pseudonyms for patient identifiers."""

    def __init__(self, secret_key: bytes) -> None:
        if not isinstance(secret_key, bytes):
            raise TypeError("secret_key must be bytes")

        if len(secret_key) < 32:
            raise ValueError("secret_key must contain at least 32 bytes")

        self._secret_key = secret_key

    def pseudonymize(self, patient_id: str) -> str:
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("patient_id must be a non-empty string")

        digest = hmac.new(
            self._secret_key,
            patient_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return f"PT-{digest}"

    def pseudonymize_many(self, patient_ids: list[str]) -> PseudonymMap:
        if not isinstance(patient_ids, list):
            raise TypeError("patient_ids must be a list")

        mapping: dict[str, str] = {}

        for patient_id in patient_ids:
            pseudonym = self.pseudonymize(patient_id)
            mapping[patient_id] = pseudonym

        return PseudonymMap(patient_to_pseudonym=mapping)
