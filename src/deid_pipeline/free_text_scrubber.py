from dataclasses import dataclass, field
import re

import spacy
from spacy.language import Language

from deid_pipeline.models import Record


@dataclass
class ScrubResult:
    text: str
    detected_categories: list[str] = field(default_factory=list)


class FreeTextScrubber:
    """Scrub common identifiers from clinical notes and report detections."""

    PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "dates",
            re.compile(
                r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
                r"|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
                r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
                r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
                r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"
                r"|\b\d{1,2}(?:st|nd|rd|th)?\s+"
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
                r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
                r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{4}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "email_addresses",
            re.compile(
                r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?!\w)",
                re.IGNORECASE,
            ),
        ),
        (
            "web_urls",
            re.compile(
                r"\b(?:https?://|www\.)[^\s<>]+",
                re.IGNORECASE,
            ),
        ),
        (
            "ip_addresses",
            re.compile(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            ),
        ),
        (
            "social_security_numbers",
            re.compile(
                r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
            ),
        ),
        (
            "medical_record_numbers",
            re.compile(
                r"\b(?:MRN|medical\s+record(?:\s+number)?)"
                r"\s*[:#]?\s*[A-Z0-9-]{3,}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "health_plan_beneficiary_numbers",
            re.compile(r"\b(?:health[\s-]*plan|beneficiary|member|insurance)(?:[\s_-]*(?:id|number|no))?\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "account_numbers",
            re.compile(r"\b(?:bank\s+)?account(?:\s+number|\s+no\.?)?\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "certificate_license_numbers",
            re.compile(r"\b(?:license|licence|certificate)(?:\s+(?:number|no\.?))?\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "vehicle_identifiers",
            re.compile(r"\b(?:VIN|vehicle\s+(?:id|identification|number)|license\s+plate|plate)\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "device_identifiers",
            re.compile(r"\b(?:device|serial)(?:\s+(?:id|number|no\.?))?\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "biometric_identifiers",
            re.compile(r"\b(?:fingerprint|voiceprint|biometric)(?:\s+(?:id|identifier|data))?\s*[:#]\s*[A-Z0-9-]{3,}\b", re.IGNORECASE),
        ),
        (
            "other_unique_identifiers",
            re.compile(r"\b(?:unique\s+identifier|other\s+identifier|subject\s+id|patient\s+id|record\s+id|patient\s+identifier|record\s+identifier)\s*[:#]\s*[A-Z0-9-]{4,}\b", re.IGNORECASE),
        ),
        (
            "fax_numbers",
            re.compile(r"\b(?:fax|fax\s+number)\s*[:#]?\s*(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b", re.IGNORECASE),
        ),

        (
            "phone_numbers",
            re.compile(
                r"(?<!\w)(?:\+?1[\s.-]?)?"
                r"(?:\(\d{3}\)|\d{3})[\s.-]?"
                r"\d{3}[\s.-]?\d{4}(?!\w)"
            ),
        ),
    )

    NER_LABELS = {
        "PERSON": "names",
        "DATE": "dates",
        "GPE": "geographic_data",
        "LOC": "geographic_data",
    }

    def __init__(
        self,
        nlp: Language | None = None,
        use_ner: bool = True,
    ) -> None:
        self.use_ner = use_ner

        if nlp is not None:
            self.nlp = nlp
        elif use_ner:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None

        if self.nlp is not None and "ner" not in self.nlp.pipe_names:
            raise ValueError("The spaCy pipeline must include NER")

    @staticmethod
    def _replace_matches(
        text: str,
        pattern: re.Pattern[str],
        category: str,
    ) -> tuple[str, bool]:
        replacement = f"[REDACTED:{category}]"
        return pattern.subn(replacement, text)[0], bool(pattern.search(text))

    def scrub_with_report(self, note: str) -> ScrubResult:
        if not isinstance(note, str):
            raise TypeError("Note must be a string")

        text = note
        detected: list[str] = []

        # Apply structured patterns first, so common identifiers are
        # removed before named-entity recognition.
        for category, pattern in self.PATTERNS:
            text, matched = self._replace_matches(text, pattern, category)
            if matched and category not in detected:
                detected.append(category)

        if self.nlp is not None:
            doc = self.nlp(text)
            replacements: list[tuple[int, int, str, str]] = []

            for entity in doc.ents:
                category = self.NER_LABELS.get(entity.label_)
                if category is None:
                    continue

                replacements.append(
                    (
                        entity.start_char,
                        entity.end_char,
                        f"[REDACTED:{category}]",
                        category,
                    )
                )

            # Replace from right to left so character offsets remain valid.
            for start, end, replacement, category in sorted(
                replacements, key=lambda item: item[0], reverse=True
            ):
                text = text[:start] + replacement + text[end:]
                if category not in detected:
                    detected.append(category)

        return ScrubResult(text=text, detected_categories=detected)

    def scrub(self, note: str) -> str:
        return self.scrub_with_report(note).text

    def transform_record(self, record: Record) -> Record:
        results = [
            self.scrub_with_report(note)
            for note in record.free_text_notes
        ]
        record.free_text_notes = [result.text for result in results]
        return record
