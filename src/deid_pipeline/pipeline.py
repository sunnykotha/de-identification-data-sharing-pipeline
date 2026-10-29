from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from deid_pipeline.demographic_transformer import DemographicTransformer
from deid_pipeline.free_text_scrubber import FreeTextScrubber
from deid_pipeline.identifier_transformer import (
    FIELD_RULES,
    REMOVED_VALUE,
    IdentifierTransformer,
)
from deid_pipeline.k_anonymity import KAnonymityChecker
from deid_pipeline.models import (
    PseudonymMap,
    QuasiIdentifierSet,
    Record,
    RiskReport,
)
from deid_pipeline.pseudonymizer import Pseudonymizer
from deid_pipeline.risk_assessor import RiskAssessor


@dataclass
class PipelineResult:
    """Result of de-identification; keep pseudonym_map confidential."""

    export_dataframe: pd.DataFrame
    risk_report: RiskReport
    pseudonym_map: PseudonymMap


class DeidentificationPipeline:
    """Coordinate de-identification and k-anonymity checks."""

    def __init__(
        self,
        pseudonymizer: Pseudonymizer,
        *,
        identifiers: QuasiIdentifierSet | None = None,
        scrubber: FreeTextScrubber | None = None,
    ) -> None:
        self.pseudonymizer = pseudonymizer
        self.identifier_transformer = IdentifierTransformer()
        self.demographic_transformer = DemographicTransformer()
        self.scrubber = scrubber or FreeTextScrubber()
        self.k_checker = KAnonymityChecker(identifiers)

        self._identifier_names = {
            name.casefold()
            for names in FIELD_RULES.values()
            for name in names
        }

    def run(
        self,
        records: Sequence[Record],
        *,
        zip_population_by_prefix: Mapping[str, int],
        suppress_unsafe: bool = True,
    ) -> PipelineResult:
        """Create a de-identified export and its risk report.

        Expected demographic inputs are age, admission_date, zip_code,
        and gender. ZIP population data is keyed by the first three digits.
        Original patient IDs and the pseudonym map are never export columns.
        """
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("records must be a sequence of Record objects")

        if not isinstance(zip_population_by_prefix, Mapping):
            raise TypeError("zip_population_by_prefix must be a mapping")

        if not isinstance(suppress_unsafe, bool):
            raise TypeError("suppress_unsafe must be a boolean")

        if any(not isinstance(record, Record) for record in records):
            raise TypeError("Every item in records must be a Record")

        if len({record.patient_id for record in records}) != len(records):
            raise ValueError("Each input record must have a unique patient_id")

        if not records:
            report = RiskReport(total_records=0)
            return PipelineResult(
                export_dataframe=pd.DataFrame(
                    columns=[
                        "pseudonym",
                        *self.k_checker.identifiers.fields,
                    ]
                ),
                risk_report=report,
                pseudonym_map=PseudonymMap(),
            )

        pseudonym_map = self.pseudonymizer.pseudonymize_many(
            [record.patient_id for record in records]
        )
        rows: list[dict[str, Any]] = []
        removed_counts: dict[str, int] = {}

        for record in records:
            fields = dict(record.fields)
            qis = dict(record.quasi_identifiers)

            age = fields.get("age", qis.get("age_band"))
            admission_date = fields.get(
                "admission_date",
                fields.get("date", qis.get("admission_year")),
            )
            zip_code = fields.get(
                "zip_code",
                fields.get("zip", qis.get("zip3")),
            )
            gender = qis.get("gender", fields.get("gender"))

            if age is None or admission_date is None or zip_code is None:
                raise ValueError(
                    "Each record needs age, admission_date, and ZIP data "
                    "in fields or the corresponding quasi-identifiers"
                )

            zip_text = str(zip_code).strip()

            # Accept an already-generalized three-digit ZIP prefix.
            # The placeholder preserves the prefix for population-based
            # generalization; it does not restore the original ZIP.
            if len(zip_text) == 3 and zip_text.isdigit():
                zip_text = zip_text + "00"

            prefix = zip_text[:3]
            if len(prefix) != 3 or not prefix.isdigit():
                raise ValueError(f"Invalid ZIP prefix: {prefix!r}")

            if prefix not in zip_population_by_prefix:
                raise ValueError(
                    f"Missing population for ZIP prefix {prefix!r}"
                )

            demographics = self.demographic_transformer.transform_demographics(
                admission_date=admission_date,
                age=int(age) if str(age).isdigit() else age,
                zip_code=zip_text,
                three_digit_area_population=zip_population_by_prefix[prefix],
            )

            # Do not export source identifiers. Preserve only non-identifier
            # clinical fields and the explicitly generalized demographics.
            transformed = self.identifier_transformer.transform_mapping(fields)
            safe_fields: dict[str, Any] = {}

            for key, value in transformed.items():
                # Never export exact age; use the generalized age_band.
                if key.casefold().strip() == "age":
                    continue

                if key.casefold().strip() in self._identifier_names:
                    category = next(
                        (
                            category.value
                            for category, names in FIELD_RULES.items()
                            if key.casefold().strip() in names
                        ),
                        "unknown",
                    )
                    removed_counts[category] = (
                        removed_counts.get(category, 0) + 1
                    )
                    continue

                if value == REMOVED_VALUE:
                    continue

                safe_fields[key] = value

            scrubbed_notes = [
                self.scrubber.scrub(note)
                for note in record.free_text_notes
            ]

            row: dict[str, Any] = {
                **safe_fields,
                "pseudonym": pseudonym_map.patient_to_pseudonym[
                    record.patient_id
                ],
                **demographics,
                "gender": gender,
                "free_text_notes": scrubbed_notes,
            }
            rows.append(row)

        dataframe = pd.DataFrame(rows)

        # Enforce the configured QI columns; missing values remain visible
        # to the checker and are counted in equivalence classes.
        for column in self.k_checker.identifiers.fields:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA

        report = RiskReport(
            total_records=len(records),
            removed_identifiers=removed_counts,
        )

        # Keep the original generalization unless broadening reduces
        # the number of records in groups below k.
        generalized = dataframe.copy()
        initial_failed_groups = self.k_checker.failed_groups(generalized)
        report.failed_groups = initial_failed_groups

        if initial_failed_groups:
            candidate = self.k_checker.generalize(generalized)
            candidate_failed_groups = self.k_checker.failed_groups(candidate)

            initial_unsafe_count = sum(
                group["count"] for group in initial_failed_groups
            )
            candidate_unsafe_count = sum(
                group["count"] for group in candidate_failed_groups
            )

            if candidate_unsafe_count < initial_unsafe_count:
                generalized = candidate
                report.failed_groups = candidate_failed_groups

        if suppress_unsafe:
            export, report = self.k_checker.suppress(generalized, report)
        else:
            export = self.k_checker.check(generalized)
            if report.failed_groups:
                report.warnings.append(
                    "UNSAFE FOR SHARING: suppression is disabled and "
                    "one or more quasi-identifier groups remain below k."
                )

        # Never include original identifiers or the internal map in the export.
        export = export.drop(
            columns=[
                column
                for column in ("patient_id", "_record_marker", "_equivalence_class_size")
                if column in export.columns
            ],
            errors="ignore",
        )

        residual_risk = RiskAssessor().assess(
            export,
            sensitive_columns=("diagnosis",),
            rare_threshold=self.k_checker.identifiers.k,
            external_linkage_possible=True,
        )
        report.warnings.extend(residual_risk.warnings)

        return PipelineResult(
            export_dataframe=export.reset_index(drop=True),
            risk_report=report,
            pseudonym_map=pseudonym_map,
        )
