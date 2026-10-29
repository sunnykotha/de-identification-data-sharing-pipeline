import pandas as pd
import pytest

from deid_pipeline.free_text_scrubber import FreeTextScrubber
from deid_pipeline.k_anonymity import KAnonymityChecker
from deid_pipeline.models import QuasiIdentifierSet, Record
from deid_pipeline.pipeline import DeidentificationPipeline
from deid_pipeline.pseudonymizer import Pseudonymizer


@pytest.fixture
def pipeline():
    return DeidentificationPipeline(
        Pseudonymizer(b"a" * 32),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=2,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )


def make_records():
    return [
        Record(
            patient_id=f"patient-{i}",
            fields={
                "name": f"Person {i}",
                "age": 93 if i < 2 else 45,
                "admission_date": "2025-04-18",
                "zip_code": "56401",
                "diagnosis": "rare diagnosis" if i == 3 else "routine follow-up",
            },
            free_text_notes=["Contact patient at 555-123-4567"],
            quasi_identifiers={"gender": "F" if i < 2 else "M"},
        )
        for i in range(4)
    ]


def test_export_excludes_original_patient_ids_and_names(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert "patient_id" not in result.export_dataframe.columns
    assert "name" not in result.export_dataframe.columns
    assert not any(
        str(value).startswith("Person ")
        for value in result.export_dataframe.to_numpy().flatten()
    )


def test_pseudonyms_are_stable_and_present_in_export(pipeline):
    records = make_records()
    result = pipeline.run(records, zip_population_by_prefix={"564": 25000})

    assert "pseudonym" in result.export_dataframe.columns
    assert len(result.pseudonym_map.patient_to_pseudonym) == len(records)
    assert result.export_dataframe["pseudonym"].is_unique


def test_exact_age_is_never_exported(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert "age" not in result.export_dataframe.columns
    assert "age_band" in result.export_dataframe.columns


def test_allowed_zip_prefix_is_preserved_when_k_is_met(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25001},
    )

    assert set(result.export_dataframe["zip3"]) == {"564"}


def test_demographics_are_generalized(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert set(result.export_dataframe["admission_year"]) == {"2025"}
    assert set(result.export_dataframe["age_band"]) == {"90+", 45}
    assert set(result.export_dataframe["zip3"]) == {"564"}


def test_low_population_zip_is_generalized_to_zero_prefix(pipeline):
    records = make_records()[:2]
    result = pipeline.run(records, zip_population_by_prefix={"564": 20000})

    assert set(result.export_dataframe["zip3"]) == {"000"}


def test_free_text_is_scrubbed(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert all(
        "555-123-4567" not in note
        for notes in result.export_dataframe["free_text_notes"]
        for note in notes
    )


def test_unsafe_records_are_suppressed(pipeline):
    records = [
        Record(
            patient_id="one",
            fields={
                "age": 30,
                "admission_date": "2025-04-18",
                "zip_code": "56401",
            },
            quasi_identifiers={"gender": "F"},
        )
    ]

    result = pipeline.run(records, zip_population_by_prefix={"564": 25000})

    assert result.export_dataframe.empty
    assert result.risk_report.suppressed_records == 1
    assert result.risk_report.total_records == 1


def test_missing_zip_population_fails_closed(pipeline):
    with pytest.raises(ValueError, match="Missing population"):
        pipeline.run(make_records(), zip_population_by_prefix={})


def test_duplicate_patient_ids_are_rejected(pipeline):
    records = make_records()
    records[1].patient_id = records[0].patient_id

    with pytest.raises(ValueError, match="unique patient_id"):
        pipeline.run(records, zip_population_by_prefix={"564": 25000})


def test_invalid_record_input_is_rejected(pipeline):
    with pytest.raises(TypeError, match="Record"):
        pipeline.run([{"patient_id": "not-a-record"}], zip_population_by_prefix={})


def test_empty_input_returns_empty_export(pipeline):
    result = pipeline.run([], zip_population_by_prefix={})

    assert result.export_dataframe.empty
    assert result.risk_report.total_records == 0

def test_pipeline_report_includes_residual_risk_warnings(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert any("External dataset linkage" in warning
               for warning in result.risk_report.warnings)
    assert any("Rare values" in warning
               for warning in result.risk_report.warnings)


def test_pipeline_report_preserves_input_count_and_suppression_count(pipeline):
    result = pipeline.run(
        make_records(),
        zip_population_by_prefix={"564": 25000},
    )

    assert result.risk_report.total_records == 4
    assert result.risk_report.suppressed_records == 0


def test_records_still_below_k_after_generalization_are_suppressed():
    strict_pipeline = DeidentificationPipeline(
        Pseudonymizer(b"b" * 32),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=3,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )

    records = [
        Record(
            patient_id=f"strict-{i}",
            fields={
                "age": 93,
                "admission_date": "2025-04-18",
                "zip_code": "56401",
            },
            quasi_identifiers={"gender": "F"},
        )
        for i in range(2)
    ]

    result = strict_pipeline.run(
        records,
        zip_population_by_prefix={"564": 25000},
    )

    assert result.export_dataframe.empty
    assert result.risk_report.total_records == 2
    assert result.risk_report.suppressed_records == 2
    assert result.risk_report.failed_groups


def test_pipeline_accepts_already_generalized_three_digit_zip_prefix():
    pipeline = DeidentificationPipeline(
        Pseudonymizer(b"z" * 32),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=5,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )
    records = [
        Record(
            patient_id=f"prefix-{i}",
            fields={"age": 45, "admission_date": "2025-04-18"},
            quasi_identifiers={"zip3": "564", "gender": "F"},
        )
        for i in range(5)
    ]

    result = pipeline.run(
        records,
        zip_population_by_prefix={"564": 25000},
    )

    assert len(result.export_dataframe) == 5
    assert set(result.export_dataframe["zip3"]) == {"564"}


def test_disabling_suppression_warns_when_records_fail_k_anonymity():
    pipeline = DeidentificationPipeline(
        Pseudonymizer(b"y" * 32),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=3,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )
    records = [
        Record(
            patient_id=f"diagnostic-{i}",
            fields={
                "age": 93,
                "admission_date": "2025-04-18",
                "zip_code": "56401",
            },
            quasi_identifiers={"gender": "F"},
        )
        for i in range(2)
    ]

    result = pipeline.run(
        records,
        zip_population_by_prefix={"564": 25000},
        suppress_unsafe=False,
    )

    assert len(result.export_dataframe) == 2
    assert not result.export_dataframe["_k_anonymous"].any()
    assert "_equivalence_class_size" not in result.export_dataframe.columns
    assert any(
        "UNSAFE FOR SHARING" in warning
        for warning in result.risk_report.warnings
    )


def test_pipeline_accepts_year_only_date_and_pre_generalized_age():
    from deid_pipeline.models import Record, QuasiIdentifierSet
    from deid_pipeline.pseudonymizer import Pseudonymizer
    from deid_pipeline.free_text_scrubber import FreeTextScrubber
    from deid_pipeline.pipeline import DeidentificationPipeline

    pipeline = DeidentificationPipeline(
        Pseudonymizer(b"w" * 32),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=5,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )

    records = [
        Record(
            patient_id=f"year-age-{i}",
            fields={"date": "2025", "zip_code": "56401"},
            quasi_identifiers={"age_band": "90+", "gender": "F"},
        )
        for i in range(5)
    ]

    result = pipeline.run(
        records,
        zip_population_by_prefix={"564": 25000},
    )

    assert len(result.export_dataframe) == 5
    assert set(result.export_dataframe["admission_year"]) == {"2025"}
    assert set(result.export_dataframe["age_band"]) == {"90+"}


def test_end_to_end_worked_example_enforces_k5_and_removes_identifiers():
    from deid_pipeline.free_text_scrubber import FreeTextScrubber
    from deid_pipeline.models import QuasiIdentifierSet, Record
    from deid_pipeline.pipeline import DeidentificationPipeline
    from deid_pipeline.pseudonymizer import Pseudonymizer

    key = b"end-to-end-test-secret-key-32-bytes!"
    pipeline = DeidentificationPipeline(
        Pseudonymizer(key),
        identifiers=QuasiIdentifierSet(k=5),
        scrubber=FreeTextScrubber(use_ner=False),
    )

    records = []
    for index in range(7):
        gender = "F" if index < 5 else "M"
        records.append(
            Record(
                patient_id=f"original-patient-{index}",
                fields={
                    "name": f"Example Patient {index}",
                    "age": 93,
                    "admission_date": "2025-04-18",
                    "zip_code": "56401",
                },
                quasi_identifiers={"gender": gender},
            )
        )

    result = pipeline.run(
        records,
        zip_population_by_prefix={"564": 25001},
    )
    export = result.export_dataframe

    assert len(export) == 5
    assert set(export["age_band"]) == {"90+"}
    assert set(export["zip3"]) == {"564"}
    assert set(export["admission_year"]) == {"2025"}
    assert set(export["gender"]) == {"F"}
    assert export["pseudonym"].is_unique
    assert not any(
        "name" in column.casefold() or "patient_id" in column.casefold()
        for column in export.columns
    )
    assert not any(
        "original-patient-" in str(value)
        or "Example Patient" in str(value)
        for value in export.to_numpy().flatten()
    )
    assert result.risk_report.suppressed_records == 2
