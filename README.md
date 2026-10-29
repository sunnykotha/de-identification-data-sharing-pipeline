# De-identification and Data-Sharing Pipeline

A Python pipeline for removing or transforming patient identifiers, creating
consistent keyed pseudonyms, enforcing k-anonymity, and reporting residual
re-identification risks.

## Features

- Models for records, the 18 HIPAA Safe Harbor identifier categories,
  pseudonym maps, quasi-identifier settings, and risk reports.
- Field-based identification and removal of recognized identifiers.
- Date generalization to year, age grouping above 89 as `90+`, and ZIP
  generalization based on a supplied population lookup.
- Free-text scrubbing using regular expressions and optional named-entity
  recognition (NER).
- Consistent HMAC-SHA256 pseudonyms using a caller-supplied secret key.
- k-anonymity checks, generalization, and suppression of records that remain
  below the configured threshold.
- Risk warnings for rare sensitive values and possible external linkage.

## Requirements

- Python 3.11 or newer
- Dependencies in `requirements.txt`
- The spaCy English model `en_core_web_sm` for the default NER-enabled scrubber

## Setup

From the project root, create and activate a virtual environment if needed:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

The default text scrubber enables named-entity recognition (NER). After
installing the project dependencies, install the English model:

    .\.venv\Scripts\python.exe -m spacy download en_core_web_sm

If you intentionally disable NER, construct the scrubber with
`FreeTextScrubber(use_ner=False)`. Regex-based scrubbing still has limitations
and should not be treated as a guarantee that all identifiers have been removed.

## Run the tests

    python -m pytest -q

## Basic usage

The following example illustrates the public pipeline API. A real export must
contain enough records in each quasi-identifier group to meet the configured
k value; records that remain below k are suppressed by default.

    from deid_pipeline.models import Record, QuasiIdentifierSet
    from deid_pipeline.pseudonymizer import Pseudonymizer
    from deid_pipeline.free_text_scrubber import FreeTextScrubber
    from deid_pipeline.pipeline import DeidentificationPipeline

    # Load this key from a secure secret store in real deployments.
    secret_key = b"replace-with-a-secure-secret-key-of-32-bytes-or-more"

    records = [
        Record(
            patient_id=f"patient-{i}",
            fields={
                "name": f"Example Patient {i}",
                "age": 93,
                "admission_date": "2025-04-18",
                "zip_code": "56401",
                "diagnosis": "example diagnosis",
            },
            quasi_identifiers={"gender": "F"},
        )
        for i in range(5)
    ]

    pipeline = DeidentificationPipeline(
        pseudonymizer=Pseudonymizer(secret_key),
        identifiers=QuasiIdentifierSet(
            fields=("age_band", "zip3", "gender"),
            k=5,
        ),
        scrubber=FreeTextScrubber(use_ner=False),
    )

    result = pipeline.run(
        records,
        zip_population_by_prefix={"564": 25000},
    )

    safe_export = result.export_dataframe
    risk_report = result.risk_report

The ZIP population mapping must come from a reliable source appropriate to
the data. The example value is illustrative, not a verified population
statistic. Never export or disclose `result.pseudonym_map`; protect it as
sensitive re-identification information.

## Important privacy limitations

- This project is an implementation aid, not a certification of HIPAA compliance or a substitute for qualified privacy and legal review.
- Regex patterns and named-entity recognition (NER) can miss identifiers, including unfamiliar field names, ambiguous text, misspellings, and identifiers not covered by configured patterns. Automated scrubbing is not exhaustive.
- Review the actual export, source-data structure, identifier-removal results, and residual-risk warnings before sharing. Document the review, its findings, and the decision to release or reject the dataset.
- Validate the pipeline against representative and adversarial data, including nested fields, unusual identifiers, dates in notes, rare diagnoses, and potential external-linkage scenarios.
- k-anonymity addresses specified quasi-identifiers only. It does not guarantee protection against attribute disclosure, all linkage attacks, or re-identification using other information.
- Confirm that every released quasi-identifier group meets the configured k threshold. The diagnostic option that disables suppression must not be treated as producing a shareable dataset.
- Keep the pseudonymization secret key and pseudonym map access-controlled and separate from research exports. The map can link pseudonyms back to patients; never include it in a shared export.
- Confirm applicable legal, contractual, institutional, and governance requirements before release. Passing automated tests or producing no warnings does not establish that a dataset is safe or legally eligible to share.

## Project layout

- src/deid_pipeline/ - implementation
- tests/ - automated tests
- examples/ - example scripts, if present
