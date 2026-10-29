from datetime import date, datetime

import pytest

from deid_pipeline.demographic_transformer import DemographicTransformer


@pytest.fixture
def transformer():
    return DemographicTransformer()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-04-18", "2025"),
        ("1990-12-31", "1990"),
        (date(2025, 4, 18), "2025"),
        (datetime(2025, 4, 18, 14, 30), "2025"),
    ],
)
def test_date_generalization_keeps_only_year(transformer, value, expected):
    assert transformer.generalize_date(value) == expected


@pytest.mark.parametrize("value", ["not-a-date", "2025-02-30", ""])
def test_invalid_date_is_rejected(transformer, value):
    with pytest.raises(ValueError):
        transformer.generalize_date(value)


def test_invalid_date_type_is_rejected(transformer):
    with pytest.raises(TypeError):
        transformer.generalize_date(20250418)


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0, 0),
        (18, 18),
        (89, 89),
        (90, "90+"),
        (93, "90+"),
        (130, "90+"),
    ],
)
def test_age_generalization(transformer, age, expected):
    assert transformer.generalize_age(age) == expected


@pytest.mark.parametrize("age", [-1, 131, True, 25.5])
def test_invalid_age_is_rejected(transformer, age):
    with pytest.raises(ValueError):
        transformer.generalize_age(age)


@pytest.mark.parametrize(
    ("zip_code", "population", "expected"),
    [
        ("56401", 20001, "564"),
        ("56401", 20000, "000"),
        ("56401", 1000, "000"),
        ("00501", 25000, "005"),
        ("56401-1234", 25001, "564"),
    ],
)
def test_zip_population_rule(transformer, zip_code, population, expected):
    assert transformer.generalize_zip(zip_code, population) == expected


@pytest.mark.parametrize(
    "zip_code",
    ["", "1234", "123456", "ABCDE", "12-345", "12345-678"],
)
def test_invalid_zip_code_is_rejected(transformer, zip_code):
    with pytest.raises(ValueError):
        transformer.generalize_zip(zip_code, 25001)


@pytest.mark.parametrize("population", [-1, True, 20000.5])
def test_invalid_population_is_rejected(transformer, population):
    with pytest.raises(ValueError):
        transformer.generalize_zip("56401", population)


def test_zip_requires_string(transformer):
    with pytest.raises(TypeError):
        transformer.generalize_zip(56401, 25001)


def test_worked_example_demographics(transformer):
    result = transformer.transform_demographics(
        admission_date="2025-04-18",
        age=93,
        zip_code="56401",
        three_digit_area_population=25001,
    )

    assert result == {
        "admission_year": "2025",
        "age_band": "90+",
        "zip3": "564",
    }


def test_worked_example_uses_suppressed_zip_when_population_is_too_small(
    transformer,
):
    result = transformer.transform_demographics(
        admission_date="2025-04-18",
        age=93,
        zip_code="56401",
        three_digit_area_population=20000,
    )

    assert result["zip3"] == "000"


def test_accepts_year_only_date_and_pre_generalized_age():
    from deid_pipeline.demographic_transformer import DemographicTransformer

    transformer = DemographicTransformer()

    assert transformer.generalize_date("2025") == "2025"
    assert transformer.generalize_age("90+") == "90+"
