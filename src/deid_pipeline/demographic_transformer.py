from datetime import date, datetime
import re
from typing import Any


class DemographicTransformer:
    """Generalize dates, ZIP codes, and ages for de-identification."""

    ZIP_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")

    @staticmethod
    def generalize_date(value: date | datetime | str) -> str:
        if isinstance(value, datetime):
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        elif isinstance(value, str):
            normalized = value.strip()
            if re.fullmatch(r"\d{4}", normalized):
                year = int(normalized)
                if year < 1:
                    raise ValueError("Date must be a valid ISO date")
                return normalized
            try:
                parsed = date.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError("Date must be a valid ISO date") from exc
        else:
            raise TypeError("Date must be a date, datetime, or ISO date string")

        return str(parsed.year)

    @staticmethod
    def generalize_age(age: int | str) -> int | str:
        if isinstance(age, str) and age.strip() == "90+":
            return "90+"

        if isinstance(age, bool) or not isinstance(age, int):
            raise ValueError("Age must be an integer or the generalized value '90+'")

        if age < 0 or age > 130:
            raise ValueError("Age must be between 0 and 130")

        if age > 89:
            return "90+"

        return age

    @classmethod
    def generalize_zip(
        cls,
        zip_code: str,
        three_digit_area_population: int,
    ) -> str:
        if not isinstance(zip_code, str):
            raise TypeError("ZIP code must be a string")

        normalized = zip_code.strip()

        if not cls.ZIP_PATTERN.fullmatch(normalized):
            raise ValueError("ZIP code must be 5 digits or ZIP+4")

        if (
            isinstance(three_digit_area_population, bool)
            or not isinstance(three_digit_area_population, int)
            or three_digit_area_population < 0
        ):
            raise ValueError("Population must be a non-negative integer")

        if three_digit_area_population > 20_000:
            return normalized[:3]

        return "000"

    def transform_demographics(
        self,
        *,
        admission_date: date | datetime | str,
        age: int,
        zip_code: str,
        three_digit_area_population: int,
    ) -> dict[str, Any]:
        return {
            "admission_year": self.generalize_date(admission_date),
            "age_band": self.generalize_age(age),
            "zip3": self.generalize_zip(
                zip_code,
                three_digit_area_population,
            ),
        }
