from collections import Counter
from typing import Any

import pandas as pd

from deid_pipeline.models import QuasiIdentifierSet, RiskReport


class KAnonymityChecker:
    """Measure k-anonymity and optionally generalize or suppress records."""

    def __init__(self, identifiers: QuasiIdentifierSet | None = None):
        self.identifiers = identifiers or QuasiIdentifierSet()

    def check(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with a group-size column and anonymity flag."""
        self._validate_dataframe(dataframe)

        result = dataframe.copy()
        columns = list(self.identifiers.fields)

        result["_equivalence_class_size"] = (
            result.groupby(columns, dropna=False)["_record_marker"]
            .transform("size")
            if "_record_marker" in result.columns
            else result.assign(_record_marker=1)
            .groupby(columns, dropna=False)["_record_marker"]
            .transform("size")
        )

        result["_k_anonymous"] = (
            result["_equivalence_class_size"] >= self.identifiers.k
        )

        return result

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = set(self.identifiers.fields) - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing quasi-identifier columns: {sorted(missing)}"
            )

    def failed_groups(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        """Return quasi-identifier groups below k, with their sizes."""
        self._validate_dataframe(dataframe)

        columns = list(self.identifiers.fields)
        counts = dataframe.groupby(columns, dropna=False).size()
        failures = []

        for key, count in counts.items():
            if count < self.identifiers.k:
                values = key if isinstance(key, tuple) else (key,)
                group = dict(zip(columns, values))
                group["count"] = int(count)
                failures.append(group)

        return failures

    def generalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply one broadening pass to age and ZIP quasi-identifiers."""
        self._validate_dataframe(dataframe)

        result = dataframe.copy()

        if "age_band" in result.columns:
            result["age_band"] = result["age_band"].map(
                self._generalize_age
            )

        if "zip3" in result.columns:
            result["zip3"] = result["zip3"].map(self._generalize_zip)

        return result

    @staticmethod
    def _generalize_age(value: Any) -> Any:
        if pd.isna(value):
            return value

        text = str(value).strip()

        if text == "90+":
            return "90+"

        if text.isdigit():
            age = int(text)
            if age >= 80:
                return "80+"
            if age >= 60:
                return "60-79"
            if age >= 40:
                return "40-59"
            if age >= 20:
                return "20-39"
            return "0-19"

        return text

    @staticmethod
    def _generalize_zip(value: Any) -> Any:
        if pd.isna(value):
            return value

        text = str(value).strip()

        if text == "000":
            return "000"

        if len(text) == 3 and text.isdigit():
            return text[0] + "00"

        return text

    def suppress(
        self,
        dataframe: pd.DataFrame,
        report: RiskReport | None = None,
    ) -> tuple[pd.DataFrame, RiskReport]:
        """Exclude records in equivalence classes smaller than k."""
        checked = self.check(dataframe)
        safe = checked.loc[checked["_k_anonymous"]].drop(
            columns=["_equivalence_class_size", "_k_anonymous"],
        ).copy()

        report = report or RiskReport(total_records=len(dataframe))
        report.failed_groups = self.failed_groups(dataframe)
        report.suppressed_records += len(dataframe) - len(safe)

        return safe, report
