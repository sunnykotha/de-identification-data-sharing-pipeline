from collections import Counter
from typing import Any, Sequence

import pandas as pd

from deid_pipeline.models import RiskReport


class RiskAssessor:
    """Report residual risks that k-anonymity alone cannot address."""

    def assess(
        self,
        dataframe: pd.DataFrame,
        *,
        sensitive_columns: Sequence[str] = ("diagnosis",),
        rare_threshold: int = 5,
        external_linkage_possible: bool = True,
    ) -> RiskReport:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if isinstance(rare_threshold, bool) or not isinstance(rare_threshold, int):
            raise ValueError("rare_threshold must be a positive integer")
        if rare_threshold < 1:
            raise ValueError("rare_threshold must be a positive integer")

        if isinstance(sensitive_columns, (str, bytes)):
            raise TypeError("sensitive_columns must be a sequence of column names")

        if not all(isinstance(column, str) and column.strip()
                   for column in sensitive_columns):
            raise ValueError("Sensitive column names must be non-empty strings")

        report = RiskReport(total_records=len(dataframe))

        for column in sensitive_columns:
            if column not in dataframe.columns:
                continue

            values = dataframe[column].dropna()
            counts = Counter(values.tolist())

            rare_values = [
                value for value, count in counts.items()
                if count < rare_threshold
            ]

            if rare_values:
                report.warnings.append(
                    f"Rare values in sensitive column {column!r} occur fewer "
                    f"than {rare_threshold} times; review or generalize them."
                )

        if external_linkage_possible and not dataframe.empty:
            report.warnings.append(
                "External dataset linkage may re-identify records even when "
                "quasi-identifier groups satisfy k-anonymity."
            )

        if dataframe.empty:
            report.warnings.append(
                "No records were available for residual-risk assessment."
            )

        return report
