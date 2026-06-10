"""
Data Validation — Loan Default Dataset
=======================================
Pandera schema for the prepared Home Credit loan dataset.
Validates feature types, null-safety, and domain-plausible value ranges
before SageMaker training starts.

Run standalone:
    python scripts/validate_data.py --csv data/loans.csv
"""

import argparse
import sys
import os
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ─────────────────────────────────────────────────────────────────────────────
# Loan Dataset Schema
# Based on the prepared Home Credit Default Risk feature set.
# Ranges reflect realistic financial values — catches bad preprocessing output.
# ─────────────────────────────────────────────────────────────────────────────

LOAN_SCHEMA = DataFrameSchema(
    columns={
        # ── Target ─────────────────────────────────────────────────────────
        "TARGET": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Loan default flag: 1 = defaulted, 0 = repaid",
        ),

        # ── Income & Credit ─────────────────────────────────────────────────
        "AMT_INCOME_TOTAL": Column(
            float,
            checks=Check.in_range(10_000, 100_000_000),
            nullable=False,
            description="Annual income (currency units)",
        ),
        "AMT_CREDIT": Column(
            float,
            checks=Check.in_range(10_000, 10_000_000),
            nullable=False,
            description="Loan amount",
        ),
        "AMT_ANNUITY": Column(
            float,
            checks=Check.in_range(1_000, 500_000),
            nullable=True,
            description="Annual repayment amount",
        ),

        # ── Applicant Demographics ──────────────────────────────────────────
        "AGE_YEARS": Column(
            float,
            checks=Check.in_range(18, 100),
            nullable=False,
            description="Applicant age in years",
        ),
        "YEARS_EMPLOYED": Column(
            float,
            checks=Check.in_range(0, 50),
            nullable=True,
            description="Years at current employer",
        ),
        "CNT_CHILDREN": Column(
            float,
            checks=Check.in_range(0, 20),
            nullable=True,
            description="Number of children",
        ),
        "CNT_FAM_MEMBERS": Column(
            float,
            checks=Check.in_range(1, 25),
            nullable=True,
            description="Number of family members",
        ),

        # ── Derived Ratios ──────────────────────────────────────────────────
        "DEBT_TO_INCOME": Column(
            float,
            checks=Check.in_range(0, 50),
            nullable=True,
            description="Annual repayment / annual income",
        ),
        "CREDIT_TO_INCOME": Column(
            float,
            checks=Check.in_range(0, 100),
            nullable=True,
            description="Total credit / annual income",
        ),

        # ── External Credit Scores ──────────────────────────────────────────
        "EXT_SOURCE_2": Column(
            float,
            checks=Check.in_range(0, 1),
            nullable=True,
            description="Normalised external credit score (0–1)",
        ),
        "EXT_SOURCE_3": Column(
            float,
            checks=Check.in_range(0, 1),
            nullable=True,
            description="Normalised external credit score (0–1)",
        ),
    },
    coerce=True,
    strict=False,   # allow extra one-hot encoded columns without failing
    name="LoanDefaultSchema",
)


def validate(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the loan CSV against the expected schema.

    Returns the validated DataFrame on success.
    Raises SystemExit(1) with a descriptive message on failure.
    """
    print(f"[validate_data] Reading: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        print(
            "Run: python data/prepare_loan_data.py  (after placing application_train.csv in data/)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[validate_data] Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Null check ────────────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        null_summary = ", ".join(f"{c}: {n}" for c, n in cols_with_nulls.items())
        print(f"[validate_data] Null counts: {null_summary}")

    # ── Target distribution ───────────────────────────────────────────────────
    if "TARGET" in df.columns:
        default_rate = df["TARGET"].mean()
        print(f"[validate_data] Default rate: {default_rate:.1%}")
        if default_rate > 0.5:
            print("[WARN] Default rate > 50% — dataset may be inverted or mislabelled")

    # ── Pandera schema validation ─────────────────────────────────────────────
    try:
        validated_df = LOAN_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        print("\n[ERROR] Schema validation failed:", file=sys.stderr)
        print(exc.failure_cases.to_string(), file=sys.stderr)
        sys.exit(1)

    print(f"[validate_data] ✅ Validation passed — {len(validated_df):,} rows ready for training")
    return validated_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate loan default CSV dataset")
    parser.add_argument("--csv", required=True, help="Path to the loans.csv file")
    args = parser.parse_args()
    validate(args.csv)
