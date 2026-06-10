"""
Loan Data Preparation — Home Credit Default Risk Dataset
=========================================================
Downloads and prepares the Home Credit Default Risk dataset from Kaggle
for use in the LoanGuard MLOps pipeline.

SETUP (one-time):
    1. Go to https://www.kaggle.com/c/home-credit-default-risk/data
    2. Download `application_train.csv` (the main file, ~170MB)
    3. Place it in the `data/` folder of this project
    4. Run: python data/prepare_loan_data.py

OUTPUT:
    data/loans.csv — cleaned, feature-engineered dataset ready for SageMaker

The target column is `TARGET`:  1 = defaulted,  0 = repaid
"""

import os
import sys
import pandas as pd
import numpy as np

RAW_PATH = os.path.join(os.path.dirname(__file__), "application_train.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "loans.csv")

# ── Feature selection ─────────────────────────────────────────────────────────
# We keep a clean subset of the most predictive features from the dataset.
# Avoids overwhelming SageMaker with 120 columns and keeps the API simple.

FEATURES = [
    "TARGET",                   # label: 1 = default, 0 = repaid (keep first for SageMaker)
    "AMT_INCOME_TOTAL",         # applicant annual income
    "AMT_CREDIT",               # loan amount requested
    "AMT_ANNUITY",              # annual loan repayment amount
    "AMT_GOODS_PRICE",          # price of goods the loan is for
    "DAYS_BIRTH",               # age in days (negative number)
    "DAYS_EMPLOYED",            # years employed (negative = currently employed)
    "DAYS_ID_PUBLISH",          # days since ID was last updated
    "CNT_FAM_MEMBERS",          # family members count
    "CNT_CHILDREN",             # number of children
    "EXT_SOURCE_1",             # external credit score 1 (like FICO)
    "EXT_SOURCE_2",             # external credit score 2
    "EXT_SOURCE_3",             # external credit score 3
    "DAYS_LAST_PHONE_CHANGE",   # days since phone number changed
    "AMT_REQ_CREDIT_BUREAU_YEAR",  # credit bureau queries last year
    "CODE_GENDER",              # M / F
    "FLAG_OWN_CAR",             # Y / N — owns a car
    "FLAG_OWN_REALTY",          # Y / N — owns property
    "NAME_EDUCATION_TYPE",      # highest education level
    "NAME_FAMILY_STATUS",       # marital status
    "OCCUPATION_TYPE",          # job type
    "ORGANIZATION_TYPE",        # employer type
    "NAME_INCOME_TYPE",         # income source (working, pensioner, etc.)
]


def prepare(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    print(f"[prep] Reading: {raw_path}")

    if not os.path.exists(raw_path):
        print(
            f"\n[ERROR] File not found: {raw_path}\n"
            "Please download application_train.csv from:\n"
            "  https://www.kaggle.com/c/home-credit-default-risk/data\n"
            "and place it in the data/ folder.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(raw_path)
    print(f"[prep] Raw shape: {df.shape}")

    # ── Select features ───────────────────────────────────────────────────────
    available = [c for c in FEATURES if c in df.columns]
    missing   = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"[prep] Note: {len(missing)} columns not in dataset, skipping: {missing}")
    df = df[available].copy()

    # ── Drop rows where target is null ────────────────────────────────────────
    df = df.dropna(subset=["TARGET"])
    df["TARGET"] = df["TARGET"].astype(int)

    # ── Feature engineering ───────────────────────────────────────────────────
    # Convert age from negative days → positive years
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
        df.drop(columns=["DAYS_BIRTH"], inplace=True)

    # Employment: negative = currently employed (flip sign, cap anomalies)
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).clip(0, 50).round(1)
        df.drop(columns=["DAYS_EMPLOYED"], inplace=True)

    # Debt-to-income ratio
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["DEBT_TO_INCOME"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)).round(4)

    # Credit utilisation ratio
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_TO_INCOME"] = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)).round(4)

    # ── Encode binary categoricals ────────────────────────────────────────────
    for col, mapping in {
        "CODE_GENDER":     {"M": 0, "F": 1, "XNA": 0},
        "FLAG_OWN_CAR":    {"N": 0, "Y": 1},
        "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
    }.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # ── One-hot encode remaining categoricals ─────────────────────────────────
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # ── Fill remaining nulls with column median ───────────────────────────────
    df = df.fillna(df.median(numeric_only=True))

    # ── Ensure TARGET is first column (SageMaker built-in XGBoost requirement) ──
    cols = ["TARGET"] + [c for c in df.columns if c != "TARGET"]
    df = df[cols]

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(out_path, index=False)

    total = len(df)
    defaults = df["TARGET"].sum()
    print(f"[prep] ✅ Saved {total:,} rows → {out_path}")
    print(f"[prep] Default rate: {defaults/total:.1%}  ({defaults:,} defaults / {total-defaults:,} repaid)")
    print(f"[prep] Final columns ({df.shape[1]}): {list(df.columns[:8])} ...")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Home Credit loan dataset")
    parser.add_argument("--raw",  default=RAW_PATH, help="Path to application_train.csv")
    parser.add_argument("--out",  default=OUT_PATH,  help="Output path for loans.csv")
    args = parser.parse_args()
    prepare(args.raw, args.out)
