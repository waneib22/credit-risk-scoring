"""Nodes du pipeline data_processing : raw → features → split.

Reproduit la logique des notebooks 01 (chargement) et 03 (feature engineering,
encodage, split) sous forme de fonctions pures orchestrées par Kedro.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# Colonnes du Standard Dataset Freddie Mac (format fixe → constantes, pas des params)
ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer_flag", "maturity_date",
    "msa", "mip", "units", "occupancy_status", "ocltv", "dti", "original_upb", "oltv",
    "original_interest_rate", "channel", "ppm_flag", "amortization_type", "property_state",
    "property_type", "postal_code", "loan_sequence_number", "loan_purpose",
    "original_loan_term", "number_of_borrowers", "seller_name", "servicer_name",
    "super_conforming_flag", "pre_harp_loan_sequence_number", "program_indicator",
    "harp_indicator", "property_valuation_method", "interest_only_indicator",
    "mi_cancellation_indicator",
]
PERF_COLS = [
    "loan_sequence_number", "monthly_reporting_period", "current_actual_upb",
    "current_loan_delinquency_status", "loan_age", "remaining_months_to_legal_maturity",
    "defect_settlement_date", "modification_flag", "zero_balance_code",
    "zero_balance_effective_date", "current_interest_rate", "current_deferred_upb",
    "ddlpi", "mi_recoveries", "net_sales_proceeds", "non_mi_recoveries", "expenses",
    "legal_costs", "maintenance_and_preservation_costs", "taxes_and_insurance",
    "miscellaneous_expenses", "actual_loss_calculation", "modification_cost",
    "step_modification_flag", "deferred_payment_plan", "estimated_ltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest", "delinquency_due_to_disaster",
    "borrower_assistance_status_code", "current_month_modification_cost", "interest_bearing_upb",
]


# --------------------------------------------------------------------------- #
# 1. Chargement des fichiers bruts
# --------------------------------------------------------------------------- #
def load_raw_data(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit les fichiers origination (orig) et performance (svcg) de data/raw."""
    raw_dir = params["raw_dir"]
    orig_frames, perf_frames = [], []

    for path in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        name = os.path.basename(path).lower()
        if "orig" in name:
            df = pd.read_csv(path, sep="|", header=None, names=ORIG_COLS,
                             dtype=str, encoding="latin-1", low_memory=False)
            assert df.shape[1] == len(ORIG_COLS), f"{name}: {df.shape[1]} cols"
            orig_frames.append(df)
        elif "svcg" in name or "time" in name:
            df = pd.read_csv(path, sep="|", header=None, names=PERF_COLS,
                             dtype=str, encoding="latin-1", low_memory=False)
            assert df.shape[1] == len(PERF_COLS), f"{name}: {df.shape[1]} cols"
            perf_frames.append(df)

    assert orig_frames, f"Aucun fichier origination dans {raw_dir}"
    assert perf_frames, f"Aucun fichier performance dans {raw_dir}"

    df_orig = pd.concat(orig_frames, ignore_index=True)
    df_perf = pd.concat(perf_frames, ignore_index=True)
    return df_orig, df_perf


# --------------------------------------------------------------------------- #
# 2. Construction de la cible à partir de la performance
# --------------------------------------------------------------------------- #
def build_target(df_perf: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Agrège la performance mensuelle en une ligne par prêt + cible `default`."""
    df_perf = df_perf.copy()
    df_perf["current_loan_delinquency_status"] = pd.to_numeric(
        df_perf["current_loan_delinquency_status"].replace({"X": np.nan, "XX": np.nan, "RA": np.nan}),
        errors="coerce",
    )
    df_perf["loan_age"] = pd.to_numeric(df_perf["loan_age"], errors="coerce")

    agg = df_perf.groupby("loan_sequence_number").agg(
        max_delinquency=("current_loan_delinquency_status", "max"),
        zero_balance_code=("zero_balance_code", "last"),
    ).reset_index()

    default_codes = set(params["default_zero_balance_codes"])
    agg["default"] = (
        (agg["max_delinquency"] >= params["delinquency_threshold"])
        | (agg["zero_balance_code"].isin(default_codes))
    ).astype(int)
    return agg[["loan_sequence_number", "default"]]


# --------------------------------------------------------------------------- #
# 3. Jointure origination + cible
# --------------------------------------------------------------------------- #
def join_data(df_orig: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Joint l'origination et la cible sur loan_sequence_number."""
    df_orig = df_orig.copy()
    df_orig["loan_sequence_number"] = df_orig["loan_sequence_number"].str.strip()
    target["loan_sequence_number"] = target["loan_sequence_number"].str.strip()
    return df_orig.merge(target, on="loan_sequence_number", how="inner")


# --------------------------------------------------------------------------- #
# 4. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage des types, codes sentinelles, features métier dérivées."""
    df = df.drop(columns=["oltv", "postal_code"])

    numeric = ["credit_score", "dti", "ocltv", "original_upb", "original_interest_rate",
               "original_loan_term", "mip", "units", "number_of_borrowers",
               "program_indicator", "property_valuation_method", "mi_cancellation_indicator"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Codes sentinelles → NaN
    df.loc[df["dti"] == 999, "dti"] = np.nan
    df.loc[df["ocltv"] == 999, "ocltv"] = np.nan
    df.loc[df["credit_score"].isin([9999, 999]), "credit_score"] = np.nan

    # Dates YYYYMM → année / mois / durée
    fpd = pd.to_numeric(df["first_payment_date"], errors="coerce")
    mat = pd.to_numeric(df["maturity_date"], errors="coerce")
    df["origination_year"] = fpd // 100
    df["origination_month"] = fpd % 100
    df["loan_term_years"] = (mat // 100) - df["origination_year"]
    df = df.drop(columns=["first_payment_date", "maturity_date"])

    # Flags de risque + interactions
    df["is_subprime"] = (df["credit_score"] < 660).astype(int)
    df["is_high_ltv"] = (df["ocltv"] > 95).astype(int)
    df["is_high_dti"] = (df["dti"] > 43).astype(int)
    df["risk_count"] = df["is_subprime"] + df["is_high_ltv"] + df["is_high_dti"]
    df["fico_deficit"] = (850 - df["credit_score"]).clip(lower=0)
    df["fico_dti_interaction"] = df["fico_deficit"] * df["dti"] / 100
    df["fico_ltv_interaction"] = df["fico_deficit"] * df["ocltv"] / 100
    df["dti_ltv_interaction"] = df["dti"] * df["ocltv"] / 100

    r = df["original_interest_rate"] / 100 / 12
    n = df["original_loan_term"]
    factor = (1 + r) ** n
    df["monthly_payment"] = np.where(r > 0, df["original_upb"] * r * factor / (factor - 1),
                                     df["original_upb"] / n)
    mean_rate = df.groupby("origination_year")["original_interest_rate"].transform("mean")
    df["rate_spread"] = df["original_interest_rate"] - mean_rate
    df["payment_to_upb_ratio"] = df["monthly_payment"] / df["original_upb"]

    # Colonnes inutiles pour le modèle (leakage, IDs, texte, constantes, fort manquant)
    drop = ["loan_sequence_number", "seller_name", "servicer_name", "ppm_flag",
            "amortization_type", "super_conforming_flag", "pre_harp_loan_sequence_number",
            "harp_indicator", "interest_only_indicator"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df


# --------------------------------------------------------------------------- #
# 5. Split + encodage
# --------------------------------------------------------------------------- #
def _target_encode(train_col, test_col, target, smoothing):
    global_mean = target.mean()
    agg = target.groupby(train_col).agg(["count", "mean"])
    smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    mapping = smooth.to_dict()
    return (train_col.map(mapping).fillna(global_mean),
            test_col.map(mapping).fillna(global_mean), mapping)


def split_and_encode(df: pd.DataFrame, params: dict):
    """Split stratifié, target encoding (train only), one-hot, préprocesseur."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["default"])
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=params["random_state"],
    )

    # Imputation médiane (numériques)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    medians = X_train[numeric_cols].median().to_dict()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    # Target encoding géographique (calculé sur le train uniquement)
    maps = {}
    for col in params["high_cardinality"]:
        X_train[col], X_test[col], mapping = _target_encode(
            X_train[col].astype(str), X_test[col].astype(str), y_train, params["smoothing"]
        )
        maps[col] = mapping

    # One-hot basse cardinalité
    low = [c for c in params["low_cardinality"] if c in X_train.columns]
    X_train["__s"] = "tr"
    X_test["__s"] = "te"
    combined = pd.get_dummies(pd.concat([X_train, X_test]), columns=low, drop_first=True, dtype=int)
    X_train = combined[combined["__s"] == "tr"].drop(columns="__s")
    X_test = combined[combined["__s"] == "te"].drop(columns="__s")

    preprocessor = {
        "feature_columns": list(X_train.columns),
        "target_encoding_maps": maps,
        "target_global_mean": float(y_train.mean()),
        "feature_medians": X_train.median(numeric_only=True).to_dict(),
    }

    return (X_train, X_test,
            y_train.to_frame("default"), y_test.to_frame("default"),
            preprocessor)
