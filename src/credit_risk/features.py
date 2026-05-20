"""Reconstruction du vecteur de features à partir d'un profil brut.

Rejoue le pipeline du notebook 03 (feature engineering + target encoding + one-hot)
pour transformer un profil emprunteur saisi par l'utilisateur en la matrice
attendue par le modèle. Utilisé par l'app Streamlit.
"""
from __future__ import annotations

import pandas as pd

# Taux moyen approximatif des millésimes 2017-2018 (pour rate_spread à l'inférence)
_MEAN_RATE_2017_2018 = 4.0


def build_feature_vector(inp: dict, pre: dict) -> pd.DataFrame:
    """Construit la ligne de features à partir d'un profil brut.

    Args:
        inp: profil emprunteur (credit_score, dti, ocltv, original_upb,
             original_interest_rate, original_loan_term, number_of_borrowers,
             property_state, channel, occupancy_status, loan_purpose,
             property_type, first_time_homebuyer_flag).
        pre: préprocesseur sauvegardé par le notebook 03 (feature_columns,
             target_encoding_maps, target_global_mean, feature_medians).

    Returns:
        DataFrame d'une ligne, colonnes dans l'ordre attendu par le modèle.
    """
    cols = pre["feature_columns"]
    row = {c: 0.0 for c in cols}

    # Défauts = médianes du train (variables non saisies : msa, mip, units…)
    for c, v in pre["feature_medians"].items():
        if c in row:
            row[c] = v

    fico = inp["credit_score"]
    dti = inp["dti"]
    ltv = inp["ocltv"]
    upb = inp["original_upb"]
    rate = inp["original_interest_rate"]
    term = inp["original_loan_term"]

    # Variables brutes
    row["credit_score"] = fico
    row["dti"] = dti
    row["ocltv"] = ltv
    row["original_upb"] = upb
    row["original_interest_rate"] = rate
    row["original_loan_term"] = term
    row["number_of_borrowers"] = inp["number_of_borrowers"]
    row["origination_year"] = 2018
    row["origination_month"] = 6
    row["loan_term_years"] = round(term / 12)

    # Features dérivées (identiques au notebook 03)
    row["is_subprime"] = int(fico < 660)
    row["is_high_ltv"] = int(ltv > 95)
    row["is_high_dti"] = int(dti > 43)
    row["risk_count"] = row["is_subprime"] + row["is_high_ltv"] + row["is_high_dti"]

    fico_deficit = max(850 - fico, 0)
    row["fico_deficit"] = fico_deficit
    row["fico_dti_interaction"] = fico_deficit * dti / 100
    row["fico_ltv_interaction"] = fico_deficit * ltv / 100
    row["dti_ltv_interaction"] = dti * ltv / 100

    r = rate / 100 / 12
    factor = (1 + r) ** term
    monthly = upb * r * factor / (factor - 1) if r > 0 else upb / term
    row["monthly_payment"] = monthly
    row["payment_to_upb_ratio"] = monthly / upb if upb else 0.0
    row["rate_spread"] = rate - _MEAN_RATE_2017_2018

    # Target encoding géographique (modalité inconnue → moyenne globale)
    state_map = pre["target_encoding_maps"].get("property_state", {})
    if "property_state" in row:
        row["property_state"] = state_map.get(inp["property_state"], pre["target_global_mean"])

    # One-hot (drop_first) : on met à 1 la colonne correspondante si elle existe
    for cat, val in [
        ("channel", inp["channel"]),
        ("occupancy_status", inp["occupancy_status"]),
        ("property_type", inp["property_type"]),
        ("loan_purpose", inp["loan_purpose"]),
        ("first_time_homebuyer_flag", inp["first_time_homebuyer_flag"]),
    ]:
        colname = f"{cat}_{val}"
        if colname in row:
            row[colname] = 1

    return pd.DataFrame([row])[cols]
