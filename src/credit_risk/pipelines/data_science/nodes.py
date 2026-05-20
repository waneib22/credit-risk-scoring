"""Nodes du pipeline data_science : entraînement et évaluation des modèles.

Reproduit la logique du notebook 04 : régression logistique (baseline
interprétable) et XGBoost (challenger), évaluées sur le test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_logreg(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict) -> dict:
    """Régression logistique avec scaling et class_weight balanced."""
    y = y_train["default"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=params["logreg_max_iter"],
        class_weight="balanced",
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    return {"model": model, "scaler": scaler}


def train_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """XGBoost avec split interne train/val pour l'early stopping."""
    import xgboost as xgb

    y = y_train["default"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=params["val_size"], stratify=y,
        random_state=params["random_state"],
    )
    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=1,
        eval_metric="aucpr",
        early_stopping_rounds=params["early_stopping_rounds"],
        tree_method="hist",
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_models(logreg_bundle: dict, xgb_model, X_test: pd.DataFrame,
                    y_test: pd.DataFrame) -> dict:
    """Calcule AUC et AP des deux modèles sur le test set."""
    y = y_test["default"]

    proba_lr = logreg_bundle["model"].predict_proba(
        logreg_bundle["scaler"].transform(X_test)
    )[:, 1]
    proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    metrics = {
        "logreg_auc": float(roc_auc_score(y, proba_lr)),
        "logreg_ap": float(average_precision_score(y, proba_lr)),
        "xgb_auc": float(roc_auc_score(y, proba_xgb)),
        "xgb_ap": float(average_precision_score(y, proba_xgb)),
        "n_test": int(len(y)),
        "default_rate": float(y.mean()),
    }
    return metrics
