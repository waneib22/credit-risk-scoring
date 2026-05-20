"""Pipeline data_science : entraînement + évaluation."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_models, train_logreg, train_xgboost


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                train_logreg,
                inputs=["X_train", "y_train", "params:data_science"],
                outputs="logreg_model",
                name="train_logreg",
            ),
            node(
                train_xgboost,
                inputs=["X_train", "y_train", "params:data_science"],
                outputs="xgb_model",
                name="train_xgboost",
            ),
            node(
                evaluate_models,
                inputs=["logreg_model", "xgb_model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_models",
            ),
        ]
    )
