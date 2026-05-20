"""Pipeline data_processing : raw → features → split."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_target,
    engineer_features,
    join_data,
    load_raw_data,
    split_and_encode,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_raw_data,
                inputs="params:data_processing",
                outputs=["df_orig", "df_perf"],
                name="load_raw_data",
            ),
            node(
                build_target,
                inputs=["df_perf", "params:data_processing"],
                outputs="target",
                name="build_target",
            ),
            node(
                join_data,
                inputs=["df_orig", "target"],
                outputs="loans_joined",
                name="join_data",
            ),
            node(
                engineer_features,
                inputs="loans_joined",
                outputs="loans_features",
                name="engineer_features",
            ),
            node(
                split_and_encode,
                inputs=["loans_features", "params:data_processing"],
                outputs=["X_train", "X_test", "y_train", "y_test", "preprocessor"],
                name="split_and_encode",
            ),
        ]
    )
