"""Positive class weight: n_rejected / n_approved, computed on train only."""

import pandas as pd


def positive_class_weight(train_df: pd.DataFrame, label_col: str = "rotulo") -> float:
    n_pos = int(train_df[label_col].sum())
    n_neg = len(train_df) - n_pos
    if n_pos == 0:
        raise ValueError("Train set has no positive examples")
    return n_neg / n_pos
