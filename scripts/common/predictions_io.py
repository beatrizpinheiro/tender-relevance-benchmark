"""Single output format for every model/baseline: `edital_id`, `split`,
`seed`, `prob` (+ `rotulo` for convenience). The evaluation script (out of
scope here) reads everyone through this same format."""

from pathlib import Path

import pandas as pd

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent.parent / "predictions"

REQUIRED_COLS = {"edital_id", "split", "seed", "prob"}


def save_predictions(df: pd.DataFrame, model_name: str) -> Path:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions: {missing}")
    out_dir = PREDICTIONS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"
    df.to_parquet(out_path, index=False)
    return out_path
