"""Generates the structured features per tender (quantity, category) used
by LightGBM and the heuristic, and joins them with the frozen temporal
split. Depends on `prepare_data.py` having already run (uses
`splits/split_temporal.csv`).
"""

from pathlib import Path

import pandas as pd

from common.structured_features import build_structured_features

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "base_score_junho_2025_ate_outubro_2026_sem_gemeos_divergentes.csv"
SPLIT_CSV = BASE_DIR / "splits" / "split_temporal.csv"
OUT_PARQUET = BASE_DIR / "data" / "processed" / "features_estruturadas.parquet"


def main() -> None:
    raw = pd.read_csv(RAW_CSV, usecols=["edital_id", "produto", "quantidade", "categoria"], dtype=str)
    feats = build_structured_features(raw)

    split = pd.read_csv(SPLIT_CSV, dtype={"edital_id": str})
    dados = split.merge(feats, on="edital_id", how="inner")
    if len(dados) != len(split):
        raise ValueError("Not every tender in the split has structured features")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(OUT_PARQUET, index=False)
    print(f"Structured features saved to {OUT_PARQUET} ({len(dados)} tenders, {dados.shape[1]} columns)")


if __name__ == "__main__":
    main()
