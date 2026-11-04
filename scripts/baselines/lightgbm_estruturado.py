"""LightGBM on the structured features (quantity, category).

Hyperparameter search with Optuna (30 trials) by the checkpoint selection
criterion (partial validation AUC, recall >= 80%), then 5 seeds on the best
configuration. `scale_pos_weight` is the project's default class weight
(n_rejected / n_approved on train).
"""

import sys
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common.class_weight import positive_class_weight  # noqa: E402
from common.partial_auc import partial_auc_min_recall  # noqa: E402
from common.predictions_io import save_predictions  # noqa: E402

FEATURES_PARQUET = BASE_DIR / "data" / "processed" / "features_estruturadas.parquet"
N_TRIALS = 30
SEEDS = [13, 42, 2027, 7, 99]


NAO_FEATURE_COLS = ("edital_id", "split", "rotulo", "data_edital")


def load_splits():
    df = pd.read_parquet(FEATURES_PARQUET)
    feature_cols = [c for c in df.columns if c not in NAO_FEATURE_COLS]
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    return splits, feature_cols


def objective(trial, X_train, y_train, X_val, y_val, pos_weight) -> float:
    params = dict(
        objective="binary",
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight=pos_weight,
        n_estimators=500,
        verbosity=-1,
        random_state=13,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    prob_val = model.predict_proba(X_val)[:, 1]
    return partial_auc_min_recall(y_val, prob_val, min_recall=0.8)


def main() -> None:
    splits, feature_cols = load_splits()
    X_train, y_train = splits["train"][feature_cols], splits["train"]["rotulo"]
    X_val, y_val = splits["val"][feature_cols], splits["val"]["rotulo"]
    pos_weight = positive_class_weight(splits["train"])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=13))
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, pos_weight), n_trials=N_TRIALS)
    print(f"Best hyperparameters: {study.best_params} (partial val AUC={study.best_value:.4f})")

    registros = []
    for seed in SEEDS:
        params = dict(
            study.best_params,
            objective="binary",
            scale_pos_weight=pos_weight,
            n_estimators=500,
            verbosity=-1,
            random_state=seed,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        for nome, split_df in splits.items():
            prob = model.predict_proba(split_df[feature_cols])[:, 1]
            registros.append(
                pd.DataFrame(
                    {
                        "edital_id": split_df["edital_id"].values,
                        "split": nome,
                        "seed": seed,
                        "prob": prob,
                        "rotulo": split_df["rotulo"].values,
                    }
                )
            )

    previsoes = pd.concat(registros, ignore_index=True)
    out_path = save_predictions(previsoes, "lightgbm_estruturado")
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
