"""TF-IDF (1-2gram words + 3-5gram chars) + logistic regression.

Hyperparameter search with Optuna (30 trials) by the checkpoint selection
criterion (partial validation AUC, recall >= 80%), then 5 seeds on the best
configuration. `class_weight="balanced"` is the project's default class
weight (n_rejected / n_approved on train). Vocabulary fit on train only.
"""

import sys
from pathlib import Path

import optuna
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common.partial_auc import partial_auc_min_recall  # noqa: E402
from common.predictions_io import save_predictions  # noqa: E402
from common.seeds import set_seed  # noqa: E402

DATA_PARQUET = BASE_DIR / "data" / "processed" / "editais_preprocessados.parquet"
TEXT_COL = "texto_com_objeto"
N_TRIALS = 30
SEEDS = [13, 42, 2027, 7, 99]


def load_splits() -> dict:
    df = pd.read_parquet(DATA_PARQUET, columns=["edital_id", "split", "rotulo", TEXT_COL])
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def vectorize(train_text, val_text, test_text):
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    X_train = hstack([word_vec.fit_transform(train_text), char_vec.fit_transform(train_text)]).tocsr()
    X_val = hstack([word_vec.transform(val_text), char_vec.transform(val_text)]).tocsr()
    X_test = hstack([word_vec.transform(test_text), char_vec.transform(test_text)]).tocsr()
    return X_train, X_val, X_test


def objective(trial, X_train, y_train, X_val, y_val) -> float:
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=13)
    clf.fit(X_train, y_train)
    prob_val = clf.predict_proba(X_val)[:, 1]
    return partial_auc_min_recall(y_val, prob_val, min_recall=0.8)


def main() -> None:
    splits = load_splits()
    X_train, X_val, X_test = vectorize(
        splits["train"][TEXT_COL], splits["val"][TEXT_COL], splits["test"][TEXT_COL]
    )
    y_train, y_val = splits["train"]["rotulo"].values, splits["val"]["rotulo"].values

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=13))
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val), n_trials=N_TRIALS)
    best_C = study.best_params["C"]
    print(f"Best C: {best_C:.4g} (partial val AUC={study.best_value:.4f})")

    X_by_split = {"train": X_train, "val": X_val, "test": X_test}
    registros = []
    for seed in SEEDS:
        set_seed(seed)
        clf = LogisticRegression(
            C=best_C, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed
        )
        clf.fit(X_train, y_train)
        for nome, split_df in splits.items():
            prob = clf.predict_proba(X_by_split[nome])[:, 1]
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
    out_path = save_predictions(previsoes, "tfidf_logreg")
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
