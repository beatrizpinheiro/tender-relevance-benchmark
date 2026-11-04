"""Orchestrates the 9 priority-1 encoder configs: short LR search (1 seed)
-> 3 seeds on the best config -> merge predictions. Each run is a new
process (`train_encoder.py`) to free VRAM between them — important on a
6 GB GPU.

Each run writes to `runs/<model_name>/{busca_lr,seeds_finais}/<run_id>/`
(config, model, metrics, predictions — never overwrites another run). After
the 3 seeds, the predictions are just concatenated (not recomputed) into
`predictions/<model_name>/predictions.parquet`, the common format read by
the future evaluation script.

**Resumable:** if the orchestrator is interrupted (session dropped, process
killed), relaunching the same command skips runs that already have a saved
`metrics.json` and continues where it left off — nothing is redone.

Usage: python run_priority1_encoders.py [--only model1,model2] [--dry_run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_encoder.py"
CONFIGS_DIR = SCRIPT_DIR / "configs"
RUNS_DIR = BASE_DIR / "runs"
PREDICTIONS_DIR = BASE_DIR / "predictions"
SEARCH_SEED = 13
FINAL_SEEDS = [13, 42, 2027]

# Execution order for the priority-1 encoders, chosen to balance GPU budget.
ORDEM_CONFIGS = [
    "bertimbau_base",
    "helbert_base",
    "govbert_br",
    "albertina_100m",
    "norberto_base_512",
    "robertalexpt_base",
    "mmbert_small_512",
    "bertimbau_large",
    "mdeberta_v3_base",
    "norberto_base_2048",
    "mmbert_small_2048",
]


def run_training(config_path: Path, lr: float, seed: int, run_dir: Path, dry_run: bool) -> dict:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"Skipping (already has a saved result): {run_dir}")
        return json.loads(metrics_path.read_text())

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(config_path),
        "--lr", str(lr),
        "--seed", str(seed),
        "--run_dir", str(run_dir),
    ]
    print("Running:", " ".join(cmd))
    if dry_run:
        return {"lr": lr, "seed": seed, "auc_parcial_val": None}

    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
    return json.loads(metrics_path.read_text())


def lr_search(model_name: str, config_path: Path, lr_grid: list, dry_run: bool) -> float:
    best_lr, best_auc = None, -1.0
    for lr in lr_grid:
        run_dir = RUNS_DIR / model_name / "busca_lr" / f"lr{lr}_seed{SEARCH_SEED}"
        resultado = run_training(config_path, lr, SEARCH_SEED, run_dir, dry_run)
        auc = resultado["auc_parcial_val"]
        print(f"  lr={lr} -> auc_parcial_val={auc}")
        if auc is not None and auc > best_auc:
            best_lr, best_auc = lr, auc

    best_lr_path = RUNS_DIR / model_name / "best_lr.json"
    best_lr_path.parent.mkdir(parents=True, exist_ok=True)
    best_lr_path.write_text(json.dumps({"lr": best_lr, "auc_parcial_val": best_auc}))
    print(f"Best LR for {model_name}: {best_lr} (auc_parcial_val={best_auc})")
    return best_lr


def final_seeds_run(model_name: str, config_path: Path, best_lr: float, dry_run: bool) -> None:
    run_dirs = []
    for seed in FINAL_SEEDS:
        run_dir = RUNS_DIR / model_name / "seeds_finais" / f"seed{seed}"
        run_training(config_path, best_lr, seed, run_dir, dry_run)
        run_dirs.append(run_dir)

    if dry_run:
        return
    previsoes = pd.concat([pd.read_parquet(d / "predictions.parquet") for d in run_dirs], ignore_index=True)
    final_dir = PREDICTIONS_DIR / model_name
    final_dir.mkdir(parents=True, exist_ok=True)
    previsoes.to_parquet(final_dir / "predictions.parquet", index=False)
    print(f"Final predictions ({len(FINAL_SEEDS)} seeds) saved to {final_dir / 'predictions.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Comma-separated list of model_name values")
    parser.add_argument("--dry_run", action="store_true", help="Only print the commands, don't train")
    args = parser.parse_args()

    modelos = ORDEM_CONFIGS
    if args.only:
        modelos = [m for m in modelos if m in set(args.only.split(","))]

    failed = []
    for model_name in modelos:
        config_path = CONFIGS_DIR / f"{model_name}.yaml"
        cfg = yaml.safe_load(config_path.read_text())
        print(f"\n=== {model_name} ===")

        try:
            if cfg.get("lr_grid"):
                best_lr = lr_search(model_name, config_path, cfg["lr_grid"], args.dry_run)
            else:
                dep = cfg["reuse_lr_from"]
                dep_best_lr_path = RUNS_DIR / dep / "best_lr.json"
                if not dep_best_lr_path.exists():
                    raise FileNotFoundError(
                        f"{model_name} depends on {dep}'s LR, but {dep_best_lr_path} doesn't exist — run {dep} first."
                    )
                best_lr = json.loads(dep_best_lr_path.read_text())["lr"]
                print(f"Reusing LR from {dep}: {best_lr}")

            final_seeds_run(model_name, config_path, best_lr, args.dry_run)
        except Exception as e:
            # A crash in one model (OOM, bad checkpoint, etc.) must not stop the
            # rest of the queue — unattended runs need every remaining model to
            # still get a chance. Whatever completed for this model is already
            # saved under runs/ (never overwritten), so nothing is lost.
            print(f"FAILED: {model_name} raised {type(e).__name__}: {e}")
            failed.append(model_name)
            continue

    if failed:
        print(f"\n{len(failed)} model(s) failed and were skipped: {', '.join(failed)}")
        print("Whatever completed for them is saved under runs/<model_name>/ — rerun this same command to retry just the missing parts.")


if __name__ == "__main__":
    main()
