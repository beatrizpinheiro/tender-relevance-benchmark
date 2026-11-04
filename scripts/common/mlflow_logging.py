"""Thin wrapper around MLflow: YAML config, commit hash, training time,
peak VRAM, inference latency and validation metrics — enough to compare
runs without reopening every `metrics.json`."""

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import mlflow
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Absolute path, not relative to cwd: scripts are invoked from different
# directories (the orchestrator runs with cwd=scripts/encoders/), and a
# relative tracking URI would fragment runs across separate databases.
DEFAULT_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"


def _git_commit_hash(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "sem-git"


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, config: dict, tracking_uri: str = None):
    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    config_path = Path("/tmp") / f"config_{run_name}.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(config))
        mlflow.log_param("git_commit", _git_commit_hash(Path(__file__).resolve().parent.parent.parent))
        mlflow.log_artifact(str(config_path))

        start = time.time()
        yield run
        mlflow.log_metric("tempo_treino_segundos", time.time() - start)


def log_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            mlflow.log_metric("vram_pico_mb", torch.cuda.max_memory_allocated() / 1024**2)
    except ImportError:
        pass


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat
