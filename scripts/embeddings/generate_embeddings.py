"""Generates each tender's embedding once per model (prefix and pooling set
in the config) and caches it. L2 normalization always applied. Runs once
per model; the heads (`train_frozen_head.py`) reuse this cache instead of
recomputing the embedding on every experiment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

DATA_PARQUET = BASE_DIR / "data" / "processed" / "editais_preprocessados.parquet"
EMBEDDINGS_DIR = BASE_DIR / "data" / "processed" / "embeddings"
DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def last_token_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    idx_ultimo = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.size(0)), idx_ultimo]


POOLINGS = {"mean": mean_pooling, "last_token": last_token_pooling}


@torch.no_grad()
def embed_texts(texts: list, tokenizer, model, device, dtype, max_length, pooling_fn, batch_size) -> np.ndarray:
    saida = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(**enc).last_hidden_state
        vec = pooling_fn(out.float(), enc["attention_mask"])
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        saida.append(vec.cpu().numpy())
    return np.concatenate(saida, axis=0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = DTYPE_MAP[cfg["precision"]]
    pooling_fn = POOLINGS[cfg["pooling"]]
    prefixo = cfg.get("prefixo", "")

    df = pd.read_parquet(DATA_PARQUET, columns=["edital_id", "split", "rotulo", "texto_com_objeto"])
    textos = (prefixo + df["texto_com_objeto"]).tolist()

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"], trust_remote_code=cfg.get("trust_remote_code", False))
    model = AutoModel.from_pretrained(
        cfg["checkpoint"], torch_dtype=dtype, trust_remote_code=cfg.get("trust_remote_code", False)
    ).to(device).eval()

    embeddings = embed_texts(
        textos, tokenizer, model, device, dtype, cfg["max_length"], pooling_fn, args.batch_size
    )

    out_dir = EMBEDDINGS_DIR / cfg["model_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "embeddings.npz",
        edital_id=df["edital_id"].values,
        split=df["split"].values,
        rotulo=df["rotulo"].values,
        embeddings=embeddings,
    )
    print(f"Embeddings saved to {out_dir / 'embeddings.npz'} (shape={embeddings.shape})")


if __name__ == "__main__":
    main()
