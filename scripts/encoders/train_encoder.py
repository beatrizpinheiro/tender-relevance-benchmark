"""Generic training of an encoder (full fine-tuning or LoRA), config-driven.

One script for the whole family of encoders with full fine-tuning or LoRA —
the protocol (optimizer, scheduler, checkpoint criterion, loss, bf16,
gradient checkpointing) is the same across the 9 priority-1 configs; only
the checkpoint, `max_length` and physical batch size vary per model. A run
= one (config, lr, seed) combination. The LR search and the 3 final seeds
are orchestrated externally (`run_priority1_encoders.py`), calling this
script one run at a time — separate processes free VRAM between runs,
important on a 6 GB GPU.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common.class_weight import positive_class_weight  # noqa: E402
from common.mlflow_logging import log_vram_peak, mlflow_run  # noqa: E402
from common.partial_auc import partial_auc_min_recall  # noqa: E402
from common.seeds import set_seed  # noqa: E402
from common.text_dataset import EditalDataset, PaddingCollator  # noqa: E402

DATA_PARQUET = Path(os.environ.get("EDITAL_DATA_PARQUET", BASE_DIR / "data" / "processed" / "editais_preprocessados.parquet"))
EFFECTIVE_BATCH = 16
MAX_EPOCHS = 5
PATIENCE_AVALIACOES = 4
WARMUP_FRACTION = 0.06
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
MIN_RECALL_CHECKPOINT = 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML in scripts/encoders/configs/")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--run_dir", required=True,
        help="This run's folder (unique per config+lr+seed): holds config.yaml, model/, metrics.json and predictions.parquet",
    )
    p.add_argument("--input_variant", default="com_objeto", choices=["com_objeto", "sem_objeto"])
    p.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--mlflow_experiment", default="editais_icde2027_encoders")
    p.add_argument("--force", action="store_true", help="Overwrites a run_dir that already has metrics.json")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits() -> dict:
    cols = ["edital_id", "split", "rotulo", "objeto_contratacao_limpo", "itens_texto"]
    df = pd.read_parquet(DATA_PARQUET, columns=cols)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def resolve_dtype(cfg: dict) -> torch.dtype:
    """bf16 by default (required on this GPU). fp32 override exists because
    some checkpoints crash in bf16 for reasons outside our code: Albertina
    100M (plain `deberta`, not `deberta-v2`) hits a transformers bug in its
    attention mask fill ("value cannot be converted to type c10::BFloat16
    without overflow") that mDeBERTa (`deberta-v2`) doesn't have. The model
    is small enough (100M params) that fp32 still fits on a 6 GB GPU."""
    return torch.float32 if cfg.get("dtype") == "fp32" else torch.bfloat16


def build_model(cfg: dict, device: torch.device, tokenizer) -> tuple:
    dtype = resolve_dtype(cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["checkpoint"],
        num_labels=2,
        torch_dtype=dtype,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    # Some published checkpoints add tokens to the tokenizer (e.g. digits) after
    # pretraining without resizing the saved embedding matrix to match — confirmed
    # for HelBERT-base: vocab_size=33000 in config, but added_tokens.json goes up
    # to id 33012, so every digit token in the input would index out of range.
    # Only grow, never shrink: a checkpoint with MORE embeddings than the
    # tokenizer's vocab is normal (reserved slots) and shrinking would discard
    # trained weights.
    n_embeddings_expandidas = 0
    current_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > current_size:
        n_embeddings_expandidas = len(tokenizer) - current_size
        print(f"Expanding embeddings from {current_size} to {len(tokenizer)} ({n_embeddings_expandidas} new, randomly initialized rows)")
        model.resize_token_embeddings(len(tokenizer))

    # use_reentrant=False: the reentrant mode (old default) breaks on DeBERTa-v2/v3
    # ("Trying to backward through the graph a second time") because of its
    # disentangled attention; non-reentrant works on every tested architecture.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = cfg.get("lora")
    if lora_cfg:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                target_modules=lora_cfg.get("target_modules", ["query", "key", "value", "dense"]),
            ),
        )
    return model.to(device), n_embeddings_expandidas


def model_input_keys(model) -> set:
    """Not every architecture accepts `token_type_ids` (ModernBERT, used by
    NorBERTo and mmBERT, doesn't have that parameter) — filters by what the
    model's forward actually accepts, instead of assuming the BERT-style format."""
    return set(inspect.signature(model.forward).parameters)


@torch.no_grad()
def predict_probs(model, loader, device, accepted_keys: set, dtype: torch.dtype) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels" and k in accepted_keys}
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        raise FileExistsError(
            f"{metrics_path} already exists — this run_dir already has a saved result. "
            "Use a different --run_dir or --force to overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()
    pos_weight = positive_class_weight(splits["train"])
    print(f"Positive class weight (train): {pos_weight:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["checkpoint"], trust_remote_code=cfg.get("trust_remote_code", False))
    max_length = cfg["max_length"]

    datasets = {
        s: EditalDataset(splits[s], tokenizer, max_length, args.input_variant) for s in ["train", "val", "test"]
    }
    collator = PaddingCollator(tokenizer)
    micro_batch = cfg["micro_batch_size"]
    accum_steps = max(EFFECTIVE_BATCH // micro_batch, 1)

    train_loader = DataLoader(datasets["train"], batch_size=micro_batch, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(datasets["val"], batch_size=micro_batch * 2, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(datasets["test"], batch_size=micro_batch * 2, shuffle=False, collate_fn=collator)

    dtype = resolve_dtype(cfg)
    model, n_embeddings_expandidas = build_model(cfg, device, tokenizer)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight], device=device, dtype=torch.float32))

    steps_per_epoch = len(train_loader) // accum_steps
    half_epoch_steps = max(steps_per_epoch // 2, 1)
    total_optimizer_steps = steps_per_epoch * args.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_FRACTION * total_optimizer_steps),
        num_training_steps=total_optimizer_steps,
    )

    config_mlflow = {
        "modelo": cfg["model_name"],
        "checkpoint": cfg["checkpoint"],
        "max_length": max_length,
        "input_variant": args.input_variant,
        "lr": args.lr,
        "seed": args.seed,
        "micro_batch_size": micro_batch,
        "effective_batch": EFFECTIVE_BATCH,
        "pos_weight": pos_weight,
        "n_embeddings_expandidas": n_embeddings_expandidas,
        "dtype": str(dtype),
    }
    # Written right at the start: even if the run crashes midway, what was
    # attempted in this run_dir stays on record.
    (run_dir / "config.yaml").write_text(yaml.dump(config_mlflow, allow_unicode=True))

    run_name = f"{cfg['model_name']}_lr{args.lr}_seed{args.seed}_{args.input_variant}"
    accepted_keys = model_input_keys(model)

    with mlflow_run(args.mlflow_experiment, run_name, config_mlflow):
        best_partial_auc = -1.0
        best_state = None
        avaliacoes_sem_melhora = 0
        global_micro_step = 0

        print(
            f"Plan: {len(train_loader)} micro-steps/epoch, {steps_per_epoch} opt-steps/epoch, "
            f"evaluating every {half_epoch_steps} opt-steps, up to {args.max_epochs} epochs."
        )

        for epoch in range(args.max_epochs):
            model.train()
            optimizer.zero_grad()
            loss_acumulada = 0.0
            pbar = tqdm(
                train_loader, desc=f"epoch {epoch}", unit="ed", mininterval=2.0, leave=True, file=sys.stdout,
            )
            for i, batch in enumerate(pbar):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items() if k in accepted_keys}
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(**batch).logits
                    loss = loss_fn(logits, labels) / accum_steps
                loss.backward()
                loss_acumulada += loss.item()
                global_micro_step += 1

                if global_micro_step % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    opt_step = global_micro_step // accum_steps
                    pbar.set_postfix(opt_step=f"{opt_step}/{steps_per_epoch}", loss=f"{loss_acumulada:.4f}", best_auc=f"{best_partial_auc:.4f}")
                    loss_acumulada = 0.0

                    if opt_step % half_epoch_steps == 0:
                        val_probs = predict_probs(model, val_loader, device, accepted_keys, dtype)
                        val_auc = partial_auc_min_recall(splits["val"]["rotulo"].values, val_probs, MIN_RECALL_CHECKPOINT)
                        pbar.write(f"epoch={epoch} opt_step={opt_step} auc_parcial_val={val_auc:.4f}")

                        if val_auc > best_partial_auc:
                            best_partial_auc = val_auc
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            avaliacoes_sem_melhora = 0
                        else:
                            avaliacoes_sem_melhora += 1

                        model.train()
                        if avaliacoes_sem_melhora >= PATIENCE_AVALIACOES:
                            break
            else:
                continue
            break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"Best partial validation AUC: {best_partial_auc:.4f}")

        t0 = time.time()
        test_probs = predict_probs(model, test_loader, device, accepted_keys, dtype)
        latencia_por_edital_ms = (time.time() - t0) * 1000 / len(splits["test"])
        val_probs_final = predict_probs(model, val_loader, device, accepted_keys, dtype)

        mlflow.log_metric("auc_parcial_val", best_partial_auc)
        mlflow.log_metric("latencia_inferencia_ms_por_edital", latencia_por_edital_ms)
        log_vram_peak()

    model_dir = run_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")

    registros = []
    for nome, split_df, probs in [("val", splits["val"], val_probs_final), ("test", splits["test"], test_probs)]:
        registros.append(
            pd.DataFrame(
                {
                    "edital_id": split_df["edital_id"].values,
                    "split": nome,
                    "seed": args.seed,
                    "prob": probs,
                    "rotulo": split_df["rotulo"].values,
                }
            )
        )
    out_df = pd.concat(registros, ignore_index=True)
    predictions_path = run_dir / "predictions.parquet"
    out_df.to_parquet(predictions_path, index=False)

    metrics_path.write_text(json.dumps({
        "modelo": cfg["model_name"],
        "lr": args.lr,
        "seed": args.seed,
        "input_variant": args.input_variant,
        "auc_parcial_val": best_partial_auc,
        "n_embeddings_expandidas": n_embeddings_expandidas,
        "latencia_inferencia_ms_por_edital": latencia_por_edital_ms,
        "vram_pico_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }, indent=2))
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
