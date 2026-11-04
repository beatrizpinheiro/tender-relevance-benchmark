# Tender Relevance Benchmark

Experimental pipeline for scoring the relevance of public-tender (government
procurement) items to an IT hardware/software vendor.

A vendor currently screens tens of thousands of public tenders a year to
decide which ones are worth a human analyst's time. This repo benchmarks a
range of ML approaches to that relevance-scoring problem — from engineered
baselines to fine-tuned transformer encoders — under a **shared,
reproducible evaluation protocol**: the same frozen temporal train/val/test
split, the same checkpoint-selection criterion, and the same final
evaluation script for every model.

## Problem framing

Each tender (`edital`) is a request for a list of items (products/services)
with quantities and categories. The label is binary: did the vendor end up
approving/registering the tender as worth pursuing (`Aprovado` / `Cadastrado`)
or not (`Reprovado`)? The dataset is heavily imbalanced (~2.6% positive).

The target use case is a **review queue**, not a binary classifier: an
analyst works down a list ranked by predicted relevance, and the metric that
matters is how small that list can be while still catching almost every
tender worth pursuing. That shapes the whole evaluation protocol below.

## Pipeline overview

```
raw item-level CSV (private, not in this repo)
        │
        ▼
scripts/prepare_data.py            → splits/split_temporal.csv (frozen)
                                    → data/processed/editais_preprocessados.parquet
        │
        ▼
scripts/build_structured_features.py → data/processed/features_estruturadas.parquet
        │
        ├─ scripts/baselines/tfidf_logreg.py          (TF-IDF + logistic regression)
        ├─ scripts/baselines/lightgbm_estruturado.py  (LightGBM on structured features)
        ├─ scripts/baselines/mil_e5.py                (multiple-instance learning, gated attention)
        │
        ├─ scripts/embeddings/generate_embeddings.py  (frozen sentence embeddings)
        │  └─ scripts/embeddings/train_frozen_head.py (logreg / MLP head on top)
        │
        └─ scripts/encoders/train_encoder.py          (full fine-tuning / LoRA)
           orchestrated by scripts/encoders/run_priority1_encoders.py
        │
        ▼
predictions/<model_name>/predictions.parquet   (common format for every model)
```

Every model/baseline writes its predictions in the same schema —
`edital_id, split, seed, prob, rotulo` — so a single evaluation script (not
included in this benchmark's scope) can score every model identically.

### 1. Data preparation (`scripts/prepare_data.py`)

Reads the raw item-level CSV (one row per *(tender, item)* pair), then:

- Builds a one-row-per-tender table and assigns a **pure temporal split**:
  sorted by `(date, edital_id)` and cut by count into 70% train / 15% val /
  15% test — not by a fixed calendar date, so the proportions hold exactly
  regardless of how tender volume varies over time. Frozen to
  `splits/split_temporal.csv`.
- Cleans item text, merges items with identical normalized descriptions
  within a tender (summing quantities), sorts items by quantity descending,
  and caps at 60 items per tender.
- Assembles two text variants per tender: with the tender's stated object
  (`objeto_contratacao`) and without it — items alone. Token-level
  truncation (per-item cap, dynamic object budget) happens later, inside
  each model's tokenizer-specific code (`scripts/common/text_dataset.py`),
  since it depends on the tokenizer in use.

### 2. Structured features (`scripts/build_structured_features.py`)

Derives per-tender numeric features from quantity and category — total/log
quantity per category, item count, fraction of quantity in categories of
interest, keyword flags — used only by the LightGBM baseline, never by text
models.

### 3. Models

| Script | Approach |
| --- | --- |
| `baselines/tfidf_logreg.py` | TF-IDF (word + char n-grams) + logistic regression |
| `baselines/lightgbm_estruturado.py` | LightGBM on the structured features |
| `baselines/mil_e5.py` | Multiple-instance learning with gated attention over item embeddings |
| `embeddings/generate_embeddings.py` + `train_frozen_head.py` | Frozen sentence embeddings (e.g. multilingual-e5, Qwen3-Embedding, Serafim) with a logreg/MLP head |
| `encoders/train_encoder.py` + `run_priority1_encoders.py` | Full fine-tuning or LoRA on Portuguese/multilingual transformer encoders (BERTimbau, HelBERT, GovBERT-BR, Albertina, mDeBERTa-v3, NorBERTo, RoBERTaLexPT, mmBERT), config-driven via `encoders/configs/*.yaml` |

Every trainable model shares: the same frozen split, the same class weight
(`n_rejected / n_approved` on train, recomputed per split), the same
checkpoint-selection criterion (partial AUC restricted to the
high-recall region, recall ≥ 80% — see `scripts/common/partial_auc.py`), and
the same multi-seed protocol (cheap models: 5 seeds + 30-trial Optuna search;
transformer encoders: a short learning-rate search followed by 3 final
seeds).

## Evaluation philosophy

Because the deployment scenario is a ranked review queue rather than a
binary classifier, evaluation is framed as a ranking problem: each model's
predicted scores are compared against the tenders' binary ground-truth
label using ranking metrics — **NDCG**, **MAP (mean average precision)**,
and **Recall@K** at several cutoffs. Threshold-based metrics like accuracy
or F1 at 0.5 are not used, since they don't reflect how an analyst actually
consumes a ranked list.

## Repository layout

```
scripts/
  prepare_data.py                  split + text preprocessing
  build_structured_features.py     quantity/category features
  common/                          shared utilities (partial AUC, class weight,
                                   tokenization, predictions I/O, seeds, MLflow)
  baselines/                       TF-IDF, LightGBM, MIL
  embeddings/                      frozen embeddings + trainable heads
  encoders/                        fine-tuning / LoRA training + orchestration + configs
requirements.txt
```

`data/`, `splits/`, `runs/`, `predictions/`, `archive/` and the raw dataset
itself are generated or private artifacts and are **not tracked in this
repository** — see *Data* below.

## Data

The raw dataset — tender metadata, item descriptions, quantities,
categories and the vendor's approval decision — is private and is **not**
included in this repository. To run the pipeline on your own data, provide
a CSV with one row per *(tender, item)* with at least these columns, and
point `RAW_CSV` in `scripts/prepare_data.py` (and in
`build_structured_features.py` / `baselines/mil_e5.py`, which also read it
directly) at your file:

| Column | Meaning |
| --- | --- |
| `edital_id` | Tender identifier (repeated for every item in the tender) |
| `objeto_contratacao` | Free-text description of what the tender is procuring |
| `produto` | Free-text description of a single line item |
| `data_captacao_origem` | Date the tender was captured/published |
| `quantidade` | Quantity requested for that item |
| `categoria` | Item category (used by the structured LightGBM baseline only) |
| `aprovacao` | Vendor's outcome for the tender (mapped to the binary label) |

## Hardware

All training and inference in this benchmark runs on a single consumer GPU
with 6 GB of VRAM. That constrains the protocol throughout: bf16 mixed
precision, gradient checkpointing, 8-bit AdamW, gradient accumulation to an
effective batch size of 16, and LoRA (instead of full fine-tuning) for any
encoder above roughly 400M parameters.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

`torch` is installed separately because it needs the `cu126` index for this
machine's NVIDIA driver (the default PyPI index resolves to a CUDA build
that isn't compatible). `transformers` is pinned below the `5.x` line
because that line imports `torchvision` on every model load, which breaks
if the system's `torchvision` build doesn't match the installed `torch`.

## Running the pipeline

```bash
# 1. Split + preprocess text
python scripts/prepare_data.py

# 2. Structured features (for LightGBM)
python scripts/build_structured_features.py

# 3. Baselines
python scripts/baselines/tfidf_logreg.py
python scripts/baselines/lightgbm_estruturado.py
python scripts/baselines/mil_e5.py

# 4. Frozen embeddings + heads (per model config in scripts/embeddings/configs/)
python scripts/embeddings/generate_embeddings.py --config scripts/embeddings/configs/e5_large.yaml
python scripts/embeddings/train_frozen_head.py --model_name e5_large

# 5. Fine-tuned / LoRA encoders (orchestrates LR search + 3 final seeds per config)
python scripts/encoders/run_priority1_encoders.py
```

The encoder orchestrator is resumable: if interrupted, rerunning the same
command skips any run that already has a saved `metrics.json` and continues
from where it left off.

## Reproducibility

- The temporal split is frozen once and reused by every model
  (`splits/split_temporal.csv`).
- Multi-seed runs use a fixed seed list; transformer encoders run 3 final
  seeds, cheaper models run 5.
- Checkpoint selection always uses the partial-AUC-on-validation criterion
  above — never full AUC, never F1, never the full test set — so the final
  test-set evaluation is a single, decision-free pass per model.
