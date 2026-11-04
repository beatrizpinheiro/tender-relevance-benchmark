"""Builds a tender's tokenized input: object + items (or just the items in
the no-object variant). Uses the tokenizer's
`build_inputs_with_special_tokens` so the pair format (BERT, RoBERTa, ...)
comes out correct for any architecture, instead of assembling the special
tokens by hand.

Two truncation steps happen here (not in `prepare_data.py`), because they
depend on each model's tokenizer:

1. **Per item:** each item (`"qty x product"`, already separated by
   `prepare_data.ITEM_SEPARATOR`) is truncated to `ITEM_MAX_TOKENS` tokens
   before entering the sequence. A product with an abnormally long
   description gets truncated instead of alone consuming the budget of all
   the other items.
2. **Dynamic object:** the object's cap drops from `OBJETO_MAX_TOKENS` (128)
   down to the floor `OBJETO_MIN_TOKENS` (64) as the items (already
   truncated per item, but not yet by the overall budget) need more room.
   Whatever truncation is left after that always falls on the items, never
   on the object.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

OBJETO_MAX_TOKENS = 128
OBJETO_MIN_TOKENS = 64
ITEM_MAX_TOKENS = 48
ITEM_SEPARATOR = " ; "


def _tokenize_itens_por_item(tokenizer, itens: str) -> list:
    """Splits on the item delimiter and truncates each one to ITEM_MAX_TOKENS
    tokens before joining them back into a single sequence of ids."""
    partes = itens.split(ITEM_SEPARATOR) if itens else []
    if not partes:
        return []
    separador_ids = tokenizer.encode(ITEM_SEPARATOR.strip(), add_special_tokens=False)
    ids: list = []
    for i, parte in enumerate(partes):
        if i > 0:
            ids.extend(separador_ids)
        ids.extend(tokenizer.encode(parte, add_special_tokens=False, truncation=True, max_length=ITEM_MAX_TOKENS))
    return ids


def encode_edital(tokenizer, objeto: str, itens: str, max_length: int, input_variant: str = "com_objeto") -> dict:
    ids_itens_full = _tokenize_itens_por_item(tokenizer, itens)

    if input_variant == "sem_objeto":
        n_especiais = tokenizer.num_special_tokens_to_add(pair=False)
        budget_itens = max(max_length - n_especiais, 0)
        ids_itens = ids_itens_full[:budget_itens]
        input_ids = tokenizer.build_inputs_with_special_tokens(ids_itens)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids_itens)
    elif input_variant == "com_objeto":
        n_especiais = tokenizer.num_special_tokens_to_add(pair=True)
        budget_total = max_length - n_especiais
        # The object stays whole (up to 128) when the per-item-truncated
        # items fit the remaining budget; yields room down to the floor of
        # 64 when they don't.
        teto_objeto = min(OBJETO_MAX_TOKENS, max(OBJETO_MIN_TOKENS, budget_total - len(ids_itens_full)))
        ids_objeto = tokenizer.encode(objeto, add_special_tokens=False, truncation=True, max_length=teto_objeto)
        budget_itens = max(budget_total - len(ids_objeto), 0)
        ids_itens = ids_itens_full[:budget_itens]
        input_ids = tokenizer.build_inputs_with_special_tokens(ids_objeto, ids_itens)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids_objeto, ids_itens)
    else:
        raise ValueError(f"invalid input_variant: {input_variant}")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "token_type_ids": token_type_ids,
    }


class EditalDataset(Dataset):
    """`df` needs `objeto_contratacao_limpo`, `itens_texto` and `rotulo`
    (from `data/processed/editais_preprocessados.parquet`)."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, input_variant: str = "com_objeto"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_variant = input_variant

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = encode_edital(
            self.tokenizer,
            row["objeto_contratacao_limpo"],
            row["itens_texto"],
            self.max_length,
            self.input_variant,
        )
        enc["labels"] = int(row["rotulo"])
        return enc


class PaddingCollator:
    """Dynamic per-batch padding; uses the tokenizer to know the correct
    padding side and value for each architecture."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list) -> dict:
        labels = torch.tensor([ex.pop("labels") for ex in batch], dtype=torch.long)
        padded = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded
