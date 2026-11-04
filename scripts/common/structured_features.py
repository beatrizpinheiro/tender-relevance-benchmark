"""Structured features per tender, from the item-level base (raw CSV):
total quantity and log quantity per category, number of items, quantity
fraction in categories of interest, and keyword flags.

Used by the structured LightGBM model and the category heuristic. Not used
by text models: `categoria_item` only enters here.
"""

import unicodedata

import numpy as np
import pandas as pd

# Categories that don't count as "relevant IT hardware" for the vendor:
# miscellaneous, service and software licensing.
CATEGORIAS_FORA_DE_INTERESSE = {"Outro", "Serviço", "Licença Microsoft"}
PALAVRAS_CHAVE = ["locacao", "aquisicao", "servico"]


def _sem_acento(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def build_structured_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[["edital_id", "produto", "quantidade", "categoria"]].copy()
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(0)
    produto_norm = df["produto"].fillna("").map(_sem_acento).str.lower()

    quantidade_total = df.groupby("edital_id")["quantidade"].sum().rename("quantidade_total")
    n_itens = df.groupby("edital_id").size().rename("n_itens")

    de_interesse = ~df["categoria"].isin(CATEGORIAS_FORA_DE_INTERESSE)
    qtd_interesse = df[de_interesse].groupby("edital_id")["quantidade"].sum()
    frac_interesse = (qtd_interesse.reindex(quantidade_total.index, fill_value=0.0) / quantidade_total.replace(0, np.nan)).fillna(0.0)
    frac_interesse = frac_interesse.rename("frac_qtd_categorias_interesse")

    qtd_por_categoria = df.pivot_table(
        index="edital_id", columns="categoria", values="quantidade", aggfunc="sum", fill_value=0.0
    )
    log_qtd_por_categoria = np.log1p(qtd_por_categoria)
    log_qtd_por_categoria.columns = [
        f"log_qtd_{c.lower().replace(' ', '_')}" for c in log_qtd_por_categoria.columns
    ]

    for kw in PALAVRAS_CHAVE:
        df[f"_kw_{kw}"] = produto_norm.str.contains(kw, regex=False)
    flags = df.groupby("edital_id")[[f"_kw_{kw}" for kw in PALAVRAS_CHAVE]].any().astype(int)
    flags.columns = [f"flag_{kw}" for kw in PALAVRAS_CHAVE]

    out = pd.concat([quantidade_total, n_itens, frac_interesse, log_qtd_por_categoria, flags], axis=1)
    out["log_quantidade_total"] = np.log1p(out["quantidade_total"])
    return out.reset_index()
