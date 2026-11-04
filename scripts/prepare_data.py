"""Pure temporal split + text preprocessing for the tender relevance score.

Reads the item-level base (`edital_id` repeated per product), generates the
frozen temporal split, and assembles, per tender, two text input variants:
with object and without object. Items with the same normalized description
in the same tender are merged (quantities summed) before sorting and
truncating to 60. Token-level truncation (48 per item, dynamic object
64-128, 512 total) is the responsibility of each model's tokenization
script (`text_dataset.py`), not this one, because it depends on each
architecture's tokenizer.
"""

import math
import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "base_score_junho_2025_ate_outubro_2026_sem_gemeos_divergentes.csv"
OUT_SPLIT_CSV = BASE_DIR / "splits" / "split_temporal.csv"
OUT_DATA_PARQUET = BASE_DIR / "data" / "processed" / "editais_preprocessados.parquet"

MAX_ITENS = 60
ITEM_SEPARATOR = " ; "
TRAIN_FRAC = 0.7
APROVACAO_POSITIVA = {"Aprovado", "Cadastrado"}

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_HIFEN_RE = re.compile(r"-")


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _chave_agrupamento_item(produto: str) -> str:
    """Normalizes the item description (lowercase, no accents, hyphen as
    space, collapsed whitespace) to use as a grouping key for identical
    items — the displayed text keeps the first spelling found."""
    sem_acento = unicodedata.normalize("NFKD", produto).encode("ascii", "ignore").decode()
    sem_hifen = _HIFEN_RE.sub(" ", sem_acento.lower())
    return _WHITESPACE_RE.sub(" ", sem_hifen).strip()


def load_raw(path: Path) -> pd.DataFrame:
    cols = [
        "edital_id",
        "objeto_contratacao",
        "produto",
        "data_captacao_origem",
        "quantidade",
        "aprovacao",
        "score_relevancia_ia",
    ]
    return pd.read_csv(path, usecols=cols, dtype=str)


def build_edital_table(df: pd.DataFrame) -> pd.DataFrame:
    editais = df.drop_duplicates("edital_id", keep="first").copy()

    n_date_variantes = df.groupby("edital_id")["data_captacao_origem"].nunique()
    if (n_date_variantes > 1).any():
        raise ValueError("data_captacao_origem varies within the same tender")
    n_obj_variantes = df.groupby("edital_id")["objeto_contratacao"].nunique()
    if (n_obj_variantes > 1).any():
        raise ValueError("objeto_contratacao varies within the same tender")
    n_aprov_variantes = df.groupby("edital_id")["aprovacao"].nunique()
    if (n_aprov_variantes > 1).any():
        raise ValueError("aprovacao varies within the same tender")

    editais["data_edital"] = pd.to_datetime(editais["data_captacao_origem"])
    editais["rotulo"] = editais["aprovacao"].isin(APROVACAO_POSITIVA).astype(int)
    editais["score_relevancia_ia"] = pd.to_numeric(editais["score_relevancia_ia"], errors="coerce")
    return editais[["edital_id", "data_edital", "rotulo", "score_relevancia_ia", "objeto_contratacao"]]


def assign_temporal_split(editais: pd.DataFrame) -> pd.DataFrame:
    """Sorts by date and cuts by count (70% / 15% / 15%), not by a fixed
    calendar cutoff — cutting by count guarantees the exact proportions even
    with uneven tender density over time."""
    ordenado = editais.sort_values(["data_edital", "edital_id"], kind="mergesort").reset_index(drop=True)
    n = len(ordenado)
    n_train = round(n * TRAIN_FRAC)
    n_val = (n - n_train) // 2

    split = pd.Series("test", index=ordenado.index)
    split.iloc[:n_train] = "train"
    split.iloc[n_train : n_train + n_val] = "val"
    ordenado["split"] = split
    return ordenado


def clean_items(df: pd.DataFrame) -> pd.DataFrame:
    itens = df[["edital_id", "produto", "quantidade"]].copy()
    itens["produto"] = itens["produto"].map(clean_text)
    itens["quantidade"] = pd.to_numeric(itens["quantidade"], errors="coerce")

    valido = itens["produto"].str.len().gt(0) & itens["quantidade"].notna()
    n_descartados = (~valido).sum()
    itens = itens[valido].copy()
    itens["quantidade"] = itens["quantidade"].map(lambda q: int(math.ceil(q)))
    print(f"Item rows dropped (no product or no quantity): {n_descartados}")
    return itens


def merge_itens_identicos(itens: pd.DataFrame) -> pd.DataFrame:
    """Items with the same normalized description in the same tender become
    one, with quantities summed. The text keeps the first spelling found,
    and the tie-break order in the next step preserves first appearance
    within the tender."""
    itens = itens.assign(
        _chave=itens["produto"].map(_chave_agrupamento_item),
        _ordem_original=range(len(itens)),
    )
    agregado = itens.groupby(["edital_id", "_chave"], sort=False).agg(
        produto=("produto", "first"),
        quantidade=("quantidade", "sum"),
        _ordem_original=("_ordem_original", "min"),
    )
    n_antes, n_depois = len(itens), len(agregado)
    print(f"Item rows after merging identical descriptions: {n_depois} (was {n_antes}, -{n_antes - n_depois})")
    return agregado.reset_index()[["edital_id", "produto", "quantidade", "_ordem_original"]]


def build_item_strings(itens: pd.DataFrame) -> pd.DataFrame:
    itens = merge_itens_identicos(itens)

    # Descending quantity; ties keep the tender's original order
    # (preserved in `_ordem_original`, assigned before the merge).
    ordenado = itens.sort_values(
        ["edital_id", "quantidade", "_ordem_original"], ascending=[True, False, True], kind="mergesort"
    )

    resumo = ordenado.groupby("edital_id").agg(
        n_itens_total=("produto", "size"),
        quantidade_total=("quantidade", "sum"),
    )

    cortado = ordenado.groupby("edital_id", sort=False).head(MAX_ITENS).copy()
    cortado["item_str"] = cortado["quantidade"].astype(str) + " x " + cortado["produto"]
    itens_texto = cortado.groupby("edital_id", sort=False)["item_str"].agg(ITEM_SEPARATOR.join)
    itens_texto.name = "itens_texto"
    n_itens_usados = cortado.groupby("edital_id", sort=False).size()
    n_itens_usados.name = "n_itens_usados"

    resultado = resumo.join(itens_texto).join(n_itens_usados)
    resultado["itens_truncados"] = resultado["n_itens_total"] > MAX_ITENS
    return resultado.reset_index()


def assemble_dataset(editais_split: pd.DataFrame, itens_por_edital: pd.DataFrame) -> pd.DataFrame:
    dados = editais_split.merge(itens_por_edital, on="edital_id", how="inner")
    n_sem_item = len(editais_split) - len(dados)
    print(f"Tenders excluded for having no valid item: {n_sem_item}")

    objeto_limpo = dados["objeto_contratacao"].map(clean_text)
    dados["objeto_contratacao_limpo"] = objeto_limpo.where(objeto_limpo.str.len().gt(0), "[sem objeto]")

    dados["texto_com_objeto"] = dados["objeto_contratacao_limpo"] + " | " + dados["itens_texto"]
    dados["texto_sem_objeto"] = dados["itens_texto"]

    colunas = [
        "edital_id",
        "split",
        "data_edital",
        "rotulo",
        "score_relevancia_ia",
        "objeto_contratacao_limpo",
        "itens_texto",
        "n_itens_total",
        "n_itens_usados",
        "itens_truncados",
        "quantidade_total",
        "texto_com_objeto",
        "texto_sem_objeto",
    ]
    return dados[colunas]


def print_split_report(dados: pd.DataFrame) -> None:
    print("\nPure temporal split summary:")
    for nome in ["train", "val", "test"]:
        parte = dados[dados["split"] == nome]
        print(
            f"  {nome:5s}: n={len(parte):6d}  approved={parte['rotulo'].sum():5d}  "
            f"rate={parte['rotulo'].mean()*100:.2f}%  "
            f"period={parte['data_edital'].min().date()} to {parte['data_edital'].max().date()}"
        )


def main() -> None:
    print(f"Reading {RAW_CSV.name} ...")
    raw = load_raw(RAW_CSV)

    editais = build_edital_table(raw)
    editais_split = assign_temporal_split(editais)

    OUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    editais_split[["edital_id", "split", "data_edital", "rotulo"]].to_csv(OUT_SPLIT_CSV, index=False)
    print(f"Frozen split saved to {OUT_SPLIT_CSV}")

    itens_limpos = clean_items(raw)
    itens_por_edital = build_item_strings(itens_limpos)

    dados = assemble_dataset(editais_split, itens_por_edital)

    OUT_DATA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(OUT_DATA_PARQUET, index=False)
    print(f"Preprocessed dataset saved to {OUT_DATA_PARQUET}")

    print_split_report(dados)


if __name__ == "__main__":
    main()
