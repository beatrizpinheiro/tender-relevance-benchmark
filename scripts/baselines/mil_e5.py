"""MIL with e5: same frozen temporal split, same checkpoint criterion
(partial validation AUC, recall >= 80%) and same class weight as the other
models in this repo.

Gated attention (Ilse et al., 2018) over the tender's items. Item =
[e5 embedding of the product, log(1+qty)/10, fraction of the tender's
quantity]. PRODUCT embeddings (not tender-level) via multilingual-e5-base —
mean pooling + L2, cached on disk per unique string.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
from common.class_weight import positive_class_weight  # noqa: E402
from common.partial_auc import partial_auc_min_recall  # noqa: E402
from common.predictions_io import save_predictions  # noqa: E402
from common.seeds import set_seed  # noqa: E402
from prepare_data import clean_text  # noqa: E402

RAW_CSV = BASE_DIR / "base_score_junho_2025_ate_outubro_2026_sem_gemeos_divergentes.csv"
SPLIT_CSV = BASE_DIR / "splits" / "split_temporal.csv"
CACHE_PATH = BASE_DIR / "data" / "processed" / "embeddings" / "mil_e5_base_itens" / "cache.npz"

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"
MAX_ITENS_MIL = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 128
MAX_EPOCAS = 20
PATIENCE_EPOCAS = 3
MIN_RECALL_CHECKPOINT = 0.8
SEEDS = [13, 42, 2027, 7, 99]


# --- data ------------------------------------------------------------------

def carregar_itens() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, usecols=["edital_id", "produto", "quantidade"], dtype=str)
    df["produto"] = df["produto"].map(clean_text)
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").fillna(1).clip(lower=0)
    df = df[df["produto"].str.len() > 0]
    # same product repeated in a tender (different lots) sums the quantity
    return df.groupby(["edital_id", "produto"], sort=False, as_index=False)["quantidade"].sum()


def carregar_splits(itens: pd.DataFrame) -> dict:
    split = pd.read_csv(SPLIT_CSV, dtype={"edital_id": str})
    return {s: split[split["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}


def numericas(itens: pd.DataFrame, editais: pd.Index) -> np.ndarray:
    lq = np.log1p(itens["quantidade"])
    g = itens.assign(lq=lq).groupby("edital_id")
    f = pd.DataFrame({
        "log_n_itens": np.log1p(g.size()),
        "log_qtd_total": np.log1p(g.quantidade.sum()),
        "log_qtd_max": g.lq.max(),
        "log_qtd_media": g.lq.mean(),
        "frac_maior_item": g.quantidade.max() / g.quantidade.sum().replace(0, np.nan),
    }).reindex(editais).fillna(0)
    return f.to_numpy(dtype=np.float32)


# --- item (product) embeddings, cached on disk -----------------------------

@torch.no_grad()
def gerar_embeddings_produto(produtos: list, device) -> dict:
    from transformers import AutoModel, AutoTokenizer

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=True)
        cache = dict(zip(d["produtos"], d["vetores"]))

    novos = [p for p in dict.fromkeys(produtos) if p not in cache]
    if novos:
        print(f"Embeddings: {len(novos)} new products (cache: {len(cache)})")
        tokenizer = AutoTokenizer.from_pretrained(MODELO_EMBEDDING)
        model = AutoModel.from_pretrained(MODELO_EMBEDDING, torch_dtype=torch.float16).to(device).eval()

        for i in range(0, len(novos), 256):
            bloco = novos[i : i + 256]
            enc = tokenizer([f"query: {t}" for t in bloco], truncation=True, max_length=64, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy().astype(np.float16)
            cache.update(zip(bloco, vec))

        todos = list(cache.keys())
        np.savez(CACHE_PATH, produtos=np.array(todos, dtype=object), vetores=np.stack([cache[p] for p in todos]))
        del model
        torch.cuda.empty_cache()
    return cache


# --- network: gated attention (Ilse et al., 2018) --------------------------

class RedeMIL(nn.Module):
    def __init__(self, dim_entrada: int):
        super().__init__()
        self.item = nn.Sequential(nn.Linear(dim_entrada, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU())
        self.att_v, self.att_u, self.att_w = nn.Linear(128, 64), nn.Linear(128, 64), nn.Linear(64, 1)
        self.saida = nn.Sequential(nn.Dropout(0.2), nn.Linear(128 + 5, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, mascara, num):
        h = self.item(x)
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mascara, -1e9), dim=1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.saida(torch.cat([z, num], 1)).squeeze(-1)


def preparar_tensores(itens: pd.DataFrame, editais: pd.Index, cache_emb: dict, dim: int) -> dict:
    it = itens.assign(lq=np.log1p(itens["quantidade"]) / 10)
    it["frac"] = it["quantidade"] / it.groupby("edital_id")["quantidade"].transform("sum").replace(0, 1)
    it = it.sort_values(["edital_id", "quantidade"], ascending=[True, False])
    it = it.groupby("edital_id", sort=False).head(MAX_ITENS_MIL)

    lin = editais.get_indexer(it["edital_id"])
    ok = lin >= 0
    it, lin = it[ok], lin[ok]

    E = np.stack([cache_emb[p] for p in it["produto"]]).astype(np.float16)
    ordem = np.argsort(lin, kind="stable")
    lin = lin[ordem]
    limites = np.searchsorted(lin, np.arange(len(editais) + 1))
    return dict(
        E=E[ordem], lq=it["lq"].to_numpy()[ordem].astype(np.float32),
        frac=it["frac"].to_numpy()[ordem].astype(np.float32), limites=limites,
        N=numericas(itens, editais),
    )


def montar_lote(dados: dict, idx: np.ndarray, device) -> tuple:
    E, lq, frac, limites = dados["E"], dados["lq"], dados["frac"], dados["limites"]
    ini, fim = limites[idx], limites[np.asarray(idx) + 1]
    comprimentos = fim - ini
    K = max(int(comprimentos.max()), 1)
    dim = E.shape[1] + 2
    X = np.zeros((len(idx), K, dim), dtype=np.float32)
    M = np.zeros((len(idx), K), dtype=bool)
    for j, (ini_j, n) in enumerate(zip(ini, comprimentos)):
        if n == 0:
            continue
        X[j, :n, :-2] = E[ini_j : ini_j + n]
        X[j, :n, -2] = lq[ini_j : ini_j + n]
        X[j, :n, -1] = frac[ini_j : ini_j + n]
        M[j, :n] = True
    return (
        torch.tensor(X, device=device), torch.tensor(M, device=device),
        torch.tensor(dados["N"][idx], device=device),
    )


@torch.no_grad()
def prever(rede, dados: dict, device) -> np.ndarray:
    rede.eval()
    idx = np.arange(len(dados["N"]))
    ps = []
    for b in np.array_split(idx, max(1, len(idx) // 512)):
        X, M, N = montar_lote(dados, b, device)
        ps.append(torch.sigmoid(rede(X, M, N)).cpu().numpy())
    return np.concatenate(ps)


def treinar_uma_seed(dados_tr, y_tr, dados_val, y_val, dim_entrada, pos_weight, seed, device):
    set_seed(seed)
    rede = RedeMIL(dim_entrada).to(device)
    opt = torch.optim.AdamW(rede.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    perda_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    rng = np.random.default_rng(seed)
    melhor_auc, sem_melhora, estado = -1.0, 0, None
    for ep in range(MAX_EPOCAS):
        rede.train()
        for b in np.array_split(rng.permutation(len(y_tr)), max(1, len(y_tr) // BATCH)):
            X, M, N = montar_lote(dados_tr, b, device)
            logit = rede(X, M, N)
            loss = perda_fn(logit, torch.tensor(y_tr[b], dtype=torch.float32, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        prob_val = prever(rede, dados_val, device)
        auc = partial_auc_min_recall(y_val, prob_val, MIN_RECALL_CHECKPOINT)
        print(f"   seed={seed} epoch={ep + 1} auc_parcial_val={auc:.4f}")
        if auc > melhor_auc:
            melhor_auc, sem_melhora = auc, 0
            estado = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= PATIENCE_EPOCAS:
                break

    rede.load_state_dict({k: v.to(device) for k, v in estado.items()})
    return rede, melhor_auc


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    itens = carregar_itens()
    splits = carregar_splits(itens)

    cache_emb = gerar_embeddings_produto(itens["produto"].unique().tolist(), device)
    dim_entrada = next(iter(cache_emb.values())).shape[0] + 2

    dados = {s: preparar_tensores(itens, pd.Index(splits[s]["edital_id"]), cache_emb, dim_entrada) for s in splits}
    sc = StandardScaler().fit(dados["train"]["N"])
    for s in dados:
        dados[s]["N"] = sc.transform(dados[s]["N"]).astype(np.float32)

    y = {s: splits[s]["rotulo"].to_numpy().astype(np.float32) for s in splits}
    pos_weight = positive_class_weight(splits["train"])
    print(f"Positive class weight (train): {pos_weight:.2f}")

    registros = []
    for seed in SEEDS:
        rede, _ = treinar_uma_seed(dados["train"], y["train"], dados["val"], y["val"], dim_entrada, pos_weight, seed, device)
        for nome in ["val", "test"]:
            prob = prever(rede, dados[nome], device)
            registros.append(pd.DataFrame({
                "edital_id": splits[nome]["edital_id"].values, "split": nome, "seed": seed,
                "prob": prob, "rotulo": splits[nome]["rotulo"].values,
            }))

    previsoes = pd.concat(registros, ignore_index=True)
    out_path = save_predictions(previsoes, "mil_e5")
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
