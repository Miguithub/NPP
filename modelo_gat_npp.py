# %% [markdown]
# # GAT crypto
#
# Prediccion conjunta, a 10 minutos, del simplex de cuotas transaccionales de
# criptoactivos. Cada activo es un nodo. El objetivo es el vector completo
# `p(t+1)` y la salida se normaliza con softmax, de modo que siempre pertenece
# al simplex.
#
# Este notebook replica para **cada** `*_p` (excepto `YS_p`) las 20 variables
# del XGBoost de factores de BTC: seis rezagos y los 14 factores que sobrevivieron
# seleccion temporal, importancia por permutacion y ablacion por familias.
#
# Contratos metodologicos:
#
# - horizonte fijo `t -> t+1`; ninguna feature usa `shift(-k)`;
# - cortes cronologicos con purga; `shuffle=False` en todos los loaders;
# - escalado ajustado exclusivamente en train;
# - aristas calculadas con CLR historico hasta `t`, nunca con `p(t+1)`;
# - test final intacto hasta terminar seleccion/early stopping;
# - `YS_p` se excluye de los nodos, pero se verifica que las cuotas restantes
#   sumen uno;
# - las cabezas `gp` y `eta` son latentes. Con datos transaccionales solamente
#   es identificable su cociente energetico `E=gp/eta`, no una interpretacion
#   causal separada de ambos componentes.

# %%
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from IPython.display import display

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "PyTorch no esta instalado. En Google Colab selecciona un runtime con GPU."
    ) from exc

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler


# %% [markdown]
# ## 1. Configuracion reproducible
#
# Las rutas corresponden a los cuatro Parquet creados por
# `EXT NPP Crypto 01/2026`. Si Drive aparece bajo otra ruta, solo hay que
# modificar este bloque.

# %%
@dataclass(frozen=True)
class Config:
    dfyp_path: str = "/content/drive/MyDrive/0626dfyp.parquet"
    dftu_path: str = "/content/drive/MyDrive/0626dftu.parquet"
    price_path: str = "/content/drive/MyDrive/0626p.parquet"
    volume_path: str = "/content/drive/MyDrive/0626v.parquet"
    results_dir: str = "/content/drive/MyDrive/Neural/NPP/Cripto/GAT_crypto_results"
    cache_dir: str = "/content/gat_crypto_cache"

    start_timestamp: str = "2026-01-01 00:00:00+00:00"
    frequency: str = "10min"
    horizon: int = 1
    validation_size: int = 2016
    test_size: int = 2016
    purge: int = 1
    sample_stride: int = 1

    edge_window: int = 1008       # siete dias de barras de 10 minutos
    edge_refresh: int = 1008      # grafo causal por semana
    edge_projection_dim: int = 24
    edge_top_k: int = 8
    edge_alpha_clr: float = 0.75
    edge_gamma_npp: float = 0.25
    min_active_share: float = 1e-12

    hidden_dim: int = 32
    attention_heads: int = 4
    dropout: float = 0.10
    batch_size: int = 8
    epochs: int = 20
    patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    lambda_npp: float = 0.05
    lambda_energy_ce: float = 0.25
    lambda_eta_anchor: float = 1e-4
    lambda_energy_smooth: float = 1e-4

    eps: float = 1e-12
    seed: int = 42
    rebuild_features: bool = True


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", DEVICE)


# %% [markdown]
# ## 2. Carga y auditoria del simplex
#
# El notebook de extraccion guardo los Parquet con `index=False`. Por eso se
# reconstruye el indice UTC a frecuencia exacta de 10 minutos. No se reordena
# ninguna fila y no se hace backward-fill.

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Fuera de Colab: se usan las rutas locales indicadas en CFG.")


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


dfyp = read_table(CFG.dfyp_path)
dftu = read_table(CFG.dftu_path)
dfp = read_table(CFG.price_path)
dfv = read_table(CFG.volume_path)

if "YS" in dfyp.columns and "YS_p" not in dfyp.columns:
    dfyp = dfyp.rename(
        columns=lambda c: c[:-2] + "_p" if c.endswith("_y") else "YS_p" if c == "YS" else c
    )

lengths = {"dfyp": len(dfyp), "dftu": len(dftu), "dfp": len(dfp), "dfv": len(dfv)}
if len(set(lengths.values())) != 1:
    raise ValueError(f"Los cuatro archivos no tienen la misma longitud: {lengths}")

timestamps = pd.date_range(
    start=pd.Timestamp(CFG.start_timestamp), periods=len(dfyp), freq=CFG.frequency
)
for frame in (dfyp, dftu, dfp, dfv):
    frame.index = timestamps

p_cols = [c for c in dfyp.columns if c.endswith("_p") and c != "YS_p"]
if not p_cols:
    raise ValueError("No se encontraron columnas *_p en dfyp.")

assets = [c[:-2] for c in p_cols]
price_cols = [f"{asset}USDT" for asset in assets]
y_cols = [f"{asset}_y" for asset in assets]
volume_cols = [
    f"{asset}_V" if f"{asset}_V" in dfv.columns else f"{asset}_v"
    for asset in assets
]

missing = {
    "precio": [c for c in price_cols if c not in dfp.columns],
    "y": [c for c in y_cols if c not in dftu.columns],
    "volumen": [c for c in volume_cols if c not in dfv.columns],
}
missing = {k: v for k, v in missing.items() if v}
if missing:
    preview = {k: v[:20] for k, v in missing.items()}
    raise ValueError(
        "No se eliminan nodos silenciosamente. Faltan columnas para algunos *_p: "
        f"{preview}"
    )

P = dfyp[p_cols].to_numpy(dtype=np.float32, copy=True)
Y = dftu[y_cols].to_numpy(dtype=np.float32, copy=True)
PRICE = dfp[price_cols].to_numpy(dtype=np.float32, copy=True)
VOLUME = dfv[volume_cols].to_numpy(dtype=np.float32, copy=True)

if np.nanmin(P) < -1e-8:
    raise ValueError("dfyp contiene probabilidades negativas.")

# En la construccion original, NaN de un par ausente significa masa cero.
P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
P = np.clip(P, 0.0, None)
row_sums = P.sum(axis=1, keepdims=True, dtype=np.float64)
if np.any(row_sums <= 0):
    bad = np.flatnonzero(row_sums.ravel() <= 0)[:20]
    raise ValueError(f"Hay filas sin masa transaccional: {bad.tolist()}")
P = (P / row_sums).astype(np.float32)

simplex_error = np.abs(P.sum(axis=1, dtype=np.float64) - 1.0)
if simplex_error.max() > 1e-5:
    raise AssertionError("La normalizacion del simplex fallo.")
if "YS_p" in dfyp.columns:
    ys_p = pd.to_numeric(dfyp["YS_p"], errors="coerce").to_numpy()
    if np.nanmax(np.abs(ys_p - 1.0)) > 1e-5:
        raise ValueError("YS_p no es identicamente uno; revisar la construccion de dfyp.")

print(f"Filas: {len(P):,}; nodos: {len(assets):,}")
print(f"Error maximo de cierre del simplex: {simplex_error.max():.3e}")
print("Rango temporal:", timestamps[0], "->", timestamps[-1])


# %% [markdown]
# ## 3. Las 20 variables del XGBoost, desarrolladas para todos los nodos
#
# Variables finales del notebook `Vars NPP Crypto 01/2026`:
#
# - `p_lag0` ... `p_lag5`;
# - sorpresa de Shannon;
# - RLIQ a 20m, 30m, 1h, 2h y 4h;
# - skew de la cuota a 1 y 2 meses;
# - indice `n = EMA_precio_1m / sorpresa`;
# - MACD de precio 1 semana vs 2 semanas;
# - estabilidad bajista a 2 y 3 meses;
# - RSIp a 3 dias;
# - curtosis de la cuota a 3 meses.
#
# A diferencia de las celdas exploratorias antiguas, skew y curtosis se calculan
# realmente sobre `p`, igual que en la celda final de preparacion del XGBoost.
# El archivo de features es un memmap float32 para no exigir que ~4 GB residan
# simultaneamente en RAM.

# %%
WINDOWS = {
    "20": 2,
    "30": 3,
    "1h": 6,
    "2h": 12,
    "4h": 24,
    "3d": 432,
    "1s": 1008,
    "2s": 2016,
    "1m": 4032,
    "2m": 8064,
    "3m": 12128,
}

FEATURE_NAMES = [f"p_lag{lag}" for lag in range(6)] + [
    "shannon",
    "RLIQ_20",
    "RLIQ_30",
    "RLIQ_1h",
    "RLIQ_2h",
    "RLIQ_4h",
    "SKEW_p_1m",
    "n_model",
    "MACD_1s_2s",
    "RSTAB_3m",
    "SKEW_p_2m",
    "RSIp_3d",
    "KURT_p_3m",
    "RSTAB_2m",
]

assert len(FEATURE_NAMES) == 20
WARMUP = max(WINDOWS["3m"], WINDOWS["2m"], WINDOWS["1m"])

cache_dir = Path(CFG.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)
feature_path = cache_dir / "gat_crypto_features_float32.dat"
feature_meta_path = cache_dir / "gat_crypto_features_meta.json"
feature_shape = (len(P), len(assets), len(FEATURE_NAMES))


def causal_ffill(values: np.ndarray) -> pd.DataFrame:
    # ffill solo usa observaciones <= t. No se aplica bfill a huecos iniciales.
    return pd.DataFrame(values, index=timestamps, columns=assets).ffill()


def build_feature_memmap() -> np.memmap:
    expected_meta = {
        "shape": list(feature_shape),
        "features": FEATURE_NAMES,
        "dtype": "float32",
    }
    if (
        feature_path.exists()
        and feature_meta_path.exists()
        and not CFG.rebuild_features
        and json.loads(feature_meta_path.read_text(encoding="utf-8")) == expected_meta
    ):
        print("Se reutiliza cache:", feature_path)
        return np.memmap(feature_path, mode="r+", dtype=np.float32, shape=feature_shape)

    print(f"Construyendo memmap de {np.prod(feature_shape) * 4 / 2**30:.2f} GiB")
    mm = np.memmap(feature_path, mode="w+", dtype=np.float32, shape=feature_shape)
    p_df = pd.DataFrame(P, index=timestamps, columns=assets)
    y_df = pd.DataFrame(np.nan_to_num(Y, nan=0.0), index=timestamps, columns=assets)
    price_df = causal_ffill(PRICE)

    slot = 0

    def write_feature(name: str, values) -> None:
        nonlocal slot
        if FEATURE_NAMES[slot] != name:
            raise AssertionError(f"Orden de features incoherente: {name} != {FEATURE_NAMES[slot]}")
        if isinstance(values, pd.DataFrame):
            arr = values.to_numpy(dtype=np.float32, copy=False)
        else:
            arr = np.asarray(values, dtype=np.float32)
        if arr.shape != feature_shape[:2]:
            raise ValueError(f"Shape invalido para {name}: {arr.shape}")
        mm[:, :, slot] = arr
        mm.flush()
        slot += 1
        del arr
        gc.collect()

    for lag in range(6):
        write_feature(f"p_lag{lag}", p_df.shift(lag))

    shannon = -np.log(np.clip(P, CFG.eps, 1.0)).astype(np.float32)
    write_feature("shannon", shannon)

    for label in ["20", "30", "1h", "2h", "4h"]:
        n = WINDOWS[label]
        p_mean = p_df.rolling(n, min_periods=n).mean()
        y_mean = y_df.clip(lower=0).rolling(n, min_periods=n).mean()
        y_std = y_df.clip(lower=0).rolling(n, min_periods=n).std()
        rliq = p_mean / (1.0 + y_std / (y_mean + CFG.eps))
        write_feature(f"RLIQ_{label}", rliq)
        del p_mean, y_mean, y_std, rliq
        gc.collect()

    write_feature(
        "SKEW_p_1m",
        p_df.rolling(WINDOWS["1m"], min_periods=WINDOWS["1m"]).skew(),
    )

    ema_1m = price_df.ewm(span=WINDOWS["1m"], adjust=False).mean()
    n_model = ema_1m.to_numpy(dtype=np.float32) / np.clip(shannon, CFG.eps, None)
    write_feature("n_model", n_model)
    del ema_1m, n_model, shannon
    gc.collect()

    ema_1s = price_df.ewm(span=WINDOWS["1s"], adjust=False).mean()
    ema_2s = price_df.ewm(span=WINDOWS["2s"], adjust=False).mean()
    write_feature("MACD_1s_2s", ema_1s - ema_2s)
    del ema_1s, ema_2s
    gc.collect()

    log_price = np.log(price_df.where(price_df > 0))
    downside = log_price.diff().clip(upper=0)

    def rstab(window: int) -> pd.DataFrame:
        semivol = np.sqrt(downside.pow(2).rolling(window, min_periods=window).mean())
        return 1.0 / (1.0 + semivol)

    write_feature("RSTAB_3m", rstab(WINDOWS["3m"]))
    write_feature(
        "SKEW_p_2m",
        p_df.rolling(WINDOWS["2m"], min_periods=WINDOWS["2m"]).skew(),
    )

    delta = price_df.diff()
    n = WINDOWS["3d"]
    gain = np.sqrt(delta.where(delta > 0, 0.0).rolling(n, min_periods=n).max())
    loss = np.sqrt((-delta.where(delta < 0, 0.0)).rolling(n, min_periods=n).max())
    rs = gain / loss.replace(0, np.nan)
    write_feature("RSIp_3d", 100.0 - 100.0 / (1.0 + rs))
    del delta, gain, loss, rs
    gc.collect()

    write_feature(
        "KURT_p_3m",
        p_df.rolling(WINDOWS["3m"], min_periods=WINDOWS["3m"]).kurt(),
    )
    write_feature("RSTAB_2m", rstab(WINDOWS["2m"]))

    if slot != len(FEATURE_NAMES):
        raise AssertionError(f"Se escribieron {slot} features, se esperaban {len(FEATURE_NAMES)}")
    mm.flush()
    feature_meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    del p_df, y_df, price_df, log_price, downside
    gc.collect()
    return mm


FEATURES = build_feature_memmap()

# Los factores ya estan materializados; solo P es necesario para targets y grafos.
del Y, PRICE, VOLUME, dfyp, dftu, dfp, dfv
gc.collect()


# %% [markdown]
# ## 4. Cortes temporales, purga y escalado solo con train

# %%
all_sample_t = np.arange(WARMUP, len(P) - CFG.horizon, CFG.sample_stride)
test_start = len(P) - CFG.horizon - CFG.test_size
validation_end = test_start - CFG.purge
validation_start = validation_end - CFG.validation_size
train_end = validation_start - CFG.purge

train_t = all_sample_t[all_sample_t < train_end]
validation_t = all_sample_t[
    (all_sample_t >= validation_start) & (all_sample_t < validation_end)
]
test_t = all_sample_t[all_sample_t >= test_start]

if min(map(len, (train_t, validation_t, test_t))) == 0:
    raise ValueError("Alguno de los cortes temporales quedo vacio.")
if train_t.max() + CFG.horizon >= validation_t.min():
    raise AssertionError("Train y validacion no respetan la purga.")
if validation_t.max() + CFG.horizon >= test_t.min():
    raise AssertionError("Validacion y test no respetan la purga.")


def fit_streaming_scaler(
    mm: np.memmap, sample_times: np.ndarray, chunk_times: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(mm.shape[-1], dtype=np.float64)
    sums_sq = np.zeros(mm.shape[-1], dtype=np.float64)
    counts = np.zeros(mm.shape[-1], dtype=np.int64)
    for start in range(0, len(sample_times), chunk_times):
        idx = sample_times[start : start + chunk_times]
        block = np.asarray(mm[idx], dtype=np.float32).reshape(-1, mm.shape[-1])
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0.0).astype(np.float64)
        sums += safe.sum(axis=0)
        sums_sq += np.square(safe).sum(axis=0)
        counts += finite.sum(axis=0)
    if np.any(counts == 0):
        bad = [FEATURE_NAMES[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"Features sin observaciones finitas en train: {bad}")
    mean = sums / counts
    var = np.maximum(sums_sq / counts - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


FEATURE_MEAN, FEATURE_STD = fit_streaming_scaler(FEATURES, train_t)
print("Train:", timestamps[train_t[0]], "->", timestamps[train_t[-1]])
print("Validacion:", timestamps[validation_t[0]], "->", timestamps[validation_t[-1]])
print("Test:", timestamps[test_t[0]], "->", timestamps[test_t[-1]])


# %% [markdown]
# ## 5. Grafo dinamico causal
#
# Correlacion directa de `p` es espuria dentro del simplex. Para cada ancla
# semanal se usa exclusivamente la ventana historica terminada en esa ancla:
#
# 1. transformacion CLR con reemplazo multiplicativo numerico de ceros;
# 2. proyeccion aleatoria determinista de las trayectorias CLR estandarizadas;
# 3. similitud absoluta aproximada entre trayectorias;
# 4. proximidad energetica NPP `exp(-|E_i-E_j|/MAD)`, con
#    `E(t)=-CLR(p(t))`, identificada salvo una constante aditiva;
# 5. top-k y simetrizacion.
#
# No se estima una matriz de precision de 1.787 x 1.787: con una ventana menor
# que el numero de nodos seria singular, y presentarla como correlacion parcial
# seria matematicamente incoherente. El sketch CLR conserva la interpretacion
# composicional y hace viable el grafo completo.

# %%
class CausalGraphCache:
    def __init__(self, p: np.ndarray, cfg: Config, warmup: int):
        self.p = p
        self.cfg = cfg
        self.warmup = warmup
        self.n_nodes = p.shape[1]
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def anchor_for(self, t: int) -> int:
        if t < self.warmup:
            raise ValueError("t anterior al warmup")
        anchor = self.warmup + ((t - self.warmup) // self.cfg.edge_refresh) * self.cfg.edge_refresh
        if anchor > t:
            raise AssertionError("El grafo intento usar un ancla futura.")
        return int(anchor)

    def _build(self, anchor: int) -> Tuple[np.ndarray, np.ndarray]:
        start = max(0, anchor - self.cfg.edge_window + 1)
        hist = np.asarray(self.p[start : anchor + 1], dtype=np.float64)
        active = np.nanmax(hist, axis=0) > self.cfg.min_active_share
        active_ids = np.flatnonzero(active)

        # Todos los nodos conservan al menos self-loop, incluso si aun no cotizan.
        src_parts = [np.arange(self.n_nodes, dtype=np.int64)]
        dst_parts = [np.arange(self.n_nodes, dtype=np.int64)]
        weight_parts = [np.ones(self.n_nodes, dtype=np.float32)]

        if len(active_ids) > 1:
            logp = np.log(np.clip(hist[:, active], self.cfg.eps, None))
            clr = logp - logp.mean(axis=1, keepdims=True)
            centered = clr - clr.mean(axis=0, keepdims=True)
            scale = centered.std(axis=0, keepdims=True)
            standardized = centered / np.where(scale > self.cfg.eps, scale, 1.0)

            rng = np.random.default_rng(self.cfg.seed)
            projection = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32),
                size=(len(hist), self.cfg.edge_projection_dim),
            ) / math.sqrt(self.cfg.edge_projection_dim)
            embedding = standardized.T @ projection
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            embedding = embedding / np.where(norm > self.cfg.eps, norm, 1.0)
            clr_similarity = np.abs(embedding @ embedding.T)

            energy = -clr[-1]
            mad = np.median(np.abs(energy - np.median(energy))) + self.cfg.eps
            energy_proximity = np.exp(-np.abs(energy[:, None] - energy[None, :]) / mad)
            score = (
                self.cfg.edge_alpha_clr * clr_similarity
                + self.cfg.edge_gamma_npp * energy_proximity
            )
            np.fill_diagonal(score, -np.inf)
            k = min(self.cfg.edge_top_k, len(active_ids) - 1)
            nbr = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
            row = np.arange(len(active_ids))[:, None]
            w = score[row, nbr].astype(np.float32)
            src = np.repeat(active_ids, k)
            dst = active_ids[nbr.reshape(-1)]

            # Direcciones reciprocas: la atencion agregara src -> dst.
            src_parts.extend([src, dst])
            dst_parts.extend([dst, src])
            weight_parts.extend([w.reshape(-1), w.reshape(-1)])

        src_all = np.concatenate(src_parts)
        dst_all = np.concatenate(dst_parts)
        w_all = np.concatenate(weight_parts)
        keys = src_all * self.n_nodes + dst_all
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        unique_w = np.zeros(len(unique_keys), dtype=np.float32)
        np.maximum.at(unique_w, inverse, w_all)
        edge_index = np.vstack(
            [unique_keys // self.n_nodes, unique_keys % self.n_nodes]
        ).astype(np.int64)
        edge_weight = np.clip(unique_w, self.cfg.eps, 1.0).astype(np.float32)
        return edge_index, edge_weight

    def get(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        anchor = self.anchor_for(t)
        if anchor not in self.cache:
            self.cache[anchor] = self._build(anchor)
        return self.cache[anchor]


GRAPH_CACHE = CausalGraphCache(P, CFG, WARMUP)


# %% [markdown]
# ## 6. Dataset secuencial y batching de grafos disjuntos

# %%
class CryptoGraphDataset(Dataset):
    def __init__(self, sample_times: np.ndarray):
        self.sample_times = np.asarray(sample_times, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sample_times)

    def __getitem__(self, idx: int):
        t = int(self.sample_times[idx])
        raw = np.asarray(FEATURES[t], dtype=np.float32).copy()
        x = (raw - FEATURE_MEAN) / FEATURE_STD
        # NaN solo puede provenir de historia insuficiente o precio aun no listado;
        # se imputa a la media de train (cero despues del escalado), nunca con futuro.
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.asarray(P[t + CFG.horizon], dtype=np.float32).copy()
        target = target / target.sum(dtype=np.float64)
        edge_index, edge_weight = GRAPH_CACHE.get(t)
        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "edge_index": torch.from_numpy(edge_index),
            "edge_weight": torch.from_numpy(edge_weight),
            "t": t,
        }


def collate_graphs(samples: Sequence[dict]) -> dict:
    batch_size = len(samples)
    n_nodes = samples[0]["x"].shape[0]
    x = torch.cat([s["x"] for s in samples], dim=0)
    target = torch.stack([s["target"] for s in samples], dim=0)
    edges, weights = [], []
    for batch_id, sample in enumerate(samples):
        edges.append(sample["edge_index"] + batch_id * n_nodes)
        weights.append(sample["edge_weight"])
    return {
        "x": x,
        "target": target,
        "edge_index": torch.cat(edges, dim=1),
        "edge_weight": torch.cat(weights, dim=0),
        "times": np.array([s["t"] for s in samples], dtype=np.int64),
        "batch_size": batch_size,
        "n_nodes": n_nodes,
    }


train_ds = CryptoGraphDataset(train_t)
validation_ds = CryptoGraphDataset(validation_t)
test_ds = CryptoGraphDataset(test_t)

train_loader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_graphs,
)
validation_loader = DataLoader(
    validation_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_graphs,
)
test_loader = DataLoader(
    test_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_graphs,
)

for loader in (train_loader, validation_loader, test_loader):
    if not isinstance(loader.sampler, SequentialSampler):
        raise AssertionError("Se detecto un sampler no secuencial.")


# %% [markdown]
# ## 7. GAT ponderado sin dependencias externas
#
# `edge_weight` entra como sesgo logaritmico de la atencion. El softmax de
# atencion se calcula por nodo destino; el softmax final se calcula por grafo y
# obliga a que la prediccion sea una distribucion.

# %%
class WeightedGATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        h = self.linear(x).view(n, self.heads, self.out_dim)
        logits = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)
        logits = logits + torch.log(edge_weight.clamp_min(1e-12)).unsqueeze(-1)

        dst_index = dst.unsqueeze(-1).expand(-1, self.heads)
        max_per_dst = torch.full(
            (n, self.heads), -torch.inf, dtype=logits.dtype, device=logits.device
        )
        max_per_dst.scatter_reduce_(
            0, dst_index, logits, reduce="amax", include_self=True
        )
        exp_logits = torch.exp(logits - max_per_dst[dst])
        denom = torch.zeros((n, self.heads), dtype=logits.dtype, device=logits.device)
        denom.index_add_(0, dst, exp_logits)
        alpha = exp_logits / denom[dst].clamp_min(1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        messages = alpha.unsqueeze(-1) * h[src]
        out = torch.zeros(
            (n, self.heads, self.out_dim), dtype=h.dtype, device=h.device
        )
        out.index_add_(0, dst, messages)
        if self.concat:
            out = out.reshape(n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out + self.bias


class CryptoGAT(nn.Module):
    def __init__(self, n_features: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gat1 = WeightedGATLayer(
            n_features,
            cfg.hidden_dim,
            heads=cfg.attention_heads,
            concat=True,
            dropout=cfg.dropout,
        )
        wide = cfg.hidden_dim * cfg.attention_heads
        self.norm1 = nn.LayerNorm(wide)
        self.gat2 = WeightedGATLayer(
            wide,
            cfg.hidden_dim,
            heads=1,
            concat=False,
            dropout=cfg.dropout,
        )
        self.skip = nn.Linear(n_features, cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.logit_head = nn.Linear(cfg.hidden_dim, 1)
        self.gp_head = nn.Linear(cfg.hidden_dim, 1)
        self.eta_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_weight = batch["edge_weight"]
        b = batch["batch_size"]
        n = batch["n_nodes"]

        h = self.gat1(x, edge_index, edge_weight)
        h = self.norm1(F.elu(h))
        h = F.dropout(h, p=self.cfg.dropout, training=self.training)
        h = self.gat2(h, edge_index, edge_weight)
        h = self.norm2(F.elu(h + self.skip(x)))

        logits = self.logit_head(h).view(b, n)
        gp = torch.sigmoid(self.gp_head(h)).view(b, n)
        eta = F.softplus(self.eta_head(h)).view(b, n) + self.cfg.eps
        energy = gp / eta
        log_p = F.log_softmax(logits, dim=1)
        log_q_npp = F.log_softmax(-energy, dim=1)
        return {
            "logits": logits,
            "log_p": log_p,
            "p": log_p.exp(),
            "gp": gp,
            "eta": eta,
            "energy": energy,
            "log_q_npp": log_q_npp,
            "q_npp": log_q_npp.exp(),
        }


# %% [markdown]
# ## 8. Loss distributiva y regularizacion NPP
#
# Para el GAT libre:
#
# `L = H(p(t+1), p_hat(t+1))`.
#
# Para GAT+NPP se agrega exactamente el residuo de la solucion cerrada del
# paper:
#
# `r_j = log(p_hat_j) + gp_j/eta_j + log Z`,
#
# donde `log Z = logsumexp(-gp/eta)`. Tambien se entrena la distribucion
# energetica auxiliar `q_NPP=softmax(-gp/eta)` contra el mismo target. Esto no
# usa informacion del test: el target entra solo en el gradiente de las muestras
# pertenecientes a train.

# %%
def distribution_ce(target: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    return -(target * log_prob).sum(dim=1).mean()


def loss_components(output: dict, target: torch.Tensor, use_npp: bool) -> dict:
    pred_ce = distribution_ce(target, output["log_p"])
    zero = pred_ce.new_zeros(())
    if not use_npp:
        return {
            "total": pred_ce,
            "pred_ce": pred_ce,
            "npp_residual": zero,
            "energy_ce": zero,
            "eta_anchor": zero,
            "smooth": zero,
        }

    energy = output["energy"]
    log_z = torch.logsumexp(-energy, dim=1, keepdim=True)
    residual = output["log_p"] + energy + log_z
    npp_residual = residual.square().mean()
    energy_ce = distribution_ce(target, output["log_q_npp"])
    eta_anchor = torch.log(output["eta"].mean(dim=1).clamp_min(CFG.eps)).square().mean()
    if energy.shape[0] > 1:
        smooth = (energy[1:] - energy[:-1]).square().mean()
    else:
        smooth = zero
    total = (
        pred_ce
        + CFG.lambda_npp * npp_residual
        + CFG.lambda_energy_ce * energy_ce
        + CFG.lambda_eta_anchor * eta_anchor
        + CFG.lambda_energy_smooth * smooth
    )
    return {
        "total": total,
        "pred_ce": pred_ce,
        "npp_residual": npp_residual,
        "energy_ce": energy_ce,
        "eta_anchor": eta_anchor,
        "smooth": smooth,
    }


def to_device(batch: dict) -> dict:
    result = dict(batch)
    for key in ("x", "target", "edge_index", "edge_weight"):
        result[key] = result[key].to(DEVICE, non_blocking=True)
    return result


@torch.no_grad()
def validation_ce(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        out = model(batch)
        per_graph = -(batch["target"] * out["log_p"]).sum(dim=1)
        total += per_graph.sum().item()
        count += len(per_graph)
    return total / count


def train_model(name: str, use_npp: bool) -> Tuple[CryptoGAT, pd.DataFrame]:
    set_seed(CFG.seed)
    model = CryptoGAT(len(FEATURE_NAMES), CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    best_state: Optional[dict] = None
    best_val = math.inf
    stale = 0
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        sums: Dict[str, float] = {}
        seen = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            pieces = loss_components(output, batch["target"], use_npp=use_npp)
            pieces["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()

            b = batch["batch_size"]
            seen += b
            for key, value in pieces.items():
                sums[key] = sums.get(key, 0.0) + value.detach().item() * b

        val_ce = validation_ce(model, validation_loader)
        row = {"model": name, "epoch": epoch, "val_ce": val_ce}
        row.update({f"train_{k}": v / seen for k, v in sums.items()})
        history.append(row)
        print(row)

        if val_ce < best_val - 1e-7:
            best_val = val_ce
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= CFG.patience:
                print(f"Early stopping de {name} en epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("No se genero un checkpoint valido.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# %% [markdown]
# ## 9. Entrenamiento: GAT libre vs GAT + NPP

# %%
FREE_MODEL, history_free = train_model("GAT libre", use_npp=False)
NPP_MODEL, history_npp = train_model("GAT + NPP", use_npp=True)


# %% [markdown]
# ## 10. Test final y benchmarks
#
# El test se consume por primera vez aqui. Ademas de CE/KL se reporta distancia
# de Aitchison, apropiada para datos composicionales. La persistencia usa
# `p(t)` para predecir `p(t+1)`.

# %%
@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    p_list, q_list, gp_list, eta_list, y_list, t_list = [], [], [], [], [], []
    for raw_batch in loader:
        times_batch = raw_batch["times"]
        batch = to_device(raw_batch)
        out = model(batch)
        p_list.append(out["p"].cpu().numpy())
        q_list.append(out["q_npp"].cpu().numpy())
        gp_list.append(out["gp"].cpu().numpy())
        eta_list.append(out["eta"].cpu().numpy())
        y_list.append(batch["target"].cpu().numpy())
        t_list.append(times_batch)
    return {
        "p": np.concatenate(p_list),
        "q_npp": np.concatenate(q_list),
        "gp": np.concatenate(gp_list),
        "eta": np.concatenate(eta_list),
        "target": np.concatenate(y_list),
        "t": np.concatenate(t_list),
    }


def clr(values: np.ndarray, eps: float) -> np.ndarray:
    safe = np.clip(values.astype(np.float64), eps, None)
    safe /= safe.sum(axis=1, keepdims=True)
    log_values = np.log(safe)
    return log_values - log_values.mean(axis=1, keepdims=True)


def distribution_metrics(name: str, target: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred.astype(np.float64), CFG.eps, None)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.clip(target.astype(np.float64), 0.0, None)
    target /= target.sum(axis=1, keepdims=True)
    target_safe = np.clip(target, CFG.eps, None)
    ce = -np.sum(target * np.log(pred), axis=1)
    entropy = -np.sum(target * np.log(target_safe), axis=1)
    midpoint = 0.5 * (target_safe + pred)
    js = 0.5 * np.sum(target * (np.log(target_safe) - np.log(midpoint)), axis=1)
    js += 0.5 * np.sum(pred * (np.log(pred) - np.log(midpoint)), axis=1)
    aitchison = np.sqrt(np.mean(np.square(clr(target, CFG.eps) - clr(pred, CFG.eps)), axis=1))
    return {
        "Modelo": name,
        "CE": ce.mean(),
        "KL": (ce - entropy).mean(),
        "JS": js.mean(),
        "Aitchison": aitchison.mean(),
        "MAE_nodo": np.mean(np.abs(target - pred)),
        "RMSE_nodo": np.sqrt(np.mean(np.square(target - pred))),
        "Error_cierre_max": np.max(np.abs(pred.sum(axis=1) - 1.0)),
        "N_test": len(target),
    }


pred_free = collect_predictions(FREE_MODEL, test_loader)
pred_npp = collect_predictions(NPP_MODEL, test_loader)
if not np.array_equal(pred_free["t"], pred_npp["t"]):
    raise AssertionError("Los modelos no fueron evaluados en las mismas fechas.")
if not np.allclose(pred_free["target"], pred_npp["target"]):
    raise AssertionError("Los modelos no comparten exactamente el mismo target.")

target_test = pred_free["target"]
persistence = P[pred_free["t"]]
metrics = pd.DataFrame(
    [
        distribution_metrics("Persistencia", target_test, persistence),
        distribution_metrics("GAT libre", target_test, pred_free["p"]),
        distribution_metrics("GAT + NPP", target_test, pred_npp["p"]),
        distribution_metrics("Cabeza energetica NPP", target_test, pred_npp["q_npp"]),
    ]
).sort_values("CE")
display(metrics)


def node_error_metrics(name: str, target: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Metricas de error por activo sobre exactamente el mismo bloque temporal de test."""
    pred = np.clip(pred.astype(np.float64), CFG.eps, None)
    pred /= pred.sum(axis=1, keepdims=True)
    target = np.clip(target.astype(np.float64), 0.0, None)
    target /= target.sum(axis=1, keepdims=True)
    error = pred - target
    clr_error = clr(pred, CFG.eps) - clr(target, CFG.eps)
    mean_share = np.mean(target, axis=0)
    return pd.DataFrame(
        {
            "Modelo": name,
            "Nodo": assets,
            "MAE": np.mean(np.abs(error), axis=0),
            "RMSE": np.sqrt(np.mean(np.square(error), axis=0)),
            "Sesgo": np.mean(error, axis=0),
            "Participacion_media_real": mean_share,
            "MAE_relativa": np.mean(np.abs(error), axis=0) / np.maximum(mean_share, CFG.eps),
            "RMSE_CLR": np.sqrt(np.mean(np.square(clr_error), axis=0)),
            "Tasa_masa_casi_cero": np.mean(target <= 1e-8, axis=0),
            "N_test": target.shape[0],
        }
    )


metrics_by_node = pd.concat(
    [
        node_error_metrics("Persistencia", target_test, persistence),
        node_error_metrics("GAT libre", target_test, pred_free["p"]),
        node_error_metrics("GAT + NPP", target_test, pred_npp["p"]),
        node_error_metrics("Cabeza energetica NPP", target_test, pred_npp["q_npp"]),
    ],
    ignore_index=True,
)
display(metrics_by_node.sort_values(["Nodo", "RMSE", "MAE"]))


# %% [markdown]
# ## 11. Auditoria final y persistencia de resultados

# %%
def audit_no_leakage() -> dict:
    feature_text = " ".join(FEATURE_NAMES).lower()
    checks = {
        "sin_target_en_features": "target" not in feature_text and "t+1" not in feature_text,
        "horizonte_positivo": CFG.horizon > 0,
        "train_antes_validacion": train_t.max() + CFG.horizon < validation_t.min(),
        "validacion_antes_test": validation_t.max() + CFG.horizon < test_t.min(),
        "train_sin_shuffle": isinstance(train_loader.sampler, SequentialSampler),
        "simplex_target": bool(np.allclose(target_test.sum(axis=1), 1.0, atol=1e-6)),
        "grafo_causal_train": all(GRAPH_CACHE.anchor_for(int(t)) <= t for t in train_t),
        "grafo_causal_validation": all(GRAPH_CACHE.anchor_for(int(t)) <= t for t in validation_t),
        "grafo_causal_test": all(GRAPH_CACHE.anchor_for(int(t)) <= t for t in test_t),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(f"Fallo la auditoria anti-leakage: {checks}")
    return checks


audit = audit_no_leakage()
print(json.dumps(audit, indent=2))

results_dir = Path(CFG.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)
metrics.to_csv(results_dir / "metricas_test.csv", index=False)
metrics_by_node.to_csv(results_dir / "metricas_test_por_nodo.csv", index=False)
for model_name, file_name in {
    "Persistencia": "metricas_por_nodo_persistencia.csv",
    "GAT libre": "metricas_por_nodo_gat_libre.csv",
    "GAT + NPP": "metricas_por_nodo_gat_npp.csv",
    "Cabeza energetica NPP": "metricas_por_nodo_cabeza_energetica_npp.csv",
}.items():
    metrics_by_node.loc[metrics_by_node["Modelo"] == model_name].to_csv(
        results_dir / file_name, index=False
    )
pd.concat([history_free, history_npp], ignore_index=True).to_csv(
    results_dir / "historial_entrenamiento.csv", index=False
)
torch.save(FREE_MODEL.state_dict(), results_dir / "gat_libre.pt")
torch.save(NPP_MODEL.state_dict(), results_dir / "gat_npp.pt")
np.savez_compressed(
    results_dir / "predicciones_test.npz",
    assets=np.asarray(assets),
    t=pred_free["t"],
    timestamps=np.asarray(timestamps[pred_free["t"] + CFG.horizon].astype(str)),
    target=target_test.astype(np.float32),
    persistence=persistence.astype(np.float32),
    gat_free=pred_free["p"].astype(np.float32),
    gat_npp=pred_npp["p"].astype(np.float32),
    q_npp=pred_npp["q_npp"].astype(np.float32),
    gp_latent=pred_npp["gp"].astype(np.float32),
    eta_latent=pred_npp["eta"].astype(np.float32),
)
(results_dir / "config.json").write_text(
    json.dumps(asdict(CFG), indent=2, ensure_ascii=False), encoding="utf-8"
)
(results_dir / "auditoria_no_leakage.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Resultados guardados en:", results_dir)


# %% [markdown]
# ## Nota de identificacion cientifica
#
# El NPP determina exactamente la energia relativa
# `E_j = gp_j / eta_j = -log(p_j) - log(Z)` y, por la restriccion de cierre,
# `p=softmax(-E)`. Con cuotas y volumenes transaccionales se puede contrastar
# esa estructura energetica y su utilidad predictiva. No obstante, separar
# causalmente `gp` y `eta` requiere proxies independientes de los capitales
# economico, social, estructural, simbolico y estrategico. Por eso este notebook
# guarda ambas cabezas como variables **latentes** y centra la comparacion
# cientifica en:
#
# 1. GAT libre vs GAT regularizado por NPP;
# 2. estabilidad fuera de muestra en el test cronologico;
# 3. CE/KL/JS y distancia de Aitchison, no MSE aislado;
# 4. coherencia del simplex y del grafo CLR causal.


# %% [markdown]
# ## Nota cientifica de la ejecucion
#
# Esta nota se calcula desde el test cronologico de la ejecucion corriente. Su
# objetivo es impedir conclusiones basadas solamente en una metrica agregada.
# CE/KL/JS evalúan la distribucion; MAE aproxima el error absoluto habitual;
# RMSE enfatiza errores grandes; y Aitchison mide discrepancias relativas en
# log-ratios, por lo que es especialmente sensible a masas nulas o diminutas.

# %%
def scientific_run_summary() -> dict:
    by_model = {name: frame.set_index("Nodo") for name, frame in metrics_by_node.groupby("Modelo")}
    persistence_nodes = by_model["Persistencia"]
    free_nodes = by_model["GAT libre"]
    npp_nodes = by_model["GAT + NPP"]
    active = persistence_nodes["Participacion_media_real"] > 0

    squared_error_weight = np.square(npp_nodes.loc[active, "RMSE"])
    squared_error_weight /= squared_error_weight.sum()
    ranked_contribution = squared_error_weight.sort_values(ascending=False)

    global_by_model = metrics.set_index("Modelo")
    persistence_global = global_by_model.loc["Persistencia"]
    npp_global = global_by_model.loc["GAT + NPP"]

    return {
        "n_test": int(npp_global["N_test"]),
        "nodos_totales": int(len(npp_nodes)),
        "nodos_activos_test": int(active.sum()),
        "mejora_npp_vs_persistencia": {
            "KL_pct": float(100 * (1 - npp_global["KL"] / persistence_global["KL"])),
            "JS_pct": float(100 * (1 - npp_global["JS"] / persistence_global["JS"])),
            "MAE_pct": float(100 * (1 - npp_global["MAE_nodo"] / persistence_global["MAE_nodo"])),
            "RMSE_pct": float(100 * (1 - npp_global["RMSE_nodo"] / persistence_global["RMSE_nodo"])),
        },
        "cobertura_npp_vs_persistencia": {
            "nodos_con_menor_RMSE": int((npp_nodes.loc[active, "RMSE"] < persistence_nodes.loc[active, "RMSE"]).sum()),
            "nodos_con_menor_MAE": int((npp_nodes.loc[active, "MAE"] < persistence_nodes.loc[active, "MAE"]).sum()),
        },
        "npp_vs_gat_libre": {
            "nodos_con_menor_RMSE": int((npp_nodes.loc[active, "RMSE"] < free_nodes.loc[active, "RMSE"]).sum()),
            "nodos_con_menor_MAE": int((npp_nodes.loc[active, "MAE"] < free_nodes.loc[active, "MAE"]).sum()),
        },
        "concentracion_error_cuadratico_npp": {
            "nodo_dominante": str(ranked_contribution.index[0]),
            "aporte_nodo_dominante_pct": float(100 * ranked_contribution.iloc[0]),
            "aporte_dos_primeros_pct": float(100 * ranked_contribution.iloc[:2].sum()),
            "aporte_cinco_primeros_pct": float(100 * ranked_contribution.iloc[:5].sum()),
        },
        "diagnostico_aitchison": {
            "fraccion_targets_cero_o_epsilon": float(np.mean(target_test <= CFG.eps)),
            "prediccion_softmax_estrictamente_positiva": bool(np.all(pred_npp["p"] > 0)),
            "interpretacion": (
                "Aitchison pondera simetricamente log-ratios y puede ser dominado por "
                "nodos nulos o diminutos, mientras CE y los errores en nivel ponderan "
                "principalmente la masa economicamente observada."
            ),
        },
        "advertencias": [
            "La ventaja NPP frente al GAT libre debe confirmarse con varias semillas y ventanas temporales.",
            "El nodo dominante del error debe identificarse contra la fuente antes de interpretarlo economicamente.",
            "RMSE no es un intervalo simetrico ni una banda de confianza.",
        ],
    }


scientific_summary = scientific_run_summary()
(results_dir / "nota_cientifica.json").write_text(
    json.dumps(scientific_summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(scientific_summary, indent=2, ensure_ascii=False))

