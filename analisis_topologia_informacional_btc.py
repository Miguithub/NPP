"""Análisis causal de la información contenida en ``BTC_y``.

El script reproduce y ordena el recorrido experimental documentado en el chat
compartido "Acceso a Drive": desde asociaciones contemporáneas hasta modelos
de régimen de volatilidad. ``BTC_y`` se interpreta como la coordenada de BTC
en el simplex de participación transaccional, no como una señal de precio.

Contratos metodológicos
-----------------------
* no se barajan observaciones;
* toda feature de t utiliza únicamente información disponible hasta t;
* los targets futuros usan t+1, ..., t+h;
* el holdout final es cronológico y se purga el borde train/test;
* cuantiles, umbrales y modelos se ajustan exclusivamente con train;
* el walk-forward es expansivo y usa ``gap=h``;
* cada fila exportada identifica experimento, modelo, features y split.

Uso en Colab (rutas originales)::

    python analisis_topologia_informacional_btc.py --mount-drive

Uso local::

    python analisis_topologia_informacional_btc.py \
      --price-path 0626p.parquet --simplex-path 0626dfyp.parquet \
      --output-dir resultados/topologia_informacional_btc

Prueba rápida sin datos externos::

    python analisis_topologia_informacional_btc.py --synthetic \
      --synthetic-rows 4000 --folds 2 --backend histgb

La salida principal es ``13_utilidad_topologia.csv``. Una mejora positiva en
``gain_macro_f1`` o ``gain_balanced_accuracy`` indica que agregar la familia
de variables de ``BTC_y`` mejora al control de igual arquitectura.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import levene
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier


SOURCE_CHAT = "https://chatgpt.com/share/6a9da6eb-5c68-83e9-8b4d-c3c4109c5ff1"
CLASS_NAMES = {
    "3_regimes": ("LOW", "MID", "HIGH"),
    "5_extreme_4_5": ("EXTREME_LOW", "LOW", "MID", "HIGH", "EXTREME_HIGH"),
    "5_quintiles": ("VERY_LOW", "LOW", "MID", "HIGH", "VERY_HIGH"),
}


@dataclass(frozen=True)
class Config:
    price_path: str = "/content/drive/MyDrive/0626p.parquet"
    simplex_path: str = "/content/drive/MyDrive/0626dfyp.parquet"
    output_dir: str = (
        "/content/drive/MyDrive/Neural/NPP/Cripto/"
        "topologia_informacional_BTC"
    )
    price_column: str = "BTCUSDT"
    topology_column: str = "BTC_y"
    timestamp_column: str | None = None
    start_timestamp: str = "2026-01-01 00:00:00+00:00"
    frequency: str = "10min"

    holdout_fraction: float = 0.30
    future_horizon: int = 2
    past_vol_window: int = 2
    max_lag: int = 20
    volatility_horizons: tuple[int, ...] = (1, 2, 3, 6, 12)
    folds: int = 5
    seed: int = 42
    extreme_tail: float = 0.045

    tree_depth: int = 4
    tree_min_leaf: int = 250
    boosting_iterations: int = 300
    boosting_learning_rate: float = 0.03
    boosting_leaves: int = 15
    boosting_depth: int = 5
    permutation_repeats: int = 8
    eps: float = 1e-12

    backend: str = "auto"
    run_walk_forward: bool = True
    synthetic: bool = False
    synthetic_rows: int = 6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-path", default=Config.price_path)
    parser.add_argument("--simplex-path", default=Config.simplex_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--price-column", default=Config.price_column)
    parser.add_argument("--topology-column", default=Config.topology_column)
    parser.add_argument("--timestamp-column")
    parser.add_argument("--start-timestamp", default=Config.start_timestamp)
    parser.add_argument("--frequency", default=Config.frequency)
    parser.add_argument("--holdout-fraction", type=float, default=Config.holdout_fraction)
    parser.add_argument("--future-horizon", type=int, default=Config.future_horizon)
    parser.add_argument("--past-vol-window", type=int, default=Config.past_vol_window)
    parser.add_argument("--max-lag", type=int, default=Config.max_lag)
    parser.add_argument("--folds", type=int, default=Config.folds)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--extreme-tail", type=float, default=Config.extreme_tail)
    parser.add_argument("--tree-min-leaf", type=int, default=Config.tree_min_leaf)
    parser.add_argument(
        "--boosting-iterations", type=int, default=Config.boosting_iterations
    )
    parser.add_argument(
        "--permutation-repeats", type=int, default=Config.permutation_repeats
    )
    parser.add_argument("--backend", choices=("auto", "lightgbm", "histgb"), default="auto")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-rows", type=int, default=Config.synthetic_rows)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        price_path=args.price_path,
        simplex_path=args.simplex_path,
        output_dir=args.output_dir,
        price_column=args.price_column,
        topology_column=args.topology_column,
        timestamp_column=args.timestamp_column,
        start_timestamp=args.start_timestamp,
        frequency=args.frequency,
        holdout_fraction=args.holdout_fraction,
        future_horizon=args.future_horizon,
        past_vol_window=args.past_vol_window,
        max_lag=args.max_lag,
        folds=args.folds,
        seed=args.seed,
        extreme_tail=args.extreme_tail,
        tree_min_leaf=args.tree_min_leaf,
        boosting_iterations=args.boosting_iterations,
        permutation_repeats=args.permutation_repeats,
        backend=args.backend,
        run_walk_forward=not args.skip_walk_forward,
        synthetic=args.synthetic,
        synthetic_rows=args.synthetic_rows,
    )


def validate_config(cfg: Config) -> None:
    if not 0.05 <= cfg.holdout_fraction <= 0.50:
        raise ValueError("holdout_fraction debe estar entre 0.05 y 0.50.")
    if cfg.future_horizon < 1 or cfg.past_vol_window < 1:
        raise ValueError("Los horizontes deben ser enteros positivos.")
    if cfg.max_lag < 0 or cfg.folds < 2:
        raise ValueError("max_lag debe ser >= 0 y folds >= 2.")
    if not 0.0 < cfg.extreme_tail < 0.10:
        raise ValueError("extreme_tail debe estar entre 0 y 0.10.")


def mount_drive_if_requested(requested: bool) -> None:
    if not requested:
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--mount-drive sólo está disponible en Google Colab.") from exc
    drive.mount("/content/drive")


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def _datetime_index(frame: pd.DataFrame, cfg: Config, label: str) -> pd.DataFrame:
    out = frame.copy()
    timestamp_column = cfg.timestamp_column
    if timestamp_column:
        if timestamp_column not in out.columns:
            raise KeyError(f"{label}: no existe timestamp_column={timestamp_column!r}.")
        index = pd.to_datetime(out.pop(timestamp_column), utc=True, errors="raise")
    elif isinstance(out.index, pd.DatetimeIndex):
        index = pd.to_datetime(out.index, utc=True, errors="raise")
    else:
        candidates = [c for c in ("timestamp", "datetime", "date", "time") if c in out.columns]
        if candidates:
            index = pd.to_datetime(out.pop(candidates[0]), utc=True, errors="raise")
        else:
            index = pd.date_range(
                start=pd.Timestamp(cfg.start_timestamp), periods=len(out), freq=cfg.frequency
            )
    out.index = pd.DatetimeIndex(index, name="timestamp")
    if out.index.has_duplicates:
        raise ValueError(f"{label}: hay timestamps duplicados.")
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"{label}: el orden temporal no es creciente; no se reordena silenciosamente.")
    return out


def make_synthetic_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.synthetic_rows < 1200:
        raise ValueError("synthetic_rows debe ser al menos 1200.")
    rng = np.random.default_rng(cfg.seed)
    index = pd.date_range(cfg.start_timestamp, periods=cfg.synthetic_rows, freq=cfg.frequency)
    latent = np.zeros(cfg.synthetic_rows, dtype=np.float64)
    for t in range(1, len(latent)):
        latent[t] = 0.985 * latent[t - 1] + rng.normal(0.0, 0.12)
    topology = 0.12 + 0.18 / (1.0 + np.exp(-latent))
    topology = np.clip(topology + rng.normal(0.0, 0.0025, len(topology)), 1e-5, 0.95)
    dy_abs = np.abs(np.diff(topology, prepend=topology[0]))
    conditional_sigma = 0.0008 + 0.0020 * (topology > np.quantile(topology, 0.80))
    conditional_sigma += 0.05 * dy_abs
    returns = rng.normal(0.0, conditional_sigma)
    price = 45_000.0 * np.exp(np.cumsum(returns))
    return (
        pd.DataFrame({cfg.price_column: price}, index=index),
        pd.DataFrame({cfg.topology_column: topology}, index=index),
    )


def load_inputs(cfg: Config) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if cfg.synthetic:
        prices, simplex = make_synthetic_data(cfg)
    else:
        prices = _datetime_index(read_table(cfg.price_path), cfg, "precios")
        simplex = _datetime_index(read_table(cfg.simplex_path), cfg, "simplex")
    if cfg.price_column not in prices.columns:
        raise KeyError(f"No existe {cfg.price_column!r} en el archivo de precios.")
    topology_column = cfg.topology_column
    if topology_column not in simplex.columns:
        aliases = [c for c in ("BTC_y", "BTC_p") if c in simplex.columns]
        if len(aliases) == 1:
            warnings.warn(
                f"Se usará {aliases[0]!r} porque {topology_column!r} no existe.", stacklevel=2
            )
            topology_column = aliases[0]
        else:
            raise KeyError(f"No existe {cfg.topology_column!r} en el archivo del simplex.")
    joined = pd.concat(
        [
            pd.to_numeric(prices[cfg.price_column], errors="coerce").rename("price"),
            pd.to_numeric(simplex[topology_column], errors="coerce").rename("BTC_y"),
        ],
        axis=1,
        join="inner",
    )
    if len(joined) == 0:
        raise ValueError("Los archivos no comparten timestamps.")
    if not joined.index.is_monotonic_increasing:
        raise AssertionError("El join alteró el orden cronológico.")
    audit = pd.DataFrame(
        [
            {
                "experiment": "input_audit",
                "model": "none",
                "feature_set": "price+BTC_y",
                "rows_joined": len(joined),
                "missing_price": int(joined["price"].isna().sum()),
                "missing_BTC_y": int(joined["BTC_y"].isna().sum()),
                "nonpositive_price": int((joined["price"] <= 0).sum()),
                "BTC_y_outside_unit_interval": int(
                    ((joined["BTC_y"] <= 0) | (joined["BTC_y"] >= 1)).sum()
                ),
                "start": joined.index.min(),
                "end": joined.index.max(),
            }
        ]
    )
    clean = joined.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[(clean["price"] > 0) & clean["BTC_y"].between(0, 1, inclusive="neither")]
    if len(clean) < 1000:
        raise ValueError(f"Sólo quedan {len(clean)} filas válidas; se requieren al menos 1000.")
    return clean["price"], clean["BTC_y"], audit


def future_rms(returns: pd.Series, horizon: int) -> pd.Series:
    future_squares = [returns.shift(-step).pow(2) for step in range(1, horizon + 1)]
    return pd.concat(future_squares, axis=1).mean(axis=1).pow(0.5)


def trailing_rms(returns: pd.Series, window: int, include_current: bool = True) -> pd.Series:
    base = returns if include_current else returns.shift(1)
    return base.pow(2).rolling(window, min_periods=window).mean().pow(0.5)


def finite_frame(data: dict[str, pd.Series] | pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(data).replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    frame = finite_frame({"x": x, "y": y})
    if len(frame) < 4 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method=method))


def mi_regression(x: pd.Series, y: pd.Series, seed: int) -> float:
    frame = finite_frame({"x": x, "y": y})
    if len(frame) < 10 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(mutual_info_regression(frame[["x"]], frame["y"], random_state=seed)[0])


def mi_classification(x: pd.Series, y: pd.Series, seed: int) -> float:
    frame = finite_frame({"x": x, "y": y})
    if len(frame) < 10 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(mutual_info_classif(frame[["x"]], frame["y"].astype(int), random_state=seed)[0])


def partial_corr(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> float:
    frame = pd.concat([x.rename("x"), y.rename("y"), controls], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 10 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    z = frame.drop(columns=["x", "y"]).to_numpy(dtype=np.float64)
    rx = frame["x"].to_numpy() - LinearRegression().fit(z, frame["x"]).predict(z)
    ry = frame["y"].to_numpy() - LinearRegression().fit(z, frame["y"]).predict(z)
    return float(np.corrcoef(rx, ry)[0, 1])


def dependency_row(
    experiment: str,
    feature: str,
    target: str,
    x: pd.Series,
    y: pd.Series,
    seed: int,
    sample: str,
) -> dict[str, object]:
    aligned = finite_frame({"x": x, "y": y})
    return {
        "experiment": experiment,
        "model": "nonparametric_dependency",
        "feature_set": feature,
        "feature": feature,
        "target": target,
        "sample": sample,
        "n": len(aligned),
        "pearson": safe_corr(x, y, "pearson"),
        "spearman": safe_corr(x, y, "spearman"),
        "mutual_information": mi_regression(x, y, seed),
    }


def build_core(price: pd.Series, topology: pd.Series, cfg: Config) -> pd.DataFrame:
    returns = np.log(price / price.shift(1))
    core = pd.DataFrame(index=price.index)
    core["price"] = price
    core["BTC_y"] = topology
    core["dBTC_y"] = topology.diff()
    core["abs_dBTC_y"] = core["dBTC_y"].abs()
    core["return"] = returns
    core["abs_return"] = returns.abs()
    core["sigma_past"] = trailing_rms(returns, cfg.past_vol_window, include_current=True)
    core["sigma_pre_return"] = trailing_rms(
        returns, cfg.past_vol_window, include_current=False
    )
    core["sigma_future"] = future_rms(returns, cfg.future_horizon)
    core["direction"] = (returns > 0).astype(int)
    return core


def build_topology_features(core: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    y = core["BTC_y"].clip(cfg.eps, 1.0 - cfg.eps)
    dy = y.diff()
    features = pd.DataFrame(index=core.index)
    features["BTC_y"] = y
    features["dBTC_y"] = dy
    features["abs_dBTC_y"] = dy.abs()
    features["dlog_BTC_y"] = np.log(y).diff()
    features["shannon_surprisal"] = -np.log(y)
    features["logit_BTC_y"] = np.log(y / (1.0 - y))

    lags = (1, 2, 3, 6, 12, 18, 36)
    for lag in lags:
        features[f"BTC_y_lag_{lag}"] = y.shift(lag)
        features[f"dBTC_y_lag_{lag}"] = dy.shift(lag)
        features[f"abs_dBTC_y_lag_{lag}"] = dy.abs().shift(lag)

    for window in (6, 12, 18, 36):
        mean = y.rolling(window, min_periods=window).mean()
        std = y.rolling(window, min_periods=window).std()
        features[f"BTC_y_mean_{window}"] = mean
        features[f"BTC_y_std_{window}"] = std
        features[f"BTC_y_dev_mean_{window}"] = y / mean.clip(lower=cfg.eps) - 1.0
        features[f"BTC_y_momentum_{window}"] = y / y.shift(window).clip(lower=cfg.eps) - 1.0

    for span in (6, 12, 18, 36):
        ema = y.ewm(span=span, adjust=False, min_periods=span).mean()
        features[f"BTC_y_dev_ema_{span}"] = y / ema.clip(lower=cfg.eps) - 1.0
    return features.replace([np.inf, -np.inf], np.nan)


def build_volatility_features(core: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sigma = core["sigma_past"].clip(lower=cfg.eps)
    features = pd.DataFrame(index=core.index)
    features["sigma_past"] = sigma
    for lag in (1, 2, 3, 6, 12, 18, 36):
        features[f"sigma_lag_{lag}"] = sigma.shift(lag)
    for window in (6, 12, 18, 36):
        mean = sigma.rolling(window, min_periods=window).mean()
        std = sigma.rolling(window, min_periods=window).std()
        features[f"sigma_mean_{window}"] = mean
        features[f"sigma_std_{window}"] = std
        features[f"sigma_dev_mean_{window}"] = sigma / mean.clip(lower=cfg.eps) - 1.0
    for span in (6, 12, 18, 36):
        ema = sigma.ewm(span=span, adjust=False, min_periods=span).mean()
        features[f"sigma_dev_ema_{span}"] = sigma / ema.clip(lower=cfg.eps) - 1.0
    return features.replace([np.inf, -np.inf], np.nan)


def build_interactions(
    core: pd.DataFrame, topology_features: pd.DataFrame
) -> pd.DataFrame:
    sigma = core["sigma_past"]
    chosen = [
        "BTC_y",
        "dBTC_y",
        "abs_dBTC_y",
        "BTC_y_dev_mean_6",
        "BTC_y_dev_mean_12",
        "BTC_y_dev_mean_18",
        "BTC_y_dev_mean_36",
        "BTC_y_std_6",
        "BTC_y_std_12",
        "BTC_y_std_18",
        "BTC_y_std_36",
    ]
    return pd.DataFrame(
        {f"sigma_x_{column}": sigma * topology_features[column] for column in chosen},
        index=core.index,
    )


def chronological_holdout(
    frame: pd.DataFrame, holdout_fraction: float, gap: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cut = int(math.floor(len(frame) * (1.0 - holdout_fraction)))
    train_end = cut - gap
    if train_end <= 0 or cut >= len(frame):
        raise ValueError("Split cronológico inválido.")
    train = frame.iloc[:train_end].copy()
    test = frame.iloc[cut:].copy()
    if train.index.max() >= test.index.min():
        raise AssertionError("El split cronológico se superpone.")
    return train, test, cut


def exploratory_analysis(
    core: pd.DataFrame,
    discovery: pd.DataFrame,
    cfg: Config,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    seed = cfg.seed
    rows = [
        dependency_row(
            "A_contemporaneous",
            "BTC_y",
            "price_t",
            discovery["BTC_y"],
            discovery["price"],
            seed,
            "development_train",
        ),
        dependency_row(
            "B_price_t_plus_1",
            "BTC_y",
            "price_t_plus_1",
            discovery["BTC_y"],
            discovery["price"].shift(-1),
            seed,
            "development_train",
        ),
    ]
    basic_dependencies = pd.DataFrame(rows)

    y = discovery["BTC_y"].clip(cfg.eps, 1.0 - cfg.eps)
    derived = pd.DataFrame(index=discovery.index)
    derived["BTC_y"] = y
    derived["dBTC_y"] = y.diff()
    derived["abs_dBTC_y"] = y.diff().abs()
    derived["dlog_BTC_y"] = np.log(y).diff()
    derived["shannon_surprisal"] = -np.log(y)
    derived["logit_BTC_y"] = np.log(y / (1.0 - y))
    ema_week = y.ewm(span=1008, adjust=False, min_periods=1008).mean()
    ema_month = y.ewm(span=4032, adjust=False, min_periods=4032).mean()
    ema_three_month = y.ewm(span=12128, adjust=False, min_periods=12128).mean()
    derived["macd_y_1w_1m"] = ema_week / ema_month.clip(lower=cfg.eps) - 1.0
    derived["macd_y_1m_3m"] = ema_month / ema_three_month.clip(lower=cfg.eps) - 1.0

    target_next_price = discovery["price"].shift(-1)
    feature_rows = []
    for column in derived.columns:
        row = dependency_row(
            "B1_derived_vs_next_price",
            column,
            "price_t_plus_1",
            derived[column],
            target_next_price,
            seed,
            "development_train",
        )
        row["partial_corr_controlling_price_t"] = partial_corr(
            derived[column], target_next_price, discovery[["price"]]
        )
        feature_rows.append(row)
    feature_dependence = pd.DataFrame(feature_rows)

    dy = discovery["dBTC_y"]
    return_rows = []
    for lag in (0, 1):
        x = dy.shift(lag)
        for target_name, target in (
            ("price_t", discovery["price"]),
            ("delta_price_t", discovery["price"].diff()),
            ("return_t", discovery["return"]),
        ):
            return_rows.append(
                dependency_row(
                    "D_dBTC_y_relations",
                    f"dBTC_y_lag_{lag}",
                    target_name,
                    x,
                    target,
                    seed,
                    "development_train",
                )
            )
    return_dependence = pd.DataFrame(return_rows)

    lag_rows = []
    for lag in range(cfg.max_lag + 1):
        lag_rows.append(
            dependency_row(
                "E_return_lag_scan",
                f"dBTC_y_lag_{lag}",
                "return_t",
                dy.shift(lag),
                discovery["return"],
                seed,
                "development_train",
            )
        )
        lag_rows[-1]["lag"] = lag
        lag_rows[-1]["is_predictive"] = lag >= 1
    lag_scan = pd.DataFrame(lag_rows)

    candidate_lag = 4 if cfg.max_lag >= 4 else cfg.max_lag
    x_lag = dy.shift(candidate_lag)
    decomposition_rows: list[dict[str, object]] = []
    direction_frame = finite_frame({"x": x_lag, "direction": discovery["direction"]})
    auc = float("nan")
    if direction_frame["direction"].nunique() == 2:
        raw_auc = roc_auc_score(direction_frame["direction"], direction_frame["x"])
        auc = float(max(raw_auc, 1.0 - raw_auc))
    decomposition_rows.append(
        {
            "experiment": "F1_direction",
            "model": "dependency_lag_decomposition",
            "feature_set": f"dBTC_y_lag_{candidate_lag}",
            "target": "return_direction",
            "sample": "development_train",
            "n": len(direction_frame),
            "mutual_information": mi_classification(
                x_lag, discovery["direction"], seed
            ),
            "auc_orientation_free": auc,
        }
    )
    magnitude = dependency_row(
        "F2_magnitude",
        f"dBTC_y_lag_{candidate_lag}",
        "abs_return",
        x_lag,
        discovery["abs_return"],
        seed,
        "development_train",
    )
    decomposition_rows.append(magnitude)
    for quantile in (0.90, 0.95, 0.975, 0.99):
        threshold = float(discovery["abs_return"].quantile(quantile))
        tail = (discovery["abs_return"] >= threshold).astype(int)
        decomposition_rows.append(
            {
                "experiment": "F3_tails",
                "model": "dependency_lag_decomposition",
                "feature_set": f"dBTC_y_lag_{candidate_lag}",
                "target": f"abs_return_top_{100 * (1 - quantile):g}pct",
                "sample": "development_train",
                "n": int(finite_frame({"x": x_lag, "y": tail}).shape[0]),
                "threshold_fitted_on_train": threshold,
                "mutual_information": mi_classification(x_lag, tail, seed),
            }
        )

    hetero = finite_frame(
        {"x": x_lag, "return": discovery["return"], "sigma": discovery["sigma_pre_return"]}
    )
    hetero["group"] = pd.qcut(hetero["x"], q=10, labels=False, duplicates="drop")
    groups = [part["return"].to_numpy() for _, part in hetero.groupby("group")]
    statistic, pvalue = levene(*groups, center="median") if len(groups) >= 2 else (np.nan, np.nan)
    group_std = hetero.groupby("group")["return"].std()
    standardized = hetero["return"] / hetero["sigma"].clip(lower=cfg.eps)
    decomposition_rows.append(
        {
            "experiment": "F4_conditional_variance",
            "model": "levene_deciles",
            "feature_set": f"dBTC_y_lag_{candidate_lag}",
            "target": "return_variance",
            "sample": "development_train",
            "n": len(hetero),
            "levene_statistic": float(statistic),
            "levene_pvalue": float(pvalue),
            "minimum_group_std": float(group_std.min()),
            "maximum_group_std": float(group_std.max()),
            "std_ratio_max_min": float(group_std.max() / group_std.min()),
            "mi_raw_return": mi_regression(hetero["x"], hetero["return"], seed),
            "mi_volatility_standardized_return": mi_regression(
                hetero["x"], standardized, seed
            ),
        }
    )
    lag_decomposition = pd.DataFrame(decomposition_rows)

    vol_rows = []
    for horizon in cfg.volatility_horizons:
        target = future_rms(discovery["return"], horizon)
        for lag in range(cfg.max_lag + 1):
            row = dependency_row(
                "G_future_volatility_scan",
                f"dBTC_y_lag_{lag}",
                f"future_rms_h{horizon}",
                dy.shift(lag),
                target,
                seed,
                "development_train",
            )
            row["horizon"] = horizon
            row["lag"] = lag
            row["is_predictive"] = True
            vol_rows.append(row)
    future_vol_scan = pd.DataFrame(vol_rows)
    midpoint = len(discovery) // 2
    stability_rows = []
    for half_name, half in (
        ("first_half", discovery.iloc[:midpoint]),
        ("second_half", discovery.iloc[midpoint:]),
    ):
        stability_rows.append(
            dependency_row(
                "G_stability_halves",
                "dBTC_y_lag_0",
                f"future_rms_h{cfg.future_horizon}",
                half["dBTC_y"],
                future_rms(half["return"], cfg.future_horizon),
                seed,
                half_name,
            )
        )
    stability = pd.DataFrame(stability_rows)

    outputs = {
        "01_dependencia_basica.csv": basic_dependencies,
        "02_features_derivadas_vs_precio.csv": feature_dependence,
        "03_dBTC_y_vs_precio_retorno.csv": return_dependence,
        "04_barrido_lags_retorno.csv": lag_scan,
        "05_descomposicion_lag_4.csv": lag_decomposition,
        "06_barrido_volatilidad_futura.csv": future_vol_scan,
        "06b_estabilidad_mitades.csv": stability,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    return outputs


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    experiment: str,
    model: str,
    feature_set: str,
    split: str,
    n_train: int,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model": model,
        "model_family": "regression",
        "feature_set": feature_set,
        "split": split,
        "n_train": n_train,
        "n_eval": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def run_regression_holdout(
    frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    del frame
    y_train = train["sigma_future"].to_numpy()
    y_test = test["sigma_future"].to_numpy()
    specs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    specs.append(
        (
            "persistence",
            "sigma_past",
            train["sigma_past"].to_numpy(),
            test["sigma_past"].to_numpy(),
        )
    )
    linear_base = LinearRegression().fit(train[["sigma_past"]], y_train)
    specs.append(
        (
            "linear_volatility",
            "sigma_past",
            linear_base.predict(train[["sigma_past"]]),
            linear_base.predict(test[["sigma_past"]]),
        )
    )
    linear_plus = LinearRegression().fit(train[["sigma_past", "dBTC_y"]], y_train)
    specs.append(
        (
            "linear_volatility_plus_topology",
            "sigma_past+dBTC_y",
            linear_plus.predict(train[["sigma_past", "dBTC_y"]]),
            linear_plus.predict(test[["sigma_past", "dBTC_y"]]),
        )
    )
    hgb_base = HistGradientBoostingRegressor(
        learning_rate=cfg.boosting_learning_rate,
        max_iter=cfg.boosting_iterations,
        max_leaf_nodes=cfg.boosting_leaves,
        max_depth=cfg.boosting_depth,
        random_state=cfg.seed,
    ).fit(train[["sigma_past"]], y_train)
    specs.append(
        (
            "histgb_volatility",
            "sigma_past",
            hgb_base.predict(train[["sigma_past"]]),
            hgb_base.predict(test[["sigma_past"]]),
        )
    )
    hgb_plus = HistGradientBoostingRegressor(
        learning_rate=cfg.boosting_learning_rate,
        max_iter=cfg.boosting_iterations,
        max_leaf_nodes=cfg.boosting_leaves,
        max_depth=cfg.boosting_depth,
        random_state=cfg.seed,
    ).fit(train[["sigma_past", "dBTC_y"]], y_train)
    specs.append(
        (
            "histgb_volatility_plus_topology",
            "sigma_past+dBTC_y",
            hgb_plus.predict(train[["sigma_past", "dBTC_y"]]),
            hgb_plus.predict(test[["sigma_past", "dBTC_y"]]),
        )
    )
    rows = []
    for model, features, train_prediction, test_prediction in specs:
        rows.extend(
            [
                regression_metrics(
                    y_train,
                    train_prediction,
                    "H_regression_holdout_train",
                    model,
                    features,
                    "holdout_train",
                    len(train),
                ),
                regression_metrics(
                    y_test,
                    test_prediction,
                    "H_regression_holdout",
                    model,
                    features,
                    "holdout",
                    len(train),
                ),
            ]
        )
    return pd.DataFrame(rows)


def run_regression_walk_forward(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.run_walk_forward:
        return pd.DataFrame()
    splitter = TimeSeriesSplit(n_splits=cfg.folds, gap=cfg.future_horizon)
    rows = []
    for fold, (train_idx, validation_idx) in enumerate(splitter.split(frame), start=1):
        train, validation = frame.iloc[train_idx], frame.iloc[validation_idx]
        fold_metrics = run_regression_holdout(frame, train, validation, cfg)
        is_train = fold_metrics["split"].eq("holdout_train")
        fold_metrics["experiment"] = np.where(
            is_train, "H_regression_walk_forward_train", "H_regression_walk_forward"
        )
        fold_metrics["split"] = np.where(is_train, f"fold_{fold}_train", f"fold_{fold}")
        fold_metrics["fold"] = fold
        rows.append(fold_metrics)
    return pd.concat(rows, ignore_index=True)


def regime_thresholds(values: pd.Series, scheme: str, cfg: Config) -> np.ndarray:
    if scheme == "3_regimes":
        probabilities = (1.0 / 3.0, 2.0 / 3.0)
    elif scheme == "5_extreme_4_5":
        middle = (1.0 - 2.0 * cfg.extreme_tail) / 3.0
        probabilities = (
            cfg.extreme_tail,
            cfg.extreme_tail + middle,
            cfg.extreme_tail + 2.0 * middle,
            1.0 - cfg.extreme_tail,
        )
    elif scheme == "5_quintiles":
        probabilities = (0.2, 0.4, 0.6, 0.8)
    else:
        raise KeyError(scheme)
    thresholds = values.quantile(probabilities).to_numpy(dtype=np.float64)
    if not np.all(np.diff(thresholds) > 0):
        raise ValueError(f"Los thresholds de {scheme} no son estrictamente crecientes.")
    return thresholds


def assign_regime(values: pd.Series, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(values.to_numpy(dtype=np.float64), thresholds, right=False)


def choose_backend(cfg: Config) -> str:
    if cfg.backend == "histgb":
        return "histgb"
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        if cfg.backend == "lightgbm":
            raise RuntimeError(
                "LightGBM no está instalado. Ejecutá `pip install lightgbm` o usá --backend histgb."
            )
        warnings.warn("LightGBM no está instalado; se usará HistGradientBoosting.", stacklevel=2)
        return "histgb"
    return "lightgbm"


def advanced_classifier(cfg: Config, backend: str) -> BaseEstimator:
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=cfg.boosting_iterations,
            learning_rate=cfg.boosting_learning_rate,
            num_leaves=cfg.boosting_leaves,
            max_depth=cfg.boosting_depth,
            random_state=cfg.seed,
            n_jobs=-1,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=cfg.boosting_learning_rate,
        max_iter=cfg.boosting_iterations,
        max_leaf_nodes=cfg.boosting_leaves,
        max_depth=cfg.boosting_depth,
        random_state=cfg.seed,
    )


def model_specs(
    topology_columns: Sequence[str],
    volatility_columns: Sequence[str],
    interaction_columns: Sequence[str],
    cfg: Config,
    backend: str,
) -> list[tuple[str, str, list[str], Callable[[], BaseEstimator]]]:
    simple_tree = lambda: DecisionTreeClassifier(
        max_depth=cfg.tree_depth,
        min_samples_leaf=cfg.tree_min_leaf,
        random_state=cfg.seed,
    )
    simple_topology = ["BTC_y", "dBTC_y"]
    simple_volatility = ["sigma_past"]
    return [
        (
            "dummy_most_frequent",
            "dummy",
            simple_volatility,
            lambda: DummyClassifier(strategy="most_frequent"),
        ),
        ("tree_volatility", "simple_volatility", simple_volatility, simple_tree),
        ("tree_topology", "simple_topology", simple_topology, simple_tree),
        (
            "tree_volatility_plus_topology",
            "simple_combined",
            simple_volatility + simple_topology,
            simple_tree,
        ),
        (
            f"{backend}_volatility_multiscale",
            "advanced_volatility",
            list(volatility_columns),
            lambda: advanced_classifier(cfg, backend),
        ),
        (
            f"{backend}_topology_multiscale",
            "advanced_topology",
            list(topology_columns),
            lambda: advanced_classifier(cfg, backend),
        ),
        (
            f"{backend}_volatility_plus_topology",
            "advanced_combined",
            list(volatility_columns) + list(topology_columns) + list(interaction_columns),
            lambda: advanced_classifier(cfg, backend),
        ),
    ]


def classification_metric_row(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    experiment: str,
    model: str,
    feature_set: str,
    split: str,
    scheme: str,
    backend: str,
    n_train: int,
    seed: int,
) -> dict[str, object]:
    n_classes = len(CLASS_NAMES[scheme])
    prediction = probabilities.argmax(axis=1)
    return {
        "experiment": experiment,
        "model": model,
        "model_family": "classification",
        "feature_set": feature_set,
        "backend": backend,
        "regime_scheme": scheme,
        "split": split,
        "seed": seed,
        "n_train": n_train,
        "n_eval": len(y_true),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probabilities, labels=np.arange(n_classes)),
    }


def aligned_probabilities(
    model: BaseEstimator, raw: np.ndarray, n_classes: int
) -> np.ndarray:
    classes = np.asarray(getattr(model, "classes_"), dtype=int)
    aligned = np.full((len(raw), n_classes), 1e-15, dtype=np.float64)
    aligned[:, classes] = raw
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def fit_classification_suite(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    scheme: str,
    split: str,
    specs: Sequence[tuple[str, str, list[str], Callable[[], BaseEstimator]]],
    cfg: Config,
    backend: str,
    collect_details: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    thresholds = regime_thresholds(train["sigma_future"], scheme, cfg)
    y_train = assign_regime(train["sigma_future"], thresholds)
    y_eval = assign_regime(evaluation["sigma_future"], thresholds)
    n_classes = len(CLASS_NAMES[scheme])
    metric_rows, train_metric_rows = [], []
    class_rows, confusion_rows, importance_rows = [], [], []

    for model_name, feature_set, columns, factory in specs:
        model = factory()
        model.fit(train[columns], y_train)
        train_probability = aligned_probabilities(
            model, model.predict_proba(train[columns]), n_classes
        )
        train_metric_rows.append(
            classification_metric_row(
                y_train,
                train_probability,
                "I_J_regime_classification_train",
                model_name,
                feature_set,
                f"{split}_train",
                scheme,
                backend if "advanced" in feature_set else "decision_tree_or_dummy",
                len(train),
                cfg.seed,
            )
        )
        raw_probability = model.predict_proba(evaluation[columns])
        probabilities = aligned_probabilities(model, raw_probability, n_classes)
        prediction = probabilities.argmax(axis=1)
        metric_rows.append(
            classification_metric_row(
                y_eval,
                probabilities,
                "I_J_regime_classification",
                model_name,
                feature_set,
                split,
                scheme,
                backend if "advanced" in feature_set else "decision_tree_or_dummy",
                len(train),
                cfg.seed,
            )
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            y_eval, prediction, labels=np.arange(n_classes), zero_division=0
        )
        for class_id, class_name in enumerate(CLASS_NAMES[scheme]):
            class_rows.append(
                {
                    "experiment": "I_J_regime_classification_by_class",
                    "model": model_name,
                    "feature_set": feature_set,
                    "regime_scheme": scheme,
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": precision[class_id],
                    "recall": recall[class_id],
                    "f1": f1[class_id],
                    "support": int(support[class_id]),
                }
            )
        matrix = confusion_matrix(y_eval, prediction, labels=np.arange(n_classes))
        for true_id in range(n_classes):
            for predicted_id in range(n_classes):
                confusion_rows.append(
                    {
                        "experiment": "I_J_confusion_matrix",
                        "model": model_name,
                        "feature_set": feature_set,
                        "regime_scheme": scheme,
                        "split": split,
                        "true_class": CLASS_NAMES[scheme][true_id],
                        "predicted_class": CLASS_NAMES[scheme][predicted_id],
                        "count": int(matrix[true_id, predicted_id]),
                    }
                )

        if collect_details and "advanced" in feature_set:
            if backend == "lightgbm" and hasattr(model, "booster_"):
                gains = model.booster_.feature_importance(importance_type="gain")
                method = "lightgbm_gain"
            else:
                result = permutation_importance(
                    model,
                    evaluation[columns],
                    y_eval,
                    scoring="f1_macro",
                    n_repeats=cfg.permutation_repeats,
                    random_state=cfg.seed,
                    n_jobs=-1,
                )
                gains = result.importances_mean
                method = "holdout_permutation_macro_f1"
            denominator = float(np.abs(gains).sum())
            normalized = gains / denominator if denominator > 0 else np.zeros_like(gains)
            for column, raw_gain, share in zip(columns, gains, normalized):
                family = (
                    "interaction"
                    if column.startswith("sigma_x_")
                    else "volatility"
                    if column.startswith("sigma")
                    else "topology"
                )
                importance_rows.append(
                    {
                        "experiment": "J_feature_importance",
                        "model": model_name,
                        "feature_set": feature_set,
                        "regime_scheme": scheme,
                        "split": split,
                        "feature": column,
                        "feature_family": family,
                        "importance_method": method,
                        "importance_raw": float(raw_gain),
                        "importance_share": float(share),
                    }
                )

    threshold_rows = [
        {
            "experiment": "regime_thresholds",
            "model": "train_quantiles",
            "feature_set": "sigma_future",
            "regime_scheme": scheme,
            "split": split,
            "threshold_order": index + 1,
            "threshold_value": float(value),
            "fitted_on": "train_only",
        }
        for index, value in enumerate(thresholds)
    ]
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(train_metric_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(importance_rows),
        pd.DataFrame(threshold_rows),
    )


def classification_experiments(
    model_frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: Sequence[tuple[str, str, list[str], Callable[[], BaseEstimator]]],
    cfg: Config,
    backend: str,
) -> dict[str, pd.DataFrame]:
    holdout_parts = [[], [], [], [], [], []]
    for scheme in CLASS_NAMES:
        result = fit_classification_suite(
            train, test, scheme, "holdout", specs, cfg, backend, collect_details=True
        )
        for destination, frame in zip(holdout_parts, result):
            if not frame.empty:
                destination.append(frame)
    holdout_outputs = [
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for parts in holdout_parts
    ]

    walk_metrics, walk_train_metrics, walk_classes, walk_thresholds = [], [], [], []
    if cfg.run_walk_forward:
        development = model_frame.loc[: train.index.max()].copy()
        splitter = TimeSeriesSplit(n_splits=cfg.folds, gap=cfg.future_horizon)
        for fold, (train_idx, validation_idx) in enumerate(splitter.split(development), start=1):
            fold_train = development.iloc[train_idx]
            validation = development.iloc[validation_idx]
            for scheme in CLASS_NAMES:
                metrics, train_metrics, classes, _, _, thresholds = fit_classification_suite(
                    fold_train,
                    validation,
                    scheme,
                    f"fold_{fold}",
                    specs,
                    cfg,
                    backend,
                    collect_details=False,
                )
                metrics["fold"] = fold
                train_metrics["fold"] = fold
                classes["fold"] = fold
                thresholds["fold"] = fold
                walk_metrics.append(metrics)
                walk_train_metrics.append(train_metrics)
                walk_classes.append(classes)
                walk_thresholds.append(thresholds)

    return {
        "09_clasificacion_holdout.csv": holdout_outputs[0],
        "09c_clasificacion_train.csv": holdout_outputs[1],
        "10_clasificacion_por_clase_holdout.csv": holdout_outputs[2],
        "10b_matrices_confusion_holdout.csv": holdout_outputs[3],
        "11_importancia_features_holdout.csv": holdout_outputs[4],
        "12_thresholds_regimen.csv": pd.concat(
            [holdout_outputs[5]] + walk_thresholds, ignore_index=True
        ),
        "09b_clasificacion_walk_forward.csv": (
            pd.concat(walk_metrics, ignore_index=True) if walk_metrics else pd.DataFrame()
        ),
        "09d_clasificacion_walk_forward_train.csv": (
            pd.concat(walk_train_metrics, ignore_index=True)
            if walk_train_metrics
            else pd.DataFrame()
        ),
        "10c_clasificacion_por_clase_walk_forward.csv": (
            pd.concat(walk_classes, ignore_index=True) if walk_classes else pd.DataFrame()
        ),
    }


def topology_utility(metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("tree_topology", "dummy_most_frequent", "intrinsic_simple_topology_vs_dummy"),
        (
            "tree_volatility_plus_topology",
            "tree_volatility",
            "incremental_simple_topology",
        ),
        (
            "lightgbm_volatility_plus_topology",
            "lightgbm_volatility_multiscale",
            "incremental_advanced_topology",
        ),
        (
            "histgb_volatility_plus_topology",
            "histgb_volatility_multiscale",
            "incremental_advanced_topology",
        ),
    )
    key_columns = ["regime_scheme", "split"]
    rows = []
    indexed = metrics.set_index(key_columns + ["model"])
    keys = metrics[key_columns].drop_duplicates().itertuples(index=False, name=None)
    for key in keys:
        for candidate, baseline, contrast in pairs:
            candidate_key = (*key, candidate)
            baseline_key = (*key, baseline)
            if candidate_key not in indexed.index or baseline_key not in indexed.index:
                continue
            cand = indexed.loc[candidate_key]
            base = indexed.loc[baseline_key]
            if isinstance(cand, pd.DataFrame) or isinstance(base, pd.DataFrame):
                raise AssertionError("Hay modelos duplicados dentro del mismo split.")
            rows.append(
                {
                    "experiment": "K_topology_utility",
                    "model": candidate,
                    "baseline_model": baseline,
                    "feature_set": cand["feature_set"],
                    "contrast": contrast,
                    "regime_scheme": key[0],
                    "split": key[1],
                    "gain_accuracy": cand["accuracy"] - base["accuracy"],
                    "gain_balanced_accuracy": (
                        cand["balanced_accuracy"] - base["balanced_accuracy"]
                    ),
                    "gain_macro_f1": cand["macro_f1"] - base["macro_f1"],
                    "gain_log_loss": base["log_loss"] - cand["log_loss"],
                    "relative_gain_macro_f1": (
                        (cand["macro_f1"] / base["macro_f1"] - 1.0)
                        if base["macro_f1"] != 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_plots(
    exploratory: dict[str, pd.DataFrame],
    classification: pd.DataFrame,
    utility: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib no está instalado; no se crearán gráficos.", stacklevel=2)
        return []
    paths: list[str] = []
    scan = exploratory["06_barrido_volatilidad_futura.csv"]
    pivot = scan.pivot(index="lag", columns="horizon", values="mutual_information")
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(pivot.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_yticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_xlabel("Lag de dBTC_y (barras de 10 minutos)")
    ax.set_ylabel("Horizonte de volatilidad futura")
    ax.set_title("Información mutua: dBTC_y → volatilidad futura (train)")
    fig.colorbar(image, ax=ax, label="Mutual information")
    fig.tight_layout()
    path = output_dir / "fig_01_mi_lag_horizonte.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    holdout = classification[classification["split"] == "holdout"].copy()
    if not holdout.empty:
        labels = holdout["model"] + "\n" + holdout["regime_scheme"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(np.arange(len(holdout)), holdout["macro_f1"])
        ax.set_xticks(np.arange(len(holdout)), labels=labels, rotation=75, ha="right")
        ax.set_ylabel("Macro-F1")
        ax.set_title("Clasificación de regímenes: holdout cronológico")
        fig.tight_layout()
        path = output_dir / "fig_02_macro_f1_holdout.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    incremental = utility[utility["contrast"].str.startswith("incremental", na=False)]
    incremental = incremental[incremental["split"] == "holdout"]
    if not incremental.empty:
        labels = incremental["contrast"] + "\n" + incremental["regime_scheme"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(len(incremental)), incremental["gain_macro_f1"])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(np.arange(len(incremental)), labels=labels, rotation=55, ha="right")
        ax.set_ylabel("Δ Macro-F1 (modelo con topología − control)")
        ax.set_title("Utilidad incremental de la topología informacional")
        fig.tight_layout()
        path = output_dir / "fig_03_utilidad_incremental.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def make_manifest(
    cfg: Config,
    backend: str,
    model_frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    topology_columns: Sequence[str],
    volatility_columns: Sequence[str],
    interaction_columns: Sequence[str],
    artifacts: Iterable[str],
) -> dict[str, object]:
    return {
        "source_chat": SOURCE_CHAT,
        "scientific_scope": (
            "Utilidad incremental de la coordenada BTC_y del simplex como descriptor "
            "de regímenes futuros de volatilidad; no es una prueba causal del NPP completo."
        ),
        "config": asdict(cfg),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "advanced_backend": backend,
        },
        "temporal_contract": {
            "shuffle": False,
            "holdout": "final chronological block",
            "purge_rows": cfg.future_horizon,
            "walk_forward": "expanding TimeSeriesSplit",
            "walk_forward_gap": cfg.future_horizon,
            "threshold_fit": "train only per split",
            "feature_information_set": "<= t",
            "target_information_set": "t+1 ... t+h",
        },
        "samples": {
            "model_frame": len(model_frame),
            "train": len(train),
            "test": len(test),
            "train_start": str(train.index.min()),
            "train_end": str(train.index.max()),
            "test_start": str(test.index.min()),
            "test_end": str(test.index.max()),
        },
        "feature_families": {
            "topology": list(topology_columns),
            "volatility": list(volatility_columns),
            "interactions": list(interaction_columns),
        },
        "artifacts": sorted(set(artifacts)),
    }


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    validate_config(cfg)
    mount_drive_if_requested(args.mount_drive)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    price, topology, input_audit = load_inputs(cfg)
    core = build_core(price, topology, cfg)

    topology_features = build_topology_features(core, cfg)
    volatility_features = build_volatility_features(core, cfg)
    interactions = build_interactions(core, topology_features)
    model_frame = pd.concat(
        [core[["sigma_future"]], topology_features, volatility_features, interactions], axis=1
    )
    model_frame = model_frame.loc[:, ~model_frame.columns.duplicated()]
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(model_frame) < max(800, 5 * cfg.tree_min_leaf):
        raise ValueError(
            f"Sólo hay {len(model_frame)} filas modelables; reducí tree_min_leaf o revisá datos."
        )

    train, test, cut = chronological_holdout(
        model_frame, cfg.holdout_fraction, cfg.future_horizon
    )
    development_end = model_frame.index[cut - 1]
    discovery = core.loc[:development_end].iloc[: -cfg.future_horizon].copy()
    input_audit["rows_model_frame"] = len(model_frame)
    input_audit["rows_train"] = len(train)
    input_audit["rows_purged"] = cfg.future_horizon
    input_audit["rows_test"] = len(test)
    input_audit["shuffle"] = False
    input_audit.to_csv(output_dir / "00_auditoria_datos.csv", index=False)

    exploratory = exploratory_analysis(core, discovery, cfg, output_dir)
    regression_holdout_all = run_regression_holdout(model_frame, train, test, cfg)
    regression_walk_all = run_regression_walk_forward(train, cfg)
    regression_holdout = regression_holdout_all.query("split == 'holdout'").copy()
    regression_train = regression_holdout_all.query("split == 'holdout_train'").copy()
    if regression_walk_all.empty:
        regression_walk = pd.DataFrame()
        regression_walk_train = pd.DataFrame()
    else:
        train_mask = regression_walk_all["split"].str.endswith("_train")
        regression_walk = regression_walk_all.loc[~train_mask].copy()
        regression_walk_train = regression_walk_all.loc[train_mask].copy()
    regression_holdout.to_csv(output_dir / "07_regresion_holdout.csv", index=False)
    regression_train.to_csv(output_dir / "07b_regresion_train.csv", index=False)
    regression_walk.to_csv(output_dir / "08_regresion_walk_forward.csv", index=False)
    regression_walk_train.to_csv(
        output_dir / "08b_regresion_walk_forward_train.csv", index=False
    )

    backend = choose_backend(cfg)
    topology_columns = list(topology_features.columns)
    volatility_columns = list(volatility_features.columns)
    interaction_columns = list(interactions.columns)
    specs = model_specs(topology_columns, volatility_columns, interaction_columns, cfg, backend)
    classification_outputs = classification_experiments(
        model_frame, train, test, specs, cfg, backend
    )
    for name, frame in classification_outputs.items():
        frame.to_csv(output_dir / name, index=False)

    all_classification = pd.concat(
        [
            classification_outputs["09_clasificacion_holdout.csv"],
            classification_outputs["09b_clasificacion_walk_forward.csv"],
        ],
        ignore_index=True,
    )
    utility = topology_utility(all_classification)
    utility.to_csv(output_dir / "13_utilidad_topologia.csv", index=False)
    utility_summary = (
        utility.groupby(["contrast", "regime_scheme"], as_index=False)
        .agg(
            splits=("split", "count"),
            mean_gain_balanced_accuracy=("gain_balanced_accuracy", "mean"),
            mean_gain_macro_f1=("gain_macro_f1", "mean"),
            mean_gain_log_loss=("gain_log_loss", "mean"),
            wins_macro_f1=("gain_macro_f1", lambda values: int((values > 0).sum())),
        )
    )
    utility_summary.insert(0, "experiment", "K_topology_utility_summary")
    utility_summary.insert(1, "model", "paired_nested_contrast")
    utility_summary.insert(2, "feature_set", "topology_increment")
    utility_summary.to_csv(output_dir / "14_resumen_utilidad_topologia.csv", index=False)

    figure_paths = save_plots(
        exploratory,
        classification_outputs["09_clasificacion_holdout.csv"],
        utility,
        output_dir,
    )
    artifacts = [str(path.name) for path in output_dir.iterdir() if path.is_file()]
    artifacts.extend(Path(path).name for path in figure_paths)
    manifest = make_manifest(
        cfg,
        backend,
        model_frame,
        train,
        test,
        topology_columns,
        volatility_columns,
        interaction_columns,
        artifacts,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAnálisis terminado sin shuffle y con holdout purgado.")
    print("Backend avanzado:", backend)
    print("Resultados:", output_dir)
    print("\nResumen de utilidad incremental:")
    print(utility_summary.to_string(index=False))


if __name__ == "__main__":
    main()
