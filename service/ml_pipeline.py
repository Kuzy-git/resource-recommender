from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import lightgbm as lgb
import tensorflow as tf
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tqdm.auto import tqdm

from service.config import (
    ARTIFACTS_DIR,
    DEFAULT_CONFIG,
    META_CLEAN_PATH,
    METADATA_PATH,
    MODELS_DIR,
    REPORT_DATA_PATH,
    SEQUENCE_SCALER_PATH,
    TABULAR_SCALER_PATH,
    WINDOW_DATA_PATH,
    PipelineConfig,
)


META_COLUMNS = [
    "container_id",
    "machine_id",
    "time_stamp",
    "app_du",
    "status",
    "cpu_request",
    "cpu_limit",
    "mem_size",
]

USAGE_COLUMNS = [
    "container_id",
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "cpi",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io_percent",
]

FEATURE_COLS = [
    "cpu_util_mean",
    "cpu_util_max",
    "cpu_util_std",
    "mem_util_mean",
    "mem_util_max",
    "mem_util_std",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "mem_lag_1",
    "mem_lag_2",
    "mem_lag_3",
    "cpu_roll_mean_3",
    "mem_roll_mean_3",
    "cpu_roll_std_3",
    "mem_roll_std_3",
    "cpu_delta_1",
    "mem_delta_1",
    "cpu_request",
    "cpu_limit",
    "mem_size",
    "cpu_request_limit_ratio",
    "cpu_limit_mem_ratio",
    "cpu_request_mem_ratio",
    "relative_hour",
    "relative_day_cycle",
    "samples_in_window",
]

SEQ_COLS = [
    "cpu_util_mean",
    "cpu_util_max",
    "cpu_util_std",
    "mem_util_mean",
    "mem_util_max",
    "mem_util_std",
    "cpu_request",
    "cpu_limit",
    "mem_size",
    "relative_hour",
    "relative_day_cycle",
]

ENTITY_COLS = ["app_du", "container_id"]
REPORT_KEY_COLS = ["app_du", "container_id", "time_window"]


def _entity_cols(frame: pd.DataFrame) -> list[str]:
    return [column for column in ENTITY_COLS if column in frame.columns]


def _sort_cols(frame: pd.DataFrame) -> list[str]:
    return _entity_cols(frame) + ["time_window"]


@dataclass
class LoadedArtifacts:
    metadata: dict[str, Any]
    cpu_model: Any
    ram_model: Any
    tabular_scaler: StandardScaler
    sequence_scaler: StandardScaler
    window_df: pd.DataFrame
    meta_df: pd.DataFrame
    bootstrap_report: pd.DataFrame

    @property
    def best_cpu_model_name(self) -> str:
        return str(self.metadata["best_cpu_model_name"])

    @property
    def best_ram_model_name(self) -> str:
        return str(self.metadata["best_ram_model_name"])

    @property
    def feature_cols(self) -> list[str]:
        return list(self.metadata["feature_cols"])

    @property
    def seq_cols(self) -> list[str]:
        return list(self.metadata["seq_cols"])


def _native_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append({key: _native_value(value) for key, value in row.items()})
    return records


def _round_metrics(metrics_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    payload = metrics_df.round(6).to_dict(orient="index")
    return {
        outer_key: {inner_key: float(inner_value) for inner_key, inner_value in outer_value.items()}
        for outer_key, outer_value in payload.items()
    }


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def clean_meta_frame(meta: pd.DataFrame) -> pd.DataFrame:
    frame = meta.copy()
    frame = frame.dropna(subset=["container_id", "machine_id", "time_stamp", "cpu_request", "cpu_limit", "mem_size"]).copy()
    frame = frame.drop_duplicates(subset=["container_id", "machine_id", "time_stamp", "app_du", "cpu_request", "cpu_limit", "mem_size"])
    frame = frame[(frame["cpu_request"] > 0) & (frame["cpu_limit"] > 0) & (frame["mem_size"] > 0)].copy()
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    frame = frame.dropna(subset=["time_stamp"]).copy()
    frame["time_stamp"] = frame["time_stamp"].astype(np.int64)
    frame["container_id"] = frame["container_id"].astype(str)
    frame["machine_id"] = frame["machine_id"].astype(str)
    frame["app_du"] = frame["app_du"].fillna("unknown").astype(str)
    frame["status"] = frame["status"].fillna("unknown").astype(str)
    return frame.reset_index(drop=True)


def clean_usage_frame(usage: pd.DataFrame) -> pd.DataFrame:
    frame = usage.copy()
    frame = frame.dropna(subset=["container_id", "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent"]).copy()
    frame = frame.drop_duplicates(subset=["container_id", "machine_id", "time_stamp"])
    frame = frame[
        frame["cpu_util_percent"].between(0, 100)
        & frame["mem_util_percent"].between(0, 100)
    ].copy()
    frame["time_stamp"] = pd.to_numeric(frame["time_stamp"], errors="coerce")
    frame = frame.dropna(subset=["time_stamp"]).copy()
    frame["time_stamp"] = frame["time_stamp"].astype(np.int64)
    frame["container_id"] = frame["container_id"].astype(str)
    frame["machine_id"] = frame["machine_id"].astype(str)
    return frame.reset_index(drop=True)


def read_training_data(config: PipelineConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(config.meta_path, header=None, names=META_COLUMNS, nrows=config.meta_nrows)
    usage = pd.read_csv(config.usage_path, header=None, names=USAGE_COLUMNS, nrows=config.usage_nrows)
    return clean_meta_frame(meta), clean_usage_frame(usage)


def aggregate_usage_to_windows(raw_df: pd.DataFrame, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    frame = raw_df.copy()
    if "app_du" not in frame.columns:
        frame["app_du"] = "unknown"
    frame["app_du"] = frame["app_du"].fillna("unknown").astype(str)

    horizon_seconds = config.horizon_minutes * 60
    frame["time_window"] = (frame["time_stamp"] // horizon_seconds) * horizon_seconds

    window_df = (
        frame.groupby(["app_du", "container_id", "time_window"], as_index=False)
        .agg(
            machine_id=("machine_id", "last"),
            cpu_util_mean=("cpu_util_percent", "mean"),
            cpu_util_max=("cpu_util_percent", "max"),
            cpu_util_std=("cpu_util_percent", "std"),
            mem_util_mean=("mem_util_percent", "mean"),
            mem_util_max=("mem_util_percent", "max"),
            mem_util_std=("mem_util_percent", "std"),
            cpu_request=("cpu_request", "last"),
            cpu_limit=("cpu_limit", "last"),
            mem_size=("mem_size", "last"),
            samples_in_window=("cpu_util_percent", "size"),
        )
    )

    window_df["cpu_util_std"] = window_df["cpu_util_std"].fillna(0)
    window_df["mem_util_std"] = window_df["mem_util_std"].fillna(0)
    return window_df.sort_values(["app_du", "container_id", "time_window"]).reset_index(drop=True)


def enrich_window_features(
    window_df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
    include_targets: bool = True,
) -> pd.DataFrame:
    frame = window_df.copy()
    if "app_du" not in frame.columns:
        frame["app_du"] = "unknown"

    group_cols = _entity_cols(frame)
    frame = frame.sort_values(group_cols + ["time_window"]).reset_index(drop=True)

    frame["relative_hour"] = ((frame["time_window"] // 3600) % 24).astype(int)
    frame["relative_day_cycle"] = ((frame["time_window"] // 86400) % 7).astype(int)

    for idx in range(1, config.lags + 1):
        frame[f"cpu_lag_{idx}"] = frame.groupby(group_cols)["cpu_util_mean"].shift(idx)
        frame[f"mem_lag_{idx}"] = frame.groupby(group_cols)["mem_util_mean"].shift(idx)

    frame["cpu_roll_mean_3"] = frame.groupby(group_cols)["cpu_util_mean"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    frame["mem_roll_mean_3"] = frame.groupby(group_cols)["mem_util_mean"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    frame["cpu_roll_std_3"] = frame.groupby(group_cols)["cpu_util_mean"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).std()
    )
    frame["mem_roll_std_3"] = frame.groupby(group_cols)["mem_util_mean"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).std()
    )
    frame["cpu_delta_1"] = frame.groupby(group_cols)["cpu_util_mean"].transform(lambda series: series.diff().shift(1))
    frame["mem_delta_1"] = frame.groupby(group_cols)["mem_util_mean"].transform(lambda series: series.diff().shift(1))
    frame["cpu_request_limit_ratio"] = frame["cpu_request"] / frame["cpu_limit"]
    frame["cpu_limit_mem_ratio"] = frame["cpu_limit"] / frame["mem_size"]
    frame["cpu_request_mem_ratio"] = frame["cpu_request"] / frame["mem_size"]

    if include_targets:
        frame["cpu_target"] = frame.groupby(group_cols)["cpu_util_mean"].shift(-1)
        frame["mem_target"] = frame.groupby(group_cols)["mem_util_mean"].shift(-1)

    return frame.replace([np.inf, -np.inf], np.nan)


def build_window_frame(
    meta: pd.DataFrame,
    usage: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_for_merge = meta[["container_id", "machine_id", "time_stamp", "app_du", "cpu_request", "cpu_limit", "mem_size"]].copy()
    meta_for_merge = meta_for_merge.sort_values(["time_stamp", "container_id", "machine_id"]).reset_index(drop=True)
    usage_for_merge = usage.sort_values(["time_stamp", "container_id", "machine_id"]).reset_index(drop=True)

    df_raw = pd.merge_asof(
        usage_for_merge,
        meta_for_merge,
        on="time_stamp",
        by=["container_id", "machine_id"],
        direction="backward",
    )

    df_raw = df_raw.dropna(subset=["cpu_request", "cpu_limit", "mem_size"]).copy()
    df_raw["app_du"] = df_raw["app_du"].fillna("unknown").astype(str)
    df_raw = df_raw.sort_values(["app_du", "container_id", "time_stamp"]).reset_index(drop=True)

    window_df = aggregate_usage_to_windows(df_raw, config=config)
    window_df = enrich_window_features(window_df, config=config, include_targets=True)
    return df_raw, window_df


def build_model_frame(window_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [column for column in ["app_du", "container_id", "machine_id", "time_window"] if column in window_df.columns]
    frame = window_df[id_cols + FEATURE_COLS + ["cpu_target", "mem_target"]].copy()
    frame = frame.dropna(subset=FEATURE_COLS + ["cpu_target", "mem_target"]).copy()
    frame = frame.sort_values([column for column in ["time_window", "app_du", "container_id"] if column in frame.columns]).reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    return frame


def build_sequences(frame: pd.DataFrame, seq_cols: list[str], lookback_value: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_cols = _entity_cols(frame)
    ordered = frame.sort_values(group_cols + ["time_window"]).reset_index(drop=True)

    x_seq: list[np.ndarray] = []
    y_cpu: list[float] = []
    y_mem: list[float] = []
    row_ids: list[int] = []

    for _, group in ordered.groupby(group_cols, sort=False):
        values = group[seq_cols].to_numpy(dtype=np.float32)
        cpu_targets = group["cpu_target"].to_numpy(dtype=np.float32)
        mem_targets = group["mem_target"].to_numpy(dtype=np.float32)
        ids = group["row_id"].to_numpy()

        if len(group) < lookback_value:
            continue

        for idx in range(lookback_value - 1, len(group)):
            x_seq.append(values[idx - lookback_value + 1 : idx + 1])
            y_cpu.append(cpu_targets[idx])
            y_mem.append(mem_targets[idx])
            row_ids.append(int(ids[idx]))

    x_seq_array = np.array(x_seq, dtype=np.float32)
    y_cpu_array = np.array(y_cpu, dtype=np.float32)
    y_mem_array = np.array(y_mem, dtype=np.float32)
    row_ids_array = np.array(row_ids)

    order = np.argsort(row_ids_array)
    return x_seq_array[order], y_cpu_array[order], y_mem_array[order], row_ids_array[order]


def build_latest_sequences(frame: pd.DataFrame, seq_cols: list[str], lookback_value: int) -> tuple[np.ndarray, pd.DataFrame]:
    group_cols = _entity_cols(frame)
    ordered = frame.sort_values(group_cols + ["time_window"]).reset_index(drop=True)

    sequences: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for _, group in ordered.groupby(group_cols, sort=False):
        if len(group) < lookback_value:
            continue

        sequence = group[seq_cols].tail(lookback_value).to_numpy(dtype=np.float32)
        row_cols = [column for column in REPORT_KEY_COLS + ["cpu_request", "cpu_limit", "mem_size"] if column in group.columns]
        last_row = group.iloc[-1][row_cols].to_dict()
        sequences.append(sequence)
        rows.append(last_row)

    return np.array(sequences, dtype=np.float32), pd.DataFrame(rows)


def build_lstm(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def generate_action(current_value: float, recommended_value: float, down_threshold: float, up_threshold: float) -> str:
    if recommended_value < current_value * down_threshold:
        return "DOWNSCALE"
    if recommended_value > current_value * up_threshold:
        return "UPSCALE"
    return "OK"


def combine_decision(cpu_action: str, ram_action: str) -> str:
    if "UPSCALE" in {cpu_action, ram_action}:
        return "UPSCALE"
    if "DOWNSCALE" in {cpu_action, ram_action}:
        return "DOWNSCALE"
    return "OK"


def decision_label(decision: str) -> str:
    mapping = {"UPSCALE": "увеличить", "DOWNSCALE": "уменьшить", "OK": "не изменять"}
    return mapping.get(decision, "не изменять")


def _fit_final_model(
    model_name: str,
    target_values: np.ndarray,
    x_full_scaled_df: pd.DataFrame,
    x_seq_full_scaled: np.ndarray,
    xgb_estimator: Any,
    lgb_estimator: Any,
) -> Any:
    if model_name == "LSTM":
        model = build_lstm((x_seq_full_scaled.shape[1], x_seq_full_scaled.shape[2]))
        model.fit(
            x_seq_full_scaled,
            target_values,
            epochs=15,
            batch_size=512,
            validation_split=0.1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            shuffle=False,
            verbose=0,
        )
        return model

    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "XGBoost":
        model = clone(xgb_estimator)
    else:
        model = clone(lgb_estimator)

    model.fit(x_full_scaled_df, target_values)
    return model


def _importance_items(feature_names: list[str], values: np.ndarray, top_n: int = 20) -> list[dict[str, float | str]]:
    scores = np.nan_to_num(np.abs(np.asarray(values, dtype=float)), nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0 or float(scores.sum()) <= 0:
        return []

    order = np.argsort(scores)[::-1][:top_n]
    total = float(scores.sum())
    return [
        {
            "feature": str(feature_names[index]),
            "importance": float(scores[index] / total),
            "raw_importance": float(scores[index]),
        }
        for index in order
    ]


def _tabular_feature_influence(
    model_name: str,
    model: Any,
    feature_names: list[str],
    x_tabular_scaled_df: pd.DataFrame,
    sample_size: int = 4096,
) -> tuple[list[dict[str, float | str]], str]:
    if not x_tabular_scaled_df.empty:
        sample_count = min(sample_size, len(x_tabular_scaled_df))
        sample_indices = np.linspace(0, len(x_tabular_scaled_df) - 1, sample_count, dtype=int)
        x_sample = x_tabular_scaled_df.iloc[sample_indices]

        try:
            if model_name == "LightGBM":
                contributions = np.asarray(model.predict(x_sample, pred_contrib=True), dtype=float)
                return _importance_items(feature_names, np.mean(np.abs(contributions[:, : len(feature_names)]), axis=0)), "mean_abs_shap_values"

            if model_name == "XGBoost":
                dmatrix = xgb.DMatrix(x_sample, feature_names=feature_names)
                contributions = np.asarray(model.get_booster().predict(dmatrix, pred_contribs=True), dtype=float)
                return _importance_items(feature_names, np.mean(np.abs(contributions[:, : len(feature_names)]), axis=0)), "mean_abs_shap_values"
        except Exception:
            pass

    if hasattr(model, "feature_importances_"):
        return _importance_items(feature_names, np.asarray(model.feature_importances_, dtype=float)), "model_feature_importance"

    if hasattr(model, "coef_"):
        return _importance_items(feature_names, np.asarray(model.coef_, dtype=float).reshape(-1)), "absolute_coefficients"

    return [], "not_available"


def _sequence_permutation_influence(
    model: Any,
    x_seq: np.ndarray,
    target_values: np.ndarray,
    feature_names: list[str],
    sample_size: int = 2048,
    seed: int = 42,
) -> list[dict[str, float | str]]:
    if x_seq.size == 0 or len(x_seq) == 0:
        return []

    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, len(x_seq))
    sample_indices = np.linspace(0, len(x_seq) - 1, sample_count, dtype=int)
    x_sample = x_seq[sample_indices].copy()
    y_sample = np.asarray(target_values, dtype=float)[sample_indices]

    baseline_pred = model.predict(x_sample, verbose=0).reshape(-1)
    baseline_mae = mean_absolute_error(y_sample, baseline_pred)
    importances: list[float] = []

    for feature_index in range(x_sample.shape[-1]):
        x_permuted = x_sample.copy()
        flat_values = x_permuted[:, :, feature_index].reshape(-1).copy()
        rng.shuffle(flat_values)
        x_permuted[:, :, feature_index] = flat_values.reshape(x_permuted[:, :, feature_index].shape)
        permuted_pred = model.predict(x_permuted, verbose=0).reshape(-1)
        permuted_mae = mean_absolute_error(y_sample, permuted_pred)
        importances.append(max(float(permuted_mae - baseline_mae), 0.0))

    return _importance_items(feature_names, np.asarray(importances, dtype=float))


def _model_feature_influence(
    model_name: str,
    model: Any,
    x_tabular_scaled_df: pd.DataFrame,
    x_seq_scaled: np.ndarray,
    target_values: np.ndarray,
    target_name: str,
    seed: int,
) -> dict[str, Any]:
    if model_name == "LSTM":
        items = _sequence_permutation_influence(
            model=model,
            x_seq=x_seq_scaled,
            target_values=target_values,
            feature_names=SEQ_COLS,
            seed=seed,
        )
        method = "permutation_importance"
        source_features = "seq_cols"
    else:
        items, method = _tabular_feature_influence(model_name, model, FEATURE_COLS, x_tabular_scaled_df)
        source_features = "feature_cols"

    return {
        "target": target_name,
        "model_name": model_name,
        "method": method,
        "source_features": source_features,
        "items": items,
    }


def _save_model(model: Any, model_name: str, output_stem: str) -> str:
    if model_name == "LSTM":
        file_name = f"{output_stem}.keras"
        model.save(MODELS_DIR / file_name)
        return file_name

    file_name = f"{output_stem}.joblib"
    joblib.dump(model, MODELS_DIR / file_name)
    return file_name


def _load_model(file_name: str, model_name: str) -> Any:
    model_path = MODELS_DIR / file_name
    if model_name == "LSTM":
        return tf.keras.models.load_model(model_path, compile=False)
    return joblib.load(model_path)


def train_and_save_artifacts(config: PipelineConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    _ensure_dirs()

    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    meta, usage = read_training_data(config)
    _, window_df = build_window_frame(meta, usage, config=config)
    df_model = build_model_frame(window_df)

    if df_model.empty:
        raise ValueError("Не удалось подготовить обучающую выборку: df_model пуст.")

    q_train = df_model["time_window"].quantile(0.70)
    q_val = df_model["time_window"].quantile(0.85)

    train_df = df_model[df_model["time_window"] < q_train].copy()
    val_df = df_model[(df_model["time_window"] >= q_train) & (df_model["time_window"] < q_val)].copy()
    test_df = df_model[df_model["time_window"] >= q_val].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Недостаточно данных для разбиения на train/val/test.")

    print(
        f"[training] prepared windows: total={len(df_model)}, "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    progress = tqdm(total=13, desc="ML training", unit="stage")

    x_train_df = train_df[FEATURE_COLS].copy()
    x_val_df = val_df[FEATURE_COLS].copy()
    x_test_df = test_df[FEATURE_COLS].copy()

    y_cpu_train = train_df["cpu_target"].to_numpy()
    y_mem_train = train_df["mem_target"].to_numpy()

    tabular_scaler = StandardScaler()
    x_train_scaled_df = pd.DataFrame(tabular_scaler.fit_transform(x_train_df), columns=FEATURE_COLS, index=train_df.index)
    x_val_scaled_df = pd.DataFrame(tabular_scaler.transform(x_val_df), columns=FEATURE_COLS, index=val_df.index)
    x_test_scaled_df = pd.DataFrame(tabular_scaler.transform(x_test_df), columns=FEATURE_COLS, index=test_df.index)
    progress.set_description("ML training | data prepared")
    progress.update(1)

    gs_sample_size = min(config.grid_sample_size, len(x_train_scaled_df))
    gs_idx = np.linspace(0, len(x_train_scaled_df) - 1, num=gs_sample_size, dtype=int)
    x_train_gs_df = x_train_scaled_df.iloc[gs_idx].copy()
    y_cpu_train_gs = train_df.iloc[gs_idx]["cpu_target"].to_numpy()
    y_mem_train_gs = train_df.iloc[gs_idx]["mem_target"].to_numpy()

    n_splits = 3 if len(x_train_gs_df) >= 12 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb_param_grid = {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
    lgb_param_grid = {"n_estimators": [100, 200], "max_depth": [4, 6, -1], "learning_rate": [0.05, 0.1]}

    xgb_grid_cpu = GridSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.random_seed,
            tree_method="hist",
            n_jobs=config.training_n_jobs,
        ),
        xgb_param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=config.training_n_jobs,
    )
    progress.set_description("ML training | tune XGBoost CPU")
    xgb_grid_cpu.fit(x_train_gs_df, y_cpu_train_gs)
    progress.update(1)

    xgb_grid_mem = GridSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=config.random_seed,
            tree_method="hist",
            n_jobs=config.training_n_jobs,
        ),
        xgb_param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=config.training_n_jobs,
    )
    progress.set_description("ML training | tune XGBoost RAM")
    xgb_grid_mem.fit(x_train_gs_df, y_mem_train_gs)
    progress.update(1)

    lgb_grid_cpu = GridSearchCV(
        lgb.LGBMRegressor(
            random_state=config.random_seed,
            verbosity=-1,
            n_jobs=config.training_n_jobs,
        ),
        lgb_param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=config.training_n_jobs,
    )
    progress.set_description("ML training | tune LightGBM CPU")
    lgb_grid_cpu.fit(x_train_gs_df, y_cpu_train_gs)
    progress.update(1)

    lgb_grid_mem = GridSearchCV(
        lgb.LGBMRegressor(
            random_state=config.random_seed,
            verbosity=-1,
            n_jobs=config.training_n_jobs,
        ),
        lgb_param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=config.training_n_jobs,
    )
    progress.set_description("ML training | tune LightGBM RAM")
    lgb_grid_mem.fit(x_train_gs_df, y_mem_train_gs)
    progress.update(1)

    x_seq_all, y_seq_cpu_all, y_seq_mem_all, seq_row_ids_all = build_sequences(df_model, SEQ_COLS, config.lookback)
    train_row_ids = train_df["row_id"].to_numpy()
    val_row_ids = val_df["row_id"].to_numpy()
    test_row_ids = test_df["row_id"].to_numpy()

    seq_train_mask = np.isin(seq_row_ids_all, train_row_ids)
    seq_val_mask = np.isin(seq_row_ids_all, val_row_ids)
    seq_test_mask = np.isin(seq_row_ids_all, test_row_ids)

    x_seq_train = x_seq_all[seq_train_mask]
    x_seq_val = x_seq_all[seq_val_mask]
    x_seq_test = x_seq_all[seq_test_mask]

    if len(x_seq_train) == 0 or len(x_seq_val) == 0 or len(x_seq_test) == 0:
        raise ValueError("Недостаточно окон для LSTM-последовательностей.")

    y_seq_cpu_train = y_seq_cpu_all[seq_train_mask]
    y_seq_cpu_val = y_seq_cpu_all[seq_val_mask]
    y_seq_cpu_test = y_seq_cpu_all[seq_test_mask]
    y_seq_mem_train = y_seq_mem_all[seq_train_mask]
    y_seq_mem_val = y_seq_mem_all[seq_val_mask]
    y_seq_mem_test = y_seq_mem_all[seq_test_mask]

    sequence_scaler = StandardScaler()
    sequence_scaler.fit(x_seq_train.reshape(-1, x_seq_train.shape[-1]))
    x_seq_train_scaled = sequence_scaler.transform(x_seq_train.reshape(-1, x_seq_train.shape[-1])).reshape(x_seq_train.shape)
    x_seq_val_scaled = sequence_scaler.transform(x_seq_val.reshape(-1, x_seq_val.shape[-1])).reshape(x_seq_val.shape)
    x_seq_test_scaled = sequence_scaler.transform(x_seq_test.reshape(-1, x_seq_test.shape[-1])).reshape(x_seq_test.shape)

    common_val_row_ids = np.sort(seq_row_ids_all[seq_val_mask])
    common_test_row_ids = np.sort(seq_row_ids_all[seq_test_mask])
    val_common_df = df_model[df_model["row_id"].isin(common_val_row_ids)].sort_values("row_id").copy()
    test_common_df = df_model[df_model["row_id"].isin(common_test_row_ids)].sort_values("row_id").copy()

    x_val_common_scaled_df = pd.DataFrame(tabular_scaler.transform(val_common_df[FEATURE_COLS]), columns=FEATURE_COLS, index=val_common_df.index)
    x_test_common_scaled_df = pd.DataFrame(tabular_scaler.transform(test_common_df[FEATURE_COLS]), columns=FEATURE_COLS, index=test_common_df.index)

    y_cpu_val_common = val_common_df["cpu_target"].to_numpy()
    y_mem_val_common = val_common_df["mem_target"].to_numpy()
    y_cpu_test_common = test_common_df["cpu_target"].to_numpy()
    y_mem_test_common = test_common_df["mem_target"].to_numpy()

    base_models = {
        "LinearRegression": (LinearRegression(), LinearRegression()),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=2,
                random_state=config.random_seed,
                n_jobs=config.training_n_jobs,
            ),
            RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=2,
                random_state=config.random_seed,
                n_jobs=config.training_n_jobs,
            ),
        ),
        "XGBoost": (clone(xgb_grid_cpu.best_estimator_), clone(xgb_grid_mem.best_estimator_)),
        "LightGBM": (clone(lgb_grid_cpu.best_estimator_), clone(lgb_grid_mem.best_estimator_)),
    }

    results: dict[str, dict[str, float]] = {}
    for model_name, (model_cpu, model_mem) in base_models.items():
        progress.set_description(f"ML training | eval {model_name}")
        model_cpu.fit(x_train_scaled_df, y_cpu_train)
        pred_cpu_val = np.clip(model_cpu.predict(x_val_common_scaled_df), 0, 100)
        pred_cpu_test = np.clip(model_cpu.predict(x_test_common_scaled_df), 0, 100)

        model_mem.fit(x_train_scaled_df, y_mem_train)
        pred_mem_val = np.clip(model_mem.predict(x_val_common_scaled_df), 0, 100)
        pred_mem_test = np.clip(model_mem.predict(x_test_common_scaled_df), 0, 100)

        results[model_name] = {
            "CPU_MAE_val": float(mean_absolute_error(y_cpu_val_common, pred_cpu_val)),
            "CPU_RMSE_val": float(np.sqrt(mean_squared_error(y_cpu_val_common, pred_cpu_val))),
            "CPU_R2_val": float(r2_score(y_cpu_val_common, pred_cpu_val)),
            "RAM_MAE_val": float(mean_absolute_error(y_mem_val_common, pred_mem_val)),
            "RAM_RMSE_val": float(np.sqrt(mean_squared_error(y_mem_val_common, pred_mem_val))),
            "RAM_R2_val": float(r2_score(y_mem_val_common, pred_mem_val)),
            "CPU_MAE_test": float(mean_absolute_error(y_cpu_test_common, pred_cpu_test)),
            "CPU_RMSE_test": float(np.sqrt(mean_squared_error(y_cpu_test_common, pred_cpu_test))),
            "CPU_R2_test": float(r2_score(y_cpu_test_common, pred_cpu_test)),
            "RAM_MAE_test": float(mean_absolute_error(y_mem_test_common, pred_mem_test)),
            "RAM_RMSE_test": float(np.sqrt(mean_squared_error(y_mem_test_common, pred_mem_test))),
            "RAM_R2_test": float(r2_score(y_mem_test_common, pred_mem_test)),
        }
        progress.update(1)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lstm_cpu = build_lstm((x_seq_train_scaled.shape[1], x_seq_train_scaled.shape[2]))
    progress.set_description("ML training | train LSTM CPU")
    lstm_cpu.fit(
        x_seq_train_scaled,
        y_seq_cpu_train,
        epochs=15,
        batch_size=512,
        validation_data=(x_seq_val_scaled, y_seq_cpu_val),
        callbacks=[early_stop],
        shuffle=False,
        verbose=0,
    )
    pred_lstm_cpu_val = np.clip(lstm_cpu.predict(x_seq_val_scaled, verbose=0).flatten(), 0, 100)
    pred_lstm_cpu_test = np.clip(lstm_cpu.predict(x_seq_test_scaled, verbose=0).flatten(), 0, 100)
    progress.update(1)

    lstm_mem = build_lstm((x_seq_train_scaled.shape[1], x_seq_train_scaled.shape[2]))
    progress.set_description("ML training | train LSTM RAM")
    lstm_mem.fit(
        x_seq_train_scaled,
        y_seq_mem_train,
        epochs=15,
        batch_size=512,
        validation_data=(x_seq_val_scaled, y_seq_mem_val),
        callbacks=[early_stop],
        shuffle=False,
        verbose=0,
    )
    pred_lstm_mem_val = np.clip(lstm_mem.predict(x_seq_val_scaled, verbose=0).flatten(), 0, 100)
    pred_lstm_mem_test = np.clip(lstm_mem.predict(x_seq_test_scaled, verbose=0).flatten(), 0, 100)
    progress.update(1)

    results["LSTM"] = {
        "CPU_MAE_val": float(mean_absolute_error(y_seq_cpu_val, pred_lstm_cpu_val)),
        "CPU_RMSE_val": float(np.sqrt(mean_squared_error(y_seq_cpu_val, pred_lstm_cpu_val))),
        "CPU_R2_val": float(r2_score(y_seq_cpu_val, pred_lstm_cpu_val)),
        "RAM_MAE_val": float(mean_absolute_error(y_seq_mem_val, pred_lstm_mem_val)),
        "RAM_RMSE_val": float(np.sqrt(mean_squared_error(y_seq_mem_val, pred_lstm_mem_val))),
        "RAM_R2_val": float(r2_score(y_seq_mem_val, pred_lstm_mem_val)),
        "CPU_MAE_test": float(mean_absolute_error(y_seq_cpu_test, pred_lstm_cpu_test)),
        "CPU_RMSE_test": float(np.sqrt(mean_squared_error(y_seq_cpu_test, pred_lstm_cpu_test))),
        "CPU_R2_test": float(r2_score(y_seq_cpu_test, pred_lstm_cpu_test)),
        "RAM_MAE_test": float(mean_absolute_error(y_seq_mem_test, pred_lstm_mem_test)),
        "RAM_RMSE_test": float(np.sqrt(mean_squared_error(y_seq_mem_test, pred_lstm_mem_test))),
        "RAM_R2_test": float(r2_score(y_seq_mem_test, pred_lstm_mem_test)),
    }

    metrics_df = pd.DataFrame(results).T[
        [
            "CPU_MAE_val",
            "CPU_RMSE_val",
            "CPU_R2_val",
            "RAM_MAE_val",
            "RAM_RMSE_val",
            "RAM_R2_val",
            "CPU_MAE_test",
            "CPU_RMSE_test",
            "CPU_R2_test",
            "RAM_MAE_test",
            "RAM_RMSE_test",
            "RAM_R2_test",
        ]
    ].sort_values(["CPU_MAE_val", "RAM_MAE_val"])

    best_cpu_model_name = min(results.keys(), key=lambda key: results[key]["CPU_MAE_val"])
    best_ram_model_name = min(results.keys(), key=lambda key: results[key]["RAM_MAE_val"])

    x_full_df = df_model[FEATURE_COLS].copy()
    y_cpu_full = df_model["cpu_target"].to_numpy()
    y_mem_full = df_model["mem_target"].to_numpy()

    tabular_scaler_full = StandardScaler()
    x_full_scaled_df = pd.DataFrame(tabular_scaler_full.fit_transform(x_full_df), columns=FEATURE_COLS, index=df_model.index)

    x_seq_full, y_seq_cpu_full, y_seq_mem_full, _ = build_sequences(df_model, SEQ_COLS, config.lookback)
    sequence_scaler_full = StandardScaler()
    sequence_scaler_full.fit(x_seq_full.reshape(-1, x_seq_full.shape[-1]))
    x_seq_full_scaled = sequence_scaler_full.transform(x_seq_full.reshape(-1, x_seq_full.shape[-1])).reshape(x_seq_full.shape)

    progress.set_description("ML training | fit final CPU model")
    final_cpu_model = _fit_final_model(
        best_cpu_model_name,
        y_seq_cpu_full if best_cpu_model_name == "LSTM" else y_cpu_full,
        x_full_scaled_df,
        x_seq_full_scaled,
        xgb_grid_cpu.best_estimator_,
        lgb_grid_cpu.best_estimator_,
    )
    progress.update(1)

    progress.set_description("ML training | fit final RAM model")
    final_ram_model = _fit_final_model(
        best_ram_model_name,
        y_seq_mem_full if best_ram_model_name == "LSTM" else y_mem_full,
        x_full_scaled_df,
        x_seq_full_scaled,
        xgb_grid_mem.best_estimator_,
        lgb_grid_mem.best_estimator_,
    )
    progress.update(1)

    feature_influence = {
        "cpu": _model_feature_influence(
            model_name=best_cpu_model_name,
            model=final_cpu_model,
            x_tabular_scaled_df=x_full_scaled_df,
            x_seq_scaled=x_seq_full_scaled,
            target_values=y_seq_cpu_full,
            target_name="cpu_target",
            seed=config.random_seed,
        ),
        "ram": _model_feature_influence(
            model_name=best_ram_model_name,
            model=final_ram_model,
            x_tabular_scaled_df=x_full_scaled_df,
            x_seq_scaled=x_seq_full_scaled,
            target_values=y_seq_mem_full,
            target_name="mem_target",
            seed=config.random_seed + 1,
        ),
    }

    scoring_cols = list(set(FEATURE_COLS + SEQ_COLS + ["app_du", "container_id", "machine_id", "time_window", "cpu_request", "cpu_limit", "mem_size"]))
    df_scoring = df_model[scoring_cols].copy()
    df_scoring = df_scoring.dropna(subset=list(set(FEATURE_COLS + SEQ_COLS))).copy()
    df_scoring = df_scoring.sort_values(["app_du", "container_id", "time_window"]).reset_index(drop=True)

    latest_tabular_df = (
        df_scoring.sort_values(["app_du", "container_id", "time_window"])
        .groupby(["app_du", "container_id"], as_index=False)
        .tail(1)
        .sort_values(["app_du", "container_id", "time_window"])
        .copy()
    )

    x_latest_tabular_scaled_df = pd.DataFrame(
        tabular_scaler_full.transform(latest_tabular_df[FEATURE_COLS]),
        columns=FEATURE_COLS,
        index=latest_tabular_df.index,
    )

    latest_seq_x, latest_seq_rows = build_latest_sequences(df_scoring, SEQ_COLS, config.lookback)
    latest_seq_x_scaled = sequence_scaler_full.transform(latest_seq_x.reshape(-1, latest_seq_x.shape[-1])).reshape(latest_seq_x.shape)

    if best_cpu_model_name == "LSTM":
        cpu_pred_latest = latest_seq_rows[REPORT_KEY_COLS].copy()
        cpu_pred_latest["predicted_cpu_percent"] = np.clip(final_cpu_model.predict(latest_seq_x_scaled, verbose=0).flatten(), 0, 100)
    else:
        cpu_pred_latest = latest_tabular_df[REPORT_KEY_COLS].copy()
        cpu_pred_latest["predicted_cpu_percent"] = np.clip(final_cpu_model.predict(x_latest_tabular_scaled_df), 0, 100)

    if best_ram_model_name == "LSTM":
        ram_pred_latest = latest_seq_rows[REPORT_KEY_COLS].copy()
        ram_pred_latest["predicted_ram_percent"] = np.clip(final_ram_model.predict(latest_seq_x_scaled, verbose=0).flatten(), 0, 100)
    else:
        ram_pred_latest = latest_tabular_df[REPORT_KEY_COLS].copy()
        ram_pred_latest["predicted_ram_percent"] = np.clip(final_ram_model.predict(x_latest_tabular_scaled_df), 0, 100)

    report_base = latest_tabular_df[REPORT_KEY_COLS + ["machine_id", "cpu_request", "cpu_limit", "mem_size"]].copy()
    report = report_base.merge(cpu_pred_latest, on=REPORT_KEY_COLS, how="left").merge(ram_pred_latest, on=REPORT_KEY_COLS, how="left")
    report = report.dropna(subset=["predicted_cpu_percent", "predicted_ram_percent"]).copy()

    report["predicted_cpu_absolute"] = (report["predicted_cpu_percent"] / 100.0) * report["cpu_limit"]
    report["predicted_ram_absolute"] = (report["predicted_ram_percent"] / 100.0) * report["mem_size"]
    report["recommended_cpu_limit"] = np.maximum(np.ceil(report["predicted_cpu_absolute"] * config.cpu_safety_margin), 1)
    report["recommended_mem_size"] = np.maximum(np.round(report["predicted_ram_absolute"] * config.ram_safety_margin, 3), 0.001)
    report["cpu_action"] = [
        generate_action(current, recommended, config.down_threshold, config.up_threshold)
        for current, recommended in zip(report["cpu_limit"], report["recommended_cpu_limit"])
    ]
    report["ram_action"] = [
        generate_action(current, recommended, config.down_threshold, config.up_threshold)
        for current, recommended in zip(report["mem_size"], report["recommended_mem_size"])
    ]
    report["cpu_difference"] = report["recommended_cpu_limit"] - report["cpu_limit"]
    report["ram_difference"] = report["recommended_mem_size"] - report["mem_size"]
    report["decision"] = [combine_decision(cpu_action, ram_action) for cpu_action, ram_action in zip(report["cpu_action"], report["ram_action"])]
    report["decision_label"] = report["decision"].map(decision_label)
    report = report[
        [
            "app_du",
            "container_id",
            "machine_id",
            "time_window",
            "cpu_request",
            "cpu_limit",
            "mem_size",
            "predicted_cpu_percent",
            "predicted_cpu_absolute",
            "recommended_cpu_limit",
            "cpu_action",
            "cpu_difference",
            "predicted_ram_percent",
            "predicted_ram_absolute",
            "recommended_mem_size",
            "ram_action",
            "ram_difference",
            "decision",
            "decision_label",
        ]
    ].copy()

    cpu_model_file = _save_model(final_cpu_model, best_cpu_model_name, "cpu_model")
    ram_model_file = _save_model(final_ram_model, best_ram_model_name, "ram_model")
    joblib.dump(tabular_scaler_full, TABULAR_SCALER_PATH)
    joblib.dump(sequence_scaler_full, SEQUENCE_SCALER_PATH)

    meta.to_csv(META_CLEAN_PATH, index=False)
    window_df.to_csv(WINDOW_DATA_PATH, index=False)
    report.to_csv(REPORT_DATA_PATH, index=False)

    metadata = {
        "service_name": config.service_name,
        "model_version": config.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": config.as_dict(),
        "feature_cols": FEATURE_COLS,
        "seq_cols": SEQ_COLS,
        "entity_cols": ENTITY_COLS,
        "report_key_cols": REPORT_KEY_COLS,
        "best_cpu_model_name": best_cpu_model_name,
        "best_ram_model_name": best_ram_model_name,
        "cpu_model_file": cpu_model_file,
        "ram_model_file": ram_model_file,
        "feature_influence": feature_influence,
        "feature_influence_version": 2,
        "tabular_scaler_file": TABULAR_SCALER_PATH.name,
        "sequence_scaler_file": SEQUENCE_SCALER_PATH.name,
        "metrics": _round_metrics(metrics_df),
        "best_params": {
            "xgb_cpu": xgb_grid_cpu.best_params_,
            "xgb_ram": xgb_grid_mem.best_params_,
            "lgb_cpu": lgb_grid_cpu.best_params_,
            "lgb_ram": lgb_grid_mem.best_params_,
        },
        "training_split": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "artifacts": {
            "meta_clean_file": META_CLEAN_PATH.name,
            "window_df_file": WINDOW_DATA_PATH.name,
            "bootstrap_report_file": REPORT_DATA_PATH.name,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.set_description("ML training | done")
    progress.close()
    print(
        "[training] done. "
        f"best CPU model={best_cpu_model_name}, best RAM model={best_ram_model_name}"
    )
    return metadata


def load_artifacts() -> LoadedArtifacts:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    cpu_model = _load_model(metadata["cpu_model_file"], metadata["best_cpu_model_name"])
    ram_model = _load_model(metadata["ram_model_file"], metadata["best_ram_model_name"])
    tabular_scaler = joblib.load(TABULAR_SCALER_PATH)
    sequence_scaler = joblib.load(SEQUENCE_SCALER_PATH)
    window_df = pd.read_csv(WINDOW_DATA_PATH)
    meta_df = pd.read_csv(META_CLEAN_PATH)
    bootstrap_report = pd.read_csv(REPORT_DATA_PATH)
    return LoadedArtifacts(
        metadata=metadata,
        cpu_model=cpu_model,
        ram_model=ram_model,
        tabular_scaler=tabular_scaler,
        sequence_scaler=sequence_scaler,
        window_df=window_df,
        meta_df=meta_df,
        bootstrap_report=bootstrap_report,
    )


def build_request_window_frame(
    meta_payload: dict[str, Any],
    usage_payload: list[dict[str, Any]],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    usage_df = pd.DataFrame(usage_payload)
    usage_df["container_id"] = meta_payload["container_id"]
    usage_df["machine_id"] = meta_payload["machine_id"]

    for column_name in USAGE_COLUMNS:
        if column_name not in usage_df.columns:
            usage_df[column_name] = np.nan

    usage_df = usage_df[USAGE_COLUMNS].copy()
    usage_df = clean_usage_frame(usage_df)
    if usage_df.empty:
        raise ValueError("После очистки входных метрик не осталось валидных наблюдений.")

    raw_df = usage_df.copy()
    raw_df["app_du"] = str(meta_payload.get("app_du") or "unknown")
    raw_df["cpu_request"] = float(meta_payload["cpu_request"])
    raw_df["cpu_limit"] = float(meta_payload["cpu_limit"])
    raw_df["mem_size"] = float(meta_payload["mem_size"])

    window_df = aggregate_usage_to_windows(raw_df, config=config)
    return enrich_window_features(window_df, config=config, include_targets=False)


def build_recommendation_response(
    artifacts: LoadedArtifacts,
    meta_payload: dict[str, Any],
    usage_payload: list[dict[str, Any]],
    include_features: bool,
    include_window_series: bool,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    request_window_df = build_request_window_frame(meta_payload, usage_payload, config=config)
    feature_ready_df = request_window_df.dropna(subset=artifacts.feature_cols).sort_values(["app_du", "container_id", "time_window"]).copy()

    if feature_ready_df.empty:
        raise ValueError(f"Недостаточно данных для формирования признаков. Нужно как минимум {config.lookback} полноценных окон.")

    latest_row = feature_ready_df.iloc[-1]
    latest_features = pd.DataFrame([latest_row[artifacts.feature_cols].to_dict()], columns=artifacts.feature_cols)
    latest_features_scaled = pd.DataFrame(artifacts.tabular_scaler.transform(latest_features), columns=artifacts.feature_cols)

    latest_sequence_scaled = None
    if artifacts.best_cpu_model_name == "LSTM" or artifacts.best_ram_model_name == "LSTM":
        if len(feature_ready_df) < config.lookback:
            raise ValueError(f"Для LSTM нужно минимум {config.lookback} окон после агрегации, сейчас доступно {len(feature_ready_df)}.")

        latest_sequence = feature_ready_df.tail(config.lookback)[artifacts.seq_cols].to_numpy(dtype=np.float32)
        latest_sequence_scaled = artifacts.sequence_scaler.transform(latest_sequence).reshape(1, config.lookback, len(artifacts.seq_cols))

    if artifacts.best_cpu_model_name == "LSTM":
        predicted_cpu_percent = float(np.clip(artifacts.cpu_model.predict(latest_sequence_scaled, verbose=0).flatten()[0], 0, 100))
    else:
        predicted_cpu_percent = float(np.clip(artifacts.cpu_model.predict(latest_features_scaled)[0], 0, 100))

    if artifacts.best_ram_model_name == "LSTM":
        predicted_ram_percent = float(np.clip(artifacts.ram_model.predict(latest_sequence_scaled, verbose=0).flatten()[0], 0, 100))
    else:
        predicted_ram_percent = float(np.clip(artifacts.ram_model.predict(latest_features_scaled)[0], 0, 100))

    current_cpu_limit = float(latest_row["cpu_limit"])
    current_mem_size = float(latest_row["mem_size"])
    current_cpu_request = float(latest_row["cpu_request"])

    predicted_cpu_absolute = float((predicted_cpu_percent / 100.0) * current_cpu_limit)
    predicted_ram_absolute = float((predicted_ram_percent / 100.0) * current_mem_size)
    recommended_cpu_limit = float(max(np.ceil(predicted_cpu_absolute * config.cpu_safety_margin), 1))
    recommended_mem_size = float(max(np.round(predicted_ram_absolute * config.ram_safety_margin, 3), 0.001))

    cpu_action = generate_action(current_cpu_limit, recommended_cpu_limit, config.down_threshold, config.up_threshold)
    ram_action = generate_action(current_mem_size, recommended_mem_size, config.down_threshold, config.up_threshold)
    decision = combine_decision(cpu_action, ram_action)
    app_du = str(meta_payload.get("app_du") or "unknown")

    response: dict[str, Any] = {
        "request_id": f"{app_du}-{meta_payload['container_id']}-{int(latest_row['time_window'])}",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "app_du": app_du,
        "container_id": meta_payload["container_id"],
        "machine_id": meta_payload["machine_id"],
        "time_window": {
            "start": int(latest_row["time_window"]),
            "end": int(latest_row["time_window"] + config.horizon_minutes * 60),
            "horizon_minutes": config.horizon_minutes,
        },
        "input_summary": {
            "samples_received": len(usage_payload),
            "windows_built": int(len(request_window_df)),
            "usable_windows": int(len(feature_ready_df)),
        },
        "current_resources": {
            "cpu_request": current_cpu_request,
            "cpu_limit": current_cpu_limit,
            "mem_size": current_mem_size,
        },
        "prediction": {
            "cpu_percent": predicted_cpu_percent,
            "cpu_absolute": predicted_cpu_absolute,
            "ram_percent": predicted_ram_percent,
            "ram_absolute": predicted_ram_absolute,
        },
        "recommendation": {"cpu_limit": recommended_cpu_limit, "mem_size": recommended_mem_size},
        "actions": {"cpu": cpu_action, "ram": ram_action},
        "deltas": {
            "cpu_difference": float(recommended_cpu_limit - current_cpu_limit),
            "ram_difference": float(recommended_mem_size - current_mem_size),
        },
        "decision": decision,
        "decision_label": decision_label(decision),
        "model_names": {"cpu": artifacts.best_cpu_model_name, "ram": artifacts.best_ram_model_name},
    }

    if include_features:
        response["features"] = {feature_name: _native_value(latest_row[feature_name]) for feature_name in artifacts.feature_cols}

    if include_window_series:
        series_frame = feature_ready_df[
            [
                "app_du",
                "container_id",
                "time_window",
                "cpu_util_mean",
                "cpu_util_max",
                "mem_util_mean",
                "mem_util_max",
                "cpu_request",
                "cpu_limit",
                "mem_size",
                "samples_in_window",
            ]
        ].tail(max(config.lookback * 3, 20))
        response["window_series"] = dataframe_to_records(series_frame)

    return response
