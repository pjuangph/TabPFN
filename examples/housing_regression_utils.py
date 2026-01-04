from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split


def set_seeds(seed: int) -> None:
    np.random.seed(seed)


def default_save_dir(script_file: str, save_dir: str | None) -> Path:
    base = Path(save_dir) if save_dir else Path(script_file).resolve().parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_run_prefix(dataset: str, model_name: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{dataset}_seed{seed}_{ts}"


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    denom = float(np.sum((y_true - float(y_true.mean())) ** 2))
    r2 = float(1.0 - float(np.sum((y_pred - y_true) ** 2)) / denom) if denom > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def load_housing_regression_dataset(
    dataset: str, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if dataset == "california":
        ds = fetch_california_housing(as_frame=True)
        X = ds.data
        y = np.asarray(ds.target, dtype=np.float32)
    elif dataset == "ames":
        ds = fetch_openml(data_id=43928, as_frame=True, parser="auto")
        X = ds.data
        y = np.asarray(ds.target, dtype=np.float32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        y = y[:max_rows]

    return X, y


def load_openml_regression_dataset_by_id(
    data_id: int, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, np.ndarray, str]:
    ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = ds.data
    y = np.asarray(ds.target, dtype=np.float32)
    name = getattr(ds, "details", {}).get("name", f"openml_{data_id}")

    if max_rows is not None and len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        y = y[:max_rows]

    return X, y, str(name)


def split_train_test(
    X: pd.DataFrame, y: np.ndarray, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, np.asarray(y_train, dtype=np.float32), np.asarray(y_test, dtype=np.float32)
