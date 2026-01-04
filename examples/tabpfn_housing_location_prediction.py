"""Use TabPFNRegressor on housing data and query location-like features.

This is *not* training TabPFN from scratch: TabPFN loads pre-trained weights and
adapts to your dataset at `.fit()` time.

For "location as an entry", you need location-related features in your table:
  - California Housing: has `Latitude`/`Longitude` already (best for coordinates)
  - Ames Housing: has neighborhood/zip-like categorical proxies (varies by version)

Usage:
  python examples/tabpfn_housing_location_prediction.py --dataset california --latitude 37.77 --longitude -122.42
  python examples/tabpfn_housing_location_prediction.py --dataset ames --neighborhood "NridgHt"
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor


def load_openml(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    if dataset == "california":
        ds = fetch_openml(data_id=534, as_frame=True, parser="auto")
    elif dataset == "ames":
        ds = fetch_openml(data_id=43928, as_frame=True, parser="auto")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = ds.data
    y = ds.target
    y_arr = np.asarray(y, dtype=np.float32)
    return X, y_arr


def make_query_row_like_training(X_train: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, object] = {}
    for col in X_train.columns:
        series = X_train[col]
        if pd.api.types.is_numeric_dtype(series):
            row[col] = float(series.median())
        else:
            row[col] = series.mode(dropna=True).iloc[0]
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california", "ames"], default="california")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--neighborhood", type=str, default=None)
    args = parser.parse_args()

    X, y = load_openml(args.dataset)
    if args.max_rows is not None and len(X) > args.max_rows:
        X = X.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        y = y[: args.max_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        ignore_pretraining_limits=True,
        inference_precision=torch.float32,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_mae = float(np.mean(np.abs(test_pred - y_test)))
    test_r2 = float(1.0 - np.sum((test_pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    print(f"Test MAE: {test_mae:.4f} | Test R2: {test_r2:.4f}")

    query = make_query_row_like_training(X_train)
    if args.dataset == "california":
        if args.latitude is not None and "Latitude" in query.columns:
            query.loc[0, "Latitude"] = float(args.latitude)
        if args.longitude is not None and "Longitude" in query.columns:
            query.loc[0, "Longitude"] = float(args.longitude)
        if (args.latitude is not None and "Latitude" not in query.columns) or (
            args.longitude is not None and "Longitude" not in query.columns
        ):
            print("Warning: California dataset is expected to have Latitude/Longitude columns, but they were not found.")
    else:
        if args.neighborhood is not None:
            if "Neighborhood" in query.columns:
                query.loc[0, "Neighborhood"] = args.neighborhood
            else:
                print("Warning: `Neighborhood` column not found; Ames versions can differ.")

    pred = float(model.predict(query)[0])
    print("Query row overrides:", {k: v for k, v in query.iloc[0].items() if k in {"Latitude", "Longitude", "Neighborhood"}})
    print("Predicted target:", pred)


if __name__ == "__main__":
    main()

