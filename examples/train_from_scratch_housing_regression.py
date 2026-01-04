"""Train a simple regression model from scratch on housing datasets.

TabPFN (the foundation model) is designed to be *pretrained* on many synthetic tasks.
This repository exposes inference and fine-tuning, but it does not provide an
official "train TabPFN from random initialization on one dataset" workflow.

This example instead shows a straightforward "from scratch" baseline: a small
PyTorch MLP trained on either:
  - California Housing (OpenML 534): includes `Latitude`/`Longitude`
  - Ames Housing (OpenML 43928): includes many categorical "location-like" fields

Usage:
  python examples/train_from_scratch_housing_regression.py --dataset california
  python examples/train_from_scratch_housing_regression.py --dataset ames --max-rows 20000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from housing_regression_utils import (
    append_jsonl,
    compute_regression_metrics,
    default_save_dir,
    load_housing_regression_dataset,
    make_run_prefix,
    set_seeds,
    split_train_test,
)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_data_id: int


DATASETS: dict[str, DatasetSpec] = {
    "california": DatasetSpec(name="california", openml_data_id=534),
    "ames": DatasetSpec(name="ames", openml_data_id=43928),
}


def build_preprocessor(X_df) -> ColumnTransformer:
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="california")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Writes epoch checkpoints and metrics into this directory (default: the script's folder).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    set_seeds(args.seed)

    X_df, y = load_housing_regression_dataset(args.dataset, args.max_rows, args.seed)
    X_train_df, X_test_df, y_train, y_test = split_train_test(
        X_df, y, test_size=args.test_size, seed=args.seed
    )

    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train, dtype=np.float32).reshape(-1, 1))
    y_test_t = torch.tensor(np.asarray(y_test, dtype=np.float32).reshape(-1, 1))

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = build_mlp(X_train_t.shape[1], args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.optim.NAdam
    loss_fn = nn.MSELoss()

    run_prefix = make_run_prefix(args.dataset, model_name="mlp", seed=args.seed)
    save_dir = default_save_dir(__file__, args.save_dir)

    # Persist preprocessing once (needed to reproduce inference).
    preprocessor_path = save_dir / f"{run_prefix}_preprocessor.joblib"
    try:
        import joblib  # type: ignore

        joblib.dump(preprocessor, preprocessor_path)
    except Exception:
        import pickle

        with preprocessor_path.open("wb") as f:
            pickle.dump(preprocessor, f)

    metrics_path = save_dir / f"{run_prefix}_metrics.jsonl"

    def eval_metrics() -> tuple[float, float, float]:
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t.to(args.device)).cpu().numpy().reshape(-1)
        m = compute_regression_metrics(y_test_t.numpy(), preds)
        return m["mse"], m["mae"], m["r2"]

    print(f"Dataset: {args.dataset} | Train: {len(X_train_t)} | Test: {len(X_test_t)}")
    print(f"Input dim after preprocessing: {X_train_t.shape[1]}")
    print(f"Device: {args.device}")
    print(f"Saving to: {save_dir}")
    print(f"Run prefix: {run_prefix}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for xb, yb in progress:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            progress.set_postfix(train_mse=f"{float(loss.detach().cpu()):.4f}")

        mse, mae, r2 = eval_metrics()
        avg_train_loss = running_loss / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d} | train_mse={avg_train_loss:.4f} | "
            f"test_mse={mse:.4f} | test_mae={mae:.4f} | test_r2={r2:.4f}"
        )

        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_mse": avg_train_loss,
                "test_mse": mse,
                "test_mae": mae,
                "test_r2": r2,
            },
        )

        if args.save_every > 0 and (epoch % args.save_every == 0):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": int(X_train_t.shape[1]),
                    "args": vars(args),
                    "metrics": {"train_mse": avg_train_loss, "test_mse": mse, "test_mae": mae, "test_r2": r2},
                },
                save_dir / f"{run_prefix}_checkpoint_epoch_{epoch:03d}.pt",
            )


if __name__ == "__main__":
    main()
