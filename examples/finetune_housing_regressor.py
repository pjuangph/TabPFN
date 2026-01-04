"""Fine-tune TabPFNRegressor on California/Ames housing.

TabPFN itself is a pretrained foundation model; this script fine-tunes the
pretrained weights on a specific dataset (it does not train TabPFN from scratch).

Usage:
  python examples/finetune_housing_regressor.py --dataset california
  python examples/finetune_housing_regressor.py --dataset ames --num-samples 20000
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator


def load_housing(dataset: str, num_samples: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if dataset == "california":
        ds = fetch_openml(data_id=534, as_frame=True, parser="auto")
    elif dataset == "ames":
        ds = fetch_openml(data_id=43928, as_frame=True, parser="auto")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X = ds.data
    y = np.asarray(ds.target, dtype=np.float32)

    if num_samples is not None and len(X) > num_samples:
        X = X.sample(n=num_samples, random_state=seed).reset_index(drop=True)
        y = y[:num_samples]

    return X, y


def evaluate(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[float, float]:
    eval_reg = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_reg.fit(X_train, y_train)
    pred = np.asarray(eval_reg.predict(X_test), dtype=np.float32)

    mae = float(np.mean(np.abs(pred - y_test)))
    denom = float(np.sum((y_test - float(y_test.mean())) ** 2))
    r2 = float(1.0 - (float(np.sum((pred - y_test) ** 2)) / denom)) if denom > 0 else float("nan")
    return mae, r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california", "ames"], default="california")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--context-samples", type=int, default=10_000)
    parser.add_argument(
        "--model-path",
        type=str,
        default="auto",
        help="Use 'auto' for pretrained weights, or 'random'/'scratch' to avoid downloads.",
    )
    args = parser.parse_args()

    X, y = load_housing(args.dataset, args.num_samples, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.valid_ratio, random_state=args.seed
    )

    regressor_config = {
        "device": args.device,
        "random_state": args.seed,
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": args.model_path,
    }
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=False
    )

    # TabPFN lazily initializes the internal torch model; do it explicitly since we
    # access `models_` for fine-tuning.
    regressor._initialize_model_variables()

    if len(regressor.models_) != 1:
        raise ValueError(
            f"Finetuning expects exactly 1 foundation model, got {len(regressor.models_)}."
        )
    model = regressor.models_[0]

    splitter = partial(train_test_split, test_size=args.valid_ratio, random_state=args.seed)
    training_datasets = regressor.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        max_data_size=min(args.context_samples, len(X_train)),
    )
    dataloader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)

    optimizer = Adam(model.parameters(), lr=args.lr)
    eval_config = {
        **regressor_config,
        "inference_config": {"SUBSAMPLE_SAMPLES": args.context_samples},
    }

    print(f"Dataset: {args.dataset} | Train: {len(X_train)} | Test: {len(X_test)} | Device: {args.device}")
    init_mae, init_r2 = evaluate(regressor, eval_config, X_train, y_train, X_test, y_test)
    print(f"Initial eval | MAE: {init_mae:.4f} | R2: {init_r2:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(dataloader, desc=f"Finetune epoch {epoch}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_znorm,
                y_test_znorm,
                cat_ixs,
                confs,
                _raw_space_bardist,
                znorm_space_bardist,
                _,
                _y_test_raw,
            ) = batch

            regressor.znorm_space_bardist_ = znorm_space_bardist[0]
            regressor.fit_from_preprocessed(X_trains_preprocessed, y_trains_znorm, cat_ixs, confs)
            logits, _, _ = regressor.forward(X_tests_preprocessed)

            loss_fn = znorm_space_bardist[0]
            loss = loss_fn(logits, y_test_znorm.to(args.device)).mean()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        mae, r2 = evaluate(regressor, eval_config, X_train, y_train, X_test, y_test)
        print(f"Epoch {epoch} eval | MAE: {mae:.4f} | R2: {r2:.4f}")


if __name__ == "__main__":
    main()
