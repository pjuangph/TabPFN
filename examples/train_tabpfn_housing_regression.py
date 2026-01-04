"""Train/evaluate TabPFNRegressor on housing datasets.

This uses TabPFN's *pretrained* weights and runs `.fit()` on your dataset to adapt
its internal preprocessing/context. This is the "TabPFN way" of training on a
single dataset (it is not training the foundation model from scratch).

For actual weight updates, see `examples/finetune_housing_regressor.py`.

Usage:
  python examples/train_tabpfn_housing_regression.py --dataset california
  python examples/train_tabpfn_housing_regression.py --dataset ames --max-rows 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tabpfn import TabPFNRegressor
from tabpfn.model_loading import save_fitted_tabpfn_model

from housing_regression_utils import (
    append_jsonl,
    compute_regression_metrics,
    default_save_dir,
    load_housing_regression_dataset,
    make_run_prefix,
    set_seeds,
    split_train_test,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california", "ames"], default="california")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=4)
    parser.add_argument(
        "--model-path",
        type=str,
        default="auto",
        help="Use 'auto' for pretrained weights, or 'random'/'scratch' to avoid downloads.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--save-fit",
        action="store_true",
        help="Also saves the fitted estimator to a `.tabpfn_fit` file.",
    )
    args = parser.parse_args()

    set_seeds(args.seed)
    torch.manual_seed(args.seed)

    X_df, y = load_housing_regression_dataset(args.dataset, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = split_train_test(
        X_df, y, test_size=args.test_size, seed=args.seed
    )

    model = TabPFNRegressor(
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        ignore_pretraining_limits=True,
        inference_precision=torch.float32,
        model_path=args.model_path,
    )

    print(f"Dataset: {args.dataset} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Device: {args.device} | n_estimators: {args.n_estimators}")

    model.fit(X_train, y_train)
    pred = np.asarray(model.predict(X_test), dtype=np.float32)
    metrics = compute_regression_metrics(y_test, pred)
    print(f"Test | MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")

    run_prefix = make_run_prefix(args.dataset, model_name="tabpfn", seed=args.seed)
    save_dir = default_save_dir(__file__, args.save_dir)
    metrics_path = save_dir / f"{run_prefix}_metrics.jsonl"
    append_jsonl(
        metrics_path,
        {
            "model": "TabPFNRegressor",
            "dataset": args.dataset,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "test_mse": metrics["mse"],
            "test_mae": metrics["mae"],
            "test_r2": metrics["r2"],
        },
    )

    if args.save_fit:
        fit_path = Path(save_dir) / f"{run_prefix}.tabpfn_fit"
        save_fitted_tabpfn_model(model, fit_path)
        print(f"Saved fitted model: {fit_path}")


if __name__ == "__main__":
    main()
