"""Train TabPFNRegressor weights from scratch on a housing dataset.

This script initializes the TabPFN foundation model with random weights
(`model_path="random:<seed>"`) so **no checkpoint download occurs**, then trains the
weights with gradient descent on a single regression dataset.

This is intentionally for learning/inspection (e.g., encoders + attention behavior).
It is not TabPFN's original pretraining recipe and will likely perform poorly.

Usage:
  python examples/train_tabpfn_from_scratch_housing_regression.py --dataset california
  python examples/train_tabpfn_from_scratch_housing_regression.py --dataset ames --max-rows 20000
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

from housing_regression_utils import (
    append_jsonl,
    default_save_dir,
    load_housing_regression_dataset,
    make_run_prefix,
    split_train_test,
)


def evaluate(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
) -> dict[str, float]:
    eval_reg = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_reg.fit(X_train, y_train)
    pred = np.asarray(eval_reg.predict(X_test), dtype=np.float32)
    y_true = np.asarray(y_test, dtype=np.float32)
    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    denom = float(np.sum((y_true - float(y_true.mean())) ** 2))
    r2 = float(1.0 - float(np.sum((pred - y_true) ** 2)) / denom) if denom > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california", "ames"], default="california")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for training/inference (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--context-samples", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
        torch.backends.cudnn.benchmark = True

    X_df, y = load_housing_regression_dataset(args.dataset, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = split_train_test(
        X_df, y, test_size=args.test_size, seed=args.seed
    )

    run_prefix = make_run_prefix(args.dataset, model_name="tabpfn_scratch", seed=args.seed)
    save_dir = default_save_dir(__file__, args.save_dir)
    metrics_path = save_dir / f"{run_prefix}_metrics.jsonl"

    regressor_config = {
        "device": args.device,
        "random_state": args.seed,
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": f"random:{args.seed}",
    }
    regressor = TabPFNRegressor(**regressor_config, fit_mode="batched", differentiable_input=False)

    # TabPFN lazily initializes internal modules; do it explicitly since we need access
    # to the underlying torch model for weight training.
    regressor._initialize_model_variables()

    if len(regressor.models_) != 1:
        raise ValueError(f"Expected exactly 1 internal model, got {len(regressor.models_)}.")
    model = regressor.models_[0]
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_config = {k: v for k, v in regressor_config.items() if k != "model_path"}

    print(f"Dataset: {args.dataset} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Device: {args.device} | model_path=random:{args.seed}")
    if str(args.device).startswith("cuda"):
        print(f"CUDA device: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"Saving to: {save_dir} | Run prefix: {run_prefix}")

    initial = evaluate(regressor, eval_config, X_train, y_train, X_test, y_test)
    print(f"Initial | MSE: {initial['mse']:.4f} | MAE: {initial['mae']:.4f} | R2: {initial['r2']:.4f}")
    append_jsonl(metrics_path, {"epoch": 0, "event": "eval", **{f"test_{k}": v for k, v in initial.items()}})

    splitter = partial(train_test_split, test_size=args.test_size, random_state=args.seed)
    # Datasets are normalized here
    training_datasets = regressor.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        max_data_size=min(args.context_samples, len(X_train)),
    )   
    dataloader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(
            dataloader,
            desc=f"Scratch train epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []

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
            grad_norm = float(clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            loss_val = float(loss.detach().cpu())
            epoch_losses.append(loss_val)
            epoch_grad_norms.append(grad_norm)
            if args.log_every > 0 and (global_step % args.log_every == 0):
                progress.set_postfix(
                    loss=f"{loss_val:.4f}",
                    grad=f"{grad_norm:.3e}",
                    step=str(global_step),
                )

        if epoch_losses:
            avg_loss = float(np.mean(epoch_losses))
            avg_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else float("nan")
            print(f"Epoch {epoch} | avg_train_loss: {avg_loss:.4f} | avg_grad_norm: {avg_gn:.3e}")
            append_jsonl(
                metrics_path,
                {
                    "epoch": epoch,
                    "event": "train",
                    "avg_train_loss": avg_loss,
                    "avg_grad_norm": avg_gn,
                    "global_step": global_step,
                },
            )

        metrics = evaluate(regressor, eval_config, X_train, y_train, X_test, y_test)
        print(f"Epoch {epoch} | MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
        append_jsonl(metrics_path, {"epoch": epoch, "event": "eval", **{f"test_{k}": v for k, v in metrics.items()}})

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = Path(save_dir) / f"{run_prefix}_checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
