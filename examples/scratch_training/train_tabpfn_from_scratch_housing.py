"""Train TabPFNRegressor weights from scratch on a housing dataset.

This script initializes the TabPFN foundation model with random weights
(`model_path="random:<seed>"`) so **no checkpoint download occurs**, then trains the
weights with gradient descent on one regression dataset.

Important terminology used here:

- `context_size` / `--context-samples`:
  Number of *training rows* included in the TabPFN "prompt" at once.
  TabPFN uses attention over (train + test) rows, so memory grows quickly with
  sequence length. Smaller contexts are the standard way to train/evaluate on large
  datasets.

- `eval_batch_size` / `--eval-batch-size`:
  Number of *test rows* sent through `predict()` per call during evaluation. This is
  unrelated to training mini-batches; it simply limits the test-side sequence length
  in a single forward pass.

- PyTorch `DataLoader(batch_size=1)`:
  This batches *datasets/contexts*, not rows. Each item yielded by
  `regressor.get_preprocessed_datasets(...)` already represents one sampled context.
"""

from __future__ import annotations

import argparse
import platform
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
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
from context_sampling import (
    EvalBatching,
    build_epoch_contexts,
    predict_with_context_batched,
    sample_train_context_indices,
)


def auto_detect_device() -> str:
    system = platform.system()
    if system == "Darwin":
        mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        return "mps" if mps_ok else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate_with_contexts(
    *,
    regressor: TabPFNRegressor,
    init_args: dict,
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    batching: EvalBatching,
) -> dict[str, float]:
    """Evaluate without ever building a (full train + full test) attention context."""
    rng = np.random.default_rng(batching.seed)
    eval_reg = clone_model_for_evaluation(regressor, init_args, TabPFNRegressor)

    n_test = len(X_test)
    sum_pred = np.zeros(n_test, dtype=np.float64)
    used_contexts = 0

    for _ in range(batching.n_contexts):
        idx = sample_train_context_indices(len(X_train), context_size=batching.context_size, rng=rng)
        X_ctx = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        y_ctx = y_train[idx]
        pred = predict_with_context_batched(
            regressor=eval_reg,
            X_ctx=X_ctx,
            y_ctx=y_ctx,
            X_test=X_test,
            test_batch_size=batching.test_batch_size,
        )
        sum_pred += pred.astype(np.float64, copy=False)
        used_contexts += 1

    pred = (sum_pred / float(used_contexts)).astype(np.float32, copy=False)
    y_true = np.asarray(y_test, dtype=np.float32)

    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    denom = float(np.sum((y_true - float(y_true.mean())) ** 2))
    r2 = float(1.0 - float(np.sum((pred - y_true) ** 2)) / denom) if denom > 0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["california", "ames"], default="ames")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for training/inference (e.g. 'auto', 'mps', 'cuda:0', or 'cpu').",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--context-samples",
        type=int,
        default=512,
        help=(
            "Context size (number of training rows) per TabPFN prompt. "
            "This is the main knob for memory use."
        ),
    )
    parser.add_argument(
        "--eval-contexts",
        type=int,
        default=1,
        help="Number of random training contexts to average over during evaluation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="Number of test rows per predict() call during evaluation.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if str(args.device).lower() == "auto":
        args.device = auto_detect_device()

    if str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
        torch.backends.cudnn.benchmark = True
    elif str(args.device) == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not mps_backend or not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device but torch.backends.mps.is_available() is False.")

    X_df, y = load_housing_regression_dataset(args.dataset, args.max_rows, args.seed)
    X_train, X_test, y_train, y_test = split_train_test(X_df, y, test_size=args.test_size, seed=args.seed)

    run_prefix = make_run_prefix(args.dataset, model_name="tabpfn_scratch", seed=args.seed)
    save_dir = default_save_dir(__file__, args.save_dir)
    metrics_path = save_dir / f"{run_prefix}_metrics.jsonl"

    regressor_init = {
        "device": args.device,
        "random_state": args.seed,
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": f"random:{args.seed}",
    }
    regressor = TabPFNRegressor(**regressor_init, fit_mode="batched", differentiable_input=False)

    regressor._initialize_model_variables()
    if len(regressor.models_) != 1:
        raise ValueError(f"Expected exactly 1 internal model, got {len(regressor.models_)}.")
    model = regressor.models_[0]
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_batching = EvalBatching(
        context_size=args.context_samples,
        n_contexts=args.eval_contexts,
        test_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    eval_init_args = {k: v for k, v in regressor_init.items() if k != "model_path"}

    print(f"Dataset: {args.dataset} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Device: {args.device} | model_path=random:{args.seed}")
    print(f"Context: {args.context_samples} train rows | Eval batch: {args.eval_batch_size} test rows")
    if str(args.device).startswith("cuda"):
        print(f"CUDA device: {torch.cuda.get_device_name(torch.device(args.device))}")
    print(f"Saving to: {save_dir} | Run prefix: {run_prefix}")

    initial = evaluate_with_contexts(
        regressor=regressor,
        init_args=eval_init_args,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batching=eval_batching,
    )
    print(f"Initial | MSE: {initial['mse']:.4f} | MAE: {initial['mae']:.4f} | R2: {initial['r2']:.4f}")
    append_jsonl(metrics_path, {"epoch": 0, "event": "eval", **{f"test_{k}": v for k, v in initial.items()}})

    splitter = partial(train_test_split, test_size=args.test_size, random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()

        # One epoch = iterate over many contexts built from the full dataset.
        X_contexts, y_contexts = build_epoch_contexts(
            X_train,
            y_train,
            context_size=args.context_samples,
            rng=rng,
        )

        training_datasets = regressor.get_preprocessed_datasets(
            X_contexts,
            y_contexts,
            splitter,
            max_data_size=None,
        )

        # `batch_size=1` here means "one dataset/context per step".
        dataloader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)

        progress = tqdm(dataloader, desc=f"Scratch train epoch {epoch}/{args.epochs}", unit="batch")
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
                progress.set_postfix(loss=f"{loss_val:.4f}", grad=f"{grad_norm:.3e}", step=str(global_step))

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

        metrics = evaluate_with_contexts(
            regressor=regressor,
            init_args=eval_init_args,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            batching=eval_batching,
        )
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

