"""Fine-tune TabPFNRegressor across many OpenML regression datasets.

This is the closest practical "dump a lot of public tabular datasets into TabPFN"
workflow that fits the current repo: you start from pretrained weights and run
gradient updates across a *corpus of datasets* (meta-finetuning).

It does NOT reproduce TabPFN's original pretraining recipe (which relied heavily
on synthetic task generation and specific priors), but it can be useful for:
  - learning how the model is optimized end-to-end
  - experimenting with multi-dataset adaptation
  - later adding instrumentation (embeddings/attention hooks) while training

Example:
  python examples/finetune_openml_regression_corpus.py --data-ids 534,43928
  python examples/finetune_openml_regression_corpus.py --from-file examples/openml_regression_ids.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

from housing_regression_utils import (
    append_jsonl,
    default_save_dir,
    load_openml_regression_dataset_by_id,
    make_run_prefix,
    split_train_test,
)


@dataclass(frozen=True)
class CorpusItem:
    data_id: int
    name: str


def parse_data_ids_csv(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    if not ids:
        raise ValueError("No data ids provided.")
    return ids


def read_ids_from_file(path: str) -> list[int]:
    p = Path(path)
    ids: list[int] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line))
    if not ids:
        raise ValueError(f"No dataset ids found in {path}")
    return ids


def evaluate_single_dataset(
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-ids", type=str, default=None, help="Comma-separated OpenML data ids.")
    parser.add_argument("--from-file", type=str, default=None, help="Path to a file with one OpenML data id per line.")
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--context-samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over the dataset corpus.")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--model-path",
        type=str,
        default="auto",
        help="Use 'auto' for pretrained weights, or 'random'/'scratch' to avoid downloads.",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-every-datasets", type=int, default=10, help="Save a checkpoint after every N datasets.")
    parser.add_argument("--eval-every-datasets", type=int, default=5, help="Evaluate on each dataset every N datasets.")
    args = parser.parse_args()

    if bool(args.data_ids) == bool(args.from_file):
        raise ValueError("Provide exactly one of --data-ids or --from-file.")

    if args.data_ids:
        data_ids = parse_data_ids_csv(args.data_ids)
    else:
        data_ids = read_ids_from_file(args.from_file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_prefix = make_run_prefix("openml_corpus", model_name="tabpfn_finetune", seed=args.seed)
    save_dir = default_save_dir(__file__, args.save_dir)
    metrics_path = save_dir / f"{run_prefix}_metrics.jsonl"

    regressor_config = {
        "device": args.device,
        "random_state": args.seed,
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": args.model_path,
    }
    regressor = TabPFNRegressor(**regressor_config, fit_mode="batched", differentiable_input=False)

    # Initialize internals so we can access the torch model for training.
    regressor._initialize_model_variables()

    if len(regressor.models_) != 1:
        raise ValueError(f"Finetuning expects exactly 1 foundation model, got {len(regressor.models_)}.")
    model = regressor.models_[0]
    optimizer = Adam(model.parameters(), lr=args.lr)

    eval_config = {**regressor_config, "inference_config": {"SUBSAMPLE_SAMPLES": args.context_samples}}

    corpus: list[CorpusItem] = []
    for data_id in data_ids:
        try:
            _, _, name = load_openml_regression_dataset_by_id(data_id, max_rows=1, seed=args.seed)
        except Exception:
            name = f"openml_{data_id}"
        corpus.append(CorpusItem(data_id=data_id, name=name))

    print(f"Corpus size: {len(corpus)} datasets | Device: {args.device} | LR: {args.lr}")
    print(f"Saving to: {save_dir} | Run prefix: {run_prefix}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_bar = tqdm(corpus, desc=f"Corpus epoch {epoch}/{args.epochs}", unit="dataset")
        for dataset_index, item in enumerate(epoch_bar, start=1):
            try:
                X_df, y, ds_name = load_openml_regression_dataset_by_id(
                    item.data_id, max_rows=args.max_rows, seed=args.seed
                )
            except Exception as e:
                append_jsonl(
                    metrics_path,
                    {"epoch": epoch, "data_id": item.data_id, "name": item.name, "event": "load_failed", "error": str(e)},
                )
                continue

            X_train, X_test, y_train, y_test = split_train_test(
                X_df, y, test_size=args.test_size, seed=args.seed
            )

            splitter = partial(train_test_split, test_size=args.test_size, random_state=args.seed)
            try:
                training_datasets = regressor.get_preprocessed_datasets(
                    X_train,
                    y_train,
                    splitter,
                    max_data_size=min(args.context_samples, len(X_train)),
                )
            except Exception as e:
                append_jsonl(
                    metrics_path,
                    {"epoch": epoch, "data_id": item.data_id, "name": ds_name, "event": "preprocess_failed", "error": str(e)},
                )
                continue

            dataloader = DataLoader(training_datasets, batch_size=1, collate_fn=meta_dataset_collator)
            batch_bar = tqdm(dataloader, desc=f"Finetune {ds_name} ({item.data_id})", unit="batch", leave=False)

            model.train()
            for batch in batch_bar:
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
                global_step += 1
                batch_bar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", step=str(global_step))

            if args.eval_every_datasets > 0 and (dataset_index % args.eval_every_datasets == 0):
                try:
                    metrics = evaluate_single_dataset(
                        regressor, eval_config, X_train, y_train, X_test, y_test
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "epoch": epoch,
                            "data_id": item.data_id,
                            "name": ds_name,
                            "event": "eval",
                            "global_step": global_step,
                            "test_mse": metrics["mse"],
                            "test_mae": metrics["mae"],
                            "test_r2": metrics["r2"],
                        },
                    )
                except Exception as e:
                    append_jsonl(
                        metrics_path,
                        {"epoch": epoch, "data_id": item.data_id, "name": ds_name, "event": "eval_failed", "error": str(e)},
                    )

            if args.save_every_datasets > 0 and (dataset_index % args.save_every_datasets == 0):
                torch.save(
                    {
                        "epoch": epoch,
                        "dataset_index": dataset_index,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    save_dir / f"{run_prefix}_checkpoint_e{epoch:02d}_d{dataset_index:05d}.pt",
                )


if __name__ == "__main__":
    main()
