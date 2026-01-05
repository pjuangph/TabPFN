from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvalBatching:
    """Controls how evaluation is chunked to avoid quadratic-attention OOM.

    `context_size` is the number of *training* rows included in the TabPFN "prompt".
    This is the dominant driver of attention memory.

    `test_batch_size` is the number of *test* rows passed to `predict()` per call.
    This controls how many queries are evaluated at once.
    """

    context_size: int = 512
    n_contexts: int = 1
    test_batch_size: int = 512
    seed: int = 0


def _select_rows(X: Any, idx: np.ndarray):
    """Index `X` by row indices for both pandas and numpy-like inputs."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def sample_train_context_indices(
    n_train: int, *, context_size: int, rng: np.random.Generator
) -> np.ndarray:
    if n_train < 2:
        raise ValueError("Need at least 2 training rows.")
    if context_size < 2:
        raise ValueError("context_size must be >= 2.")
    replace = n_train < context_size
    return rng.choice(n_train, size=min(context_size, n_train), replace=replace)


def build_epoch_contexts(
    X_train: Any,
    y_train: np.ndarray,
    *,
    context_size: int,
    rng: np.random.Generator,
) -> tuple[list[Any], list[np.ndarray]]:
    """Shuffle the full dataset and slice it into many contexts for one epoch."""
    if context_size < 2:
        raise ValueError("context_size must be >= 2.")
    perm = rng.permutation(len(X_train))
    context_size = min(context_size, len(perm))

    X_contexts: list[Any] = []
    y_contexts: list[np.ndarray] = []
    for start in range(0, len(perm), context_size):
        idx = perm[start : start + context_size]
        if len(idx) < 2:
            continue
        X_contexts.append(_select_rows(X_train, idx))
        y_contexts.append(y_train[idx])
    return X_contexts, y_contexts


def predict_with_context_batched(
    *,
    regressor,
    X_ctx: Any,
    y_ctx: np.ndarray,
    X_test: Any,
    test_batch_size: int,
) -> np.ndarray:
    """Fit on one context and predict the full test set in chunks."""
    if test_batch_size < 1:
        raise ValueError("test_batch_size must be >= 1.")

    regressor.fit(X_ctx, y_ctx)
    n_test = len(X_test)
    out = np.empty(n_test, dtype=np.float32)
    for start in range(0, n_test, test_batch_size):
        end = min(start + test_batch_size, n_test)
        X_batch = X_test.iloc[start:end] if hasattr(X_test, "iloc") else X_test[start:end]
        out[start:end] = np.asarray(regressor.predict(X_batch), dtype=np.float32).reshape(-1)
    return out

