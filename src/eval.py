# src/eval.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf


def predict_regression(model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Returns y_true and y_pred for a dataset"""
    y_true = []
    y_pred = []

    for x, y in ds:
        preds = model.predict(x, verbose=0).reshape(-1)
        y_true.append(y.numpy().reshape(-1))
        y_pred.append(preds)

    return np.concatenate(y_true), np.concatenate(y_pred)


def mae_by_age_bins(
    y_true: np.ndarray, y_pred: np.ndarray, bins: list[int]
) -> pd.DataFrame:
    """Get MAE for age bins to detect where the model fails the most"""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    df["bin"] = pd.cut(df["y_true"], bins=bins, include_lowest=True)

    out = (
        df.groupby("bin")
        .agg(count=("abs_err", "size"), mae=("abs_err", "mean"))
        .reset_index()
    )
    return out
