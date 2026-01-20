# src/data.py

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load labels from a CSV file


def load_labels(csv_path: Path) -> pd.DataFrame:
    """Load labels from a CSV file into a pandas DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def validate_schema(df: pd.DataFrame, filename_col: str, label_col: str) -> None:
    """Validate that required columns exist in the DataFrame and contain no NaN values."""
    if filename_col not in df.columns:
        raise ValueError(f"Column '{filename_col}' not found in DataFrame")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame")

    # Validate NaN values
    if df[filename_col].isnull().any():
        nan = int(df[filename_col].isnull().sum())
        raise ValueError(f"Column '{filename_col}' contains {nan} NaN values")
    if df[label_col].isnull().any():
        nan = int(df[label_col].isnull().sum())
        raise ValueError(f"Column '{label_col}' contains {nan} NaN values")


def resolve_image_paths(
    df: pd.DataFrame, images_dir: Path, filename_col: str
) -> tuple[pd.Series, float]:
    """Resolve image paths based on a directory and filenames from the DataFrame"""

    images_dir = Path(images_dir)

    return df[filename_col].apply(lambda x: images_dir / x)


def check_missing_files(images_dir: pd.Series) -> pd.Series:
    """Check for missing image files in the given directory."""
    p = images_dir.apply(Path)

    exist = p.apply(lambda x: x.exists())

    missing = p[~exist]

    missing_rate = float(len(missing) / len(p)) if len(p) > 0 else 0.0

    return missing, missing_rate


# Reproducible samples
def get_sample(df: pd.DataFrame, seed: int, n: int) -> pd.DataFrame:
    """Get a reproducible random sample of n rows from the DataFrame."""
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _make_age_bins(y: pd.Series, n_bins: int) -> pd.Series:
    """Create age bins for stratified sampling using pd.qcut and pd.cut as fallback."""
    y = y.astype(float)

    try:
        return pd.qcut(y, bins=n_bins, duplicates="drop").astype(str)
    except Exception:
        return pd.cut(y, bins=n_bins).astype(str)


def make_splits(
    df: pd.DataFrame,
    *,
    filename_col: str,
    target_col: str,
    val_size: float,
    test_size: float,
    seed: int,
    stratify_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train, validation, and test sets."""
    if not (0 < val_size < 1) or not (0 < test_size < 1) or (val_size + test_size >= 1):
        raise ValueError(
            "val_size and test_size must be in (0,1) and val_size + test_size < 1"
        )

    df = df.copy()
    # Make bins to stratify on
    bins = _make_age_bins(df[target_col], stratify_bins)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=bins,
        shuffle=True,
    )
    # Recompute bins for train_val split
    bins_train_val = _make_age_bins(train_val_df[target_col], stratify_bins)

    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=seed,
        stratify=bins_train_val,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path | str,
    *,
    filename_col: str,
    target_col: str,
) -> dict:
    """Save splits as light CSV + JSON summary artifacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    data_cols = [filename_col, target_col]

    train_path = out_dir / "split_train.csv"
    val_path = out_dir / "split_val.csv"
    test_path = out_dir / "split_test.csv"

    train_df[data_cols].to_csv(train_path, index=False)
    val_df[data_cols].to_csv(val_path, index=False)
    test_df[data_cols].to_csv(test_path, index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "columns": data_cols,
    }

    summary_dir = out_dir / "split_summary.json"

    summary_dir.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "summary_json": str(summary_dir),
    }


def _decode_and_resize(img_path: tf.Tensor, img_size: tuple[int, int]) -> tf.Tensor:
    """
    Reads image from path and decodes as JPG, resizes and normalize to (0,1)
    """

    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32) / 255

    return img


def build_dataset(
    df: pd.DataFrame,
    *,
    images_dir: Path | str,
    filename_col: str,
    target_col: str,
    img_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
    cache: bool = False,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Builds a TensorFlow dataset (paths->imagen tensor) ready for training
    """

    paths = (images_dir / df[filename_col].astype(str)).astype(str).values
    labels = df[target_col].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 10_000), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, y: (_decode_and_resize(p, img_size), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        ds.cache()

    # batch
    ds = ds.batch(batch_size, drop_remainder=False)

    if prefetch:
        ds.prefetch(tf.data.AUTOTUNE)

    return ds
