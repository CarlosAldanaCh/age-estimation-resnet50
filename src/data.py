# src/data.py

from __future__ import annotations
from pathlib import Path
import pandas as pd

# Load labels from a CSV file


def load_labels(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def validate_schema(df: pd.DataFrame, filename_col: str, label_col: str) -> None:
    # Validate that required columns exist in the DataFrame
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


# Resolve image paths based on a directory and filenames from the DataFrame


def resolve_image_paths(
    df: pd.DataFrame, images_dir: Path, filename_col: str
) -> tuple[pd.Series, float]:

    images_dir = Path(images_dir)

    return df[filename_col].apply(lambda x: images_dir / x)


# Check missing files
def check_missing_files(images_dir: pd.Series) -> pd.Series:
    p = images_dir.apply(Path)

    exist = p.apply(lambda x: x.exists())

    missing = p[~exist]

    missing_rate = float(len(missing) / len(p)) if len(p) > 0 else 0.0

    return missing, missing_rate


# Reproducible samples
def get_sample(df: pd.DataFrame, seed: int, n: int) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
