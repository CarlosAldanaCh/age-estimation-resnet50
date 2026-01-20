# src/train.py

from __future__ import annotations

from pathlib import Path
import json
import tensorflow as tf


def make_callbacks(
    *, models_dir: Path | str, metrics_dir: Path | str, monitor: str = "val_mae"
) -> list:
    """
    Creates standard callbacks such as checkpoints, early stopping, reduce LR and CSV logger
    """

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "baseline_best.keras"),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(metrics_dir / "baseline_history.csv"),
            append=False,
        ),
    ]


def compile_model(model: tf.keras.Model, lr: float) -> None:
    """
    Compiles the Keras model using Adam + MAE
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mae",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )


def train_stage_1(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int,
    callbacks: list,
) -> tf.keras.callbacks.History:
    """
    Trains the model in feature extraction mode (frozen backbone).
    """
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )


def save_test_metrics(metrics: dict, out_dir: Path | str) -> None:
    """
    Save a JSON file with the best metrics for traceability
    """
    out_dir = Path(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
