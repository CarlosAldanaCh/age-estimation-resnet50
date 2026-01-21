# src/model.py

from __future__ import annotations

import tensorflow as tf


def build_resnet50_regressor(
    *,
    img_size: tuple[int, int],
    dropout: float = 0.2,
    dense_units: int = 128,
) -> tf.keras.Model:
    """
    Creates a ResNet50 pretrained regressor model (frozen backbone)
    """

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
        pooling="avg",
    )

    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))

    x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)
    x = backbone(x, training=False)

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    outputs = tf.keras.layers.Dense(1, name="age")(x)

    model = tf.keras.Model(inputs, outputs, name="resnet50_age_regressor")

    return model


def set_fine_tuning(
    model,
    backbone_name: str = "resnet50",
    fine_tune_n_last: int = 30,
    freeze_batchnorm: bool = True,
) -> None:
    """
    Sets the model for fine tuning mode, unfreezing n last backbone layers,
    Optional but very recommended freeze BatchNorm for stability
    """

    backbone = model.get_layer(backbone_name)
    backbone.trainable = True

    # Freeze everything and then unfreeze N last

    for layer in backbone.layers:
        layer.trainable = False

    if fine_tune_n_last > 1:
        for layer in backbone.layers[-fine_tune_n_last:]:
            layer.trainable = True

    if freeze_batchnorm:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
