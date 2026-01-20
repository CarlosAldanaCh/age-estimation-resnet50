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
