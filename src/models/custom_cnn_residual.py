import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def residual_block(
    x,
    filters,
    initializer,
    seed,
    pool=True
):
    """
    Residual block composed of two convolutional layers.
    """

    shortcut = x

    # ---------------------------------------------------------
    # Conv 1
    # ---------------------------------------------------------
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # ---------------------------------------------------------
    # Conv 2
    # ---------------------------------------------------------
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    # ---------------------------------------------------------
    # Match channels if necessary
    # ---------------------------------------------------------
    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # ---------------------------------------------------------
    # Residual connection
    # ---------------------------------------------------------
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    # ---------------------------------------------------------
    # Pooling
    # ---------------------------------------------------------
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)

    return x

def create_custom_cnn(
    input_shape=(128, 97, 1),
    num_classes=4,
    seed=12345
):

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    inputs = layers.Input(shape=input_shape)

    # ==========================================================
    # BLOCK 1
    # ==========================================================
    x = residual_block(
        inputs,
        filters=32,
        initializer=initializer,
        seed=seed,
        pool=True
    )

    # ==========================================================
    # BLOCK 2
    # ==========================================================
    x = residual_block(
        x,
        filters=64,
        initializer=initializer,
        seed=seed,
        pool=True
    )

    # ==========================================================
    # BLOCK 3
    # ==========================================================
    x = residual_block(
        x,
        filters=128,
        initializer=initializer,
        seed=seed,
        pool=True
    )

    # ==========================================================
    # BLOCK 4
    # ==========================================================
    x = residual_block(
        x,
        filters=256,
        initializer=initializer,
        seed=seed,
        pool=False
    )

    # ==========================================================
    # BLOCK 5
    # ==========================================================
    x = residual_block(
        x,
        filters=256,
        initializer=initializer,
        seed=seed,
        pool=False
    )

    # ==========================================================
    # CLASSIFIER
    # ==========================================================

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_initializer=initializer,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(
        0.5,
        seed=seed
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=initializer
    )(x)

    model = models.Model(inputs, outputs)

    return model