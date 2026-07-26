import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def create_custom_cnn(
    input_shape=(128, 129, 1),
    num_classes=4,
    seed=12345
):

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    model = models.Sequential([

        layers.Input(shape=input_shape),

        # ==========================================================
        # BLOCK 1
        # ==========================================================
        layers.Conv2D(
            32,
            (3, 3),
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # ==========================================================
        # BLOCK 2
        # ==========================================================
        layers.Conv2D(
            64,
            (3, 3),
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # ==========================================================
        # BLOCK 3
        # ==========================================================
        layers.Conv2D(
            128,
            (3, 3),
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # ==========================================================
        # BLOCK 4
        # ==========================================================
        layers.Conv2D(
            256,
            (3, 3),
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # ==========================================================
        # BLOCK 5
        # ==========================================================
        layers.Conv2D(
            256,
            (3, 3),
            padding='same',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # ==========================================================
        # CLASSIFIER
        # ==========================================================
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            256,
            activation='relu',
            kernel_initializer=initializer,
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.5,
            seed=seed
        ),

        layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer=initializer
        )
    ])

    return model