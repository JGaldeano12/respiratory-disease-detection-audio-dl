from tensorflow.keras import layers, models, regularizers
import tensorflow as tf, argparse, os

def create_efficientnet_model(input_shape=(128, 129, 1), num_classes=4, seed = 12345):
    tf.random.set_seed(seed)

    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 129, 3)
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model