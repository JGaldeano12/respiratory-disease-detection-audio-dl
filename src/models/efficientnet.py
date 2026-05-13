from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

def create_efficientnet_model(input_shape=(128, 129, 1), num_classes=4):
    # EfficientNetB0 espera 3 canales, replicamos el canal 1 -> 3
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)  # (128, 129, 3)

    # Base preentrenada congelada inicialmente
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 129, 3)
    )
    base_model.trainable = False  # Fase 1: congelado

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base_model