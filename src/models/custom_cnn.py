from tensorflow.keras import layers, models, regularizers

def create_custom_cnn(input_shape=(128, 501, 1), num_classes=4):
    model = models.Sequential([
        layers.Input(input_shape),

        # Bloque 1
        layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        # Bloque 2
        layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        # Bloque 3
        layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        # Bloque 4 — más abstracción sin aumentar params
        layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # GlobalAveragePooling en lugar de Flatten
        # 16×62×128 → 128  (elimina 65M parámetros de golpe)
        layers.GlobalAveragePooling2D(),

        # Cabeza de clasificación ligera
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model