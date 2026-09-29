import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_resnet_50(input_shape=(128, 129, 1), num_classes=4):
    """
    Create a ResNet50-based model for multi-class image classification.

    The model uses ResNet50 pre-trained on ImageNet as a frozen feature
    extractor. Since the input data contains a single channel, the channel is
    replicated three times to match the three-channel input expected by the
    pre-trained ResNet50 model.

    The extracted features are aggregated with global average pooling and
    processed by two fully connected layers (with batch normalization and
    dropout) before the final softmax classification layer.

    Args:
        input_shape (tuple[int, int, int], optional): Shape of a single input
            sample, excluding the batch dimension. Defaults to (128, 129, 1).
        num_classes (int, optional): Number of output classes. Defaults to 4.

    Returns:
        tf.keras.Model: Uncompiled ResNet50-based classification model.
    """

    # Define the input tensor using the shape expected by the dataset.
    inputs = layers.Input(shape=input_shape)

    # Convert the single-channel input into a three-channel representation to
    # make it compatible with the ImageNet pre-trained ResNet50 model.
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)

    # Load ResNet50 without its original ImageNet classification layers and use
    # the pre-trained convolutional network as a feature extractor.
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape[:-1] + (3,))

    # Freeze the pre-trained layers during the initial training phase so that
    # only the newly added classification layers are updated.
    base_model.trainable = False

    # Extract high-level feature maps from the input using the pre-trained
    # ResNet50 convolutional base.
    x = base_model(x)

    # Aggregate the spatial feature maps into a single 2048-d vector per
    # sample. GlobalAveragePooling2D is used instead of Flatten: flattening
    # the (4, 8, 2048) feature map produces a 65,536-d vector feeding a
    # randomly-initialized Dense(1024) layer (~67M parameters), which at the
    # phase-1 learning rate is prone to exploding activations and permanent
    # ReLU death (constant-output collapse). GAP avoids both the dimension
    # blow-up and that instability.
    x = layers.GlobalAveragePooling2D()(x)

    # Learn task-specific feature representations through two dense layers.
    # BatchNormalization is applied before each activation to keep
    # pre-activation statistics centered and prevent dying ReLUs; Dropout is
    # applied after each block to reduce overfitting.
    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1024, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    # Generate a probability distribution across all target classes.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Connect the input, ResNet50 feature extractor, and classification head
    # into a single Keras model.
    model = models.Model(inputs, outputs)
    
    # Return the complete uncompiled model.
    return model, base_model