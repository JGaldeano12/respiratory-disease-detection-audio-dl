import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19


def create_vgg_19(input_shape=(128, 255, 1), num_classes=4):
    """
    Create a VGG19-based model for multi-class image classification.

    The model uses VGG19 pre-trained on ImageNet as a frozen feature extractor.
    Since the input data contains a single channel, the channel is replicated
    three times to match the three-channel input expected by the pre-trained
    VGG19 model.

    The extracted features are aggregated with global average pooling and
    processed by two fully connected layers (with batch normalization and
    dropout) before the final softmax classification layer.

    Args:
        input_shape (tuple[int, int, int], optional): Shape of a single input
            sample, excluding the batch dimension. Defaults to (128, 255, 1).
        num_classes (int, optional): Number of output classes. Defaults to 4.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]:
            A tuple containing the complete classification model and the
            VGG19 base model.
    """

    # Define the input tensor using the shape expected by the dataset.
    inputs = layers.Input(shape=input_shape)

    # Convert the single-channel input into a three-channel representation
    # to make it compatible with the ImageNet pre-trained VGG19 model.
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)

    # Load VGG19 without its original ImageNet classification layers.
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape[:-1] + (3,))

    # Freeze the pre-trained layers during the initial training phase.
    base_model.trainable = False

    # Extract high-level feature maps from the input.
    x = base_model(x)

    # Aggregate the spatial feature maps into a single feature vector per
    # sample using global average pooling, instead of Flatten. This keeps
    # the input to the new Dense layers compact and avoids feeding a large,
    # unnormalized flattened vector into randomly-initialized weights (the
    # same instability source addressed in the ResNet50 head).
    x = layers.GlobalAveragePooling2D()(x)

    # First fully connected layer. BatchNormalization is applied before the
    # activation to keep pre-activation statistics centered and prevent
    # dying ReLUs.
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    # Second fully connected layer.
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)

    # Final classification layer.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the complete model.
    model = models.Model(inputs, outputs)

    # Return both the complete model and the VGG19 base model.
    return model, base_model