import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def conv_block(x, filters, initializer, pool=True):
    """
    Apply a plain convolutional block to an input tensor.

    The block consists of two 3x3 convolutional layers, each followed by batch
    normalization and a ReLU activation. L2 regularization is applied to the
    convolutional kernels. Optional max pooling is applied at the end of the
    block.

    Args:
        x (tf.Tensor): Input tensor to the block.
        filters (int): Number of filters used in the convolutional layers.
        initializer: Initializer used for the convolutional kernels.
        pool (bool, optional): Whether to apply 2x2 max pooling after the
            block. Defaults to True.

    Returns:
        tf.Tensor: Output tensor produced by the block.
    """
    # Apply two convolutional layers to extract features from the input.
    # First convolutional layer: 3x3 kernel, same padding, ReLU activation.
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Second convolutional layer: 3x3 kernel, same padding, ReLU activation.
    x = layers.Conv2D(filters, (3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # If `pool` is True, apply 2x2 max pooling to reduce the spatial dimensions of the output.
    if pool:
        x = layers.MaxPooling2D((2, 2))(x)

    # Return the output tensor of the block.
    return x

def create_custom_cnn(input_shape=(128, 129, 1), num_classes=4, seed=12345):
    """
    Create a custom plain convolutional neural network for image classification.

    The model is composed of four convolutional blocks with progressively
    increasing numbers of filters. The first three blocks include max pooling,
    while the last preserves the spatial dimensions. The convolutional
    feature maps are then aggregated using global average pooling and passed
    through a fully connected layer with batch normalization and dropout before
    the final softmax classification layer.

    Args:
        input_shape (tuple[int, int, int], optional): Shape of a single input
            sample, excluding the batch dimension. Defaults to (128, 129, 1).
        num_classes (int, optional): Number of output classes. Defaults to 4.
        seed (int, optional): Random seed used to initialize the Glorot
            initializer and the dropout layer. Defaults to 12345.

    Returns:
        tf.keras.Model: Uncompiled custom CNN with `num_classes` softmax
        output units.
    """
    # Initialize the Glorot uniform initializer with the provided seed for reproducibility.
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    inputs = layers.Input(shape=input_shape)

    # Apply a series of convolutional blocks with increasing filter sizes.
    # The first three blocks include max pooling to reduce spatial dimensions,
    # while the last block preserves the spatial dimensions.
    x = conv_block(inputs, filters=32,  initializer=initializer, pool=True)
    x = conv_block(x, filters=64,  initializer=initializer, pool=True)
    x = conv_block(x, filters=64, initializer=initializer, pool=True)
    x = conv_block(x, filters=128, initializer=initializer, pool=False)

    # Apply global average pooling to reduce the spatial dimensions of the feature maps to a single value per channel. This is
    # followed by a fully connected layer with 256 units, ReLU activation, batch normalization, and dropout for regularization.
    # Finally, a softmax output layer is added for multi-class classification.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=seed)(x)
    x = layers.Dense(256, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, seed=seed)(x)

    # Add the final output layer with softmax activation for multi-class classification. The number of units in this layer corresponds
    # to the number of classes specified by `num_classes`.
    outputs = layers.Dense(num_classes, activation="softmax", kernel_initializer=initializer)(x)

    # Create the Keras model by specifying the input and output tensors. The model is uncompiled and ready for training.
    model = models.Model(inputs, outputs)

    # Return the uncompiled model.
    return model