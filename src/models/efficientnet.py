import os, random, numpy as np, tensorflow as tf, glob
from tensorflow.keras import layers, models, regularizers

def create_efficientnet_model(input_shape=(128, 129, 1), num_classes=4, seed=12345):
    """
    Function to create an EfficientNet-based model for respiratory cycle classification.
    """
    # Use GlorotUniform initializer with the specified seed for reproducibility.
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    # Define the input layer with the specified input shape.
    inputs = layers.Input(shape=input_shape)

    # Since EfficientNetB0 expects 3-channel input, we repeat the single channel 3 times to create a 3-channel input.
    x = layers.Lambda(lambda t: tf.repeat(t, 3, axis=-1))(inputs)

    # Load the EfficientNetB0 model pre-trained on ImageNet, excluding the top classification layers, and set it to non-trainable for transfer learning.
    base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape[:-1] + (3,))

    # Freeze the base model to prevent its weights from being updated during the initial training phase.
    base_model.trainable = False

    # Pass the input through the base model. We set training=False to ensure that layers like BatchNormalization behave in inference mode, 
    # which is important since the base model is frozen.
    # x = base_model(x, training=False)
    x = base_model(x)

    # Apply global average pooling to reduce the spatial dimensions of the feature maps output by the base model.
    x = layers.GlobalAveragePooling2D()(x)

    # Add a fully connected layer with 128 units, ReLU activation, GlorotUniform initialization, and L2 regularization to help prevent overfitting.
    x = layers.Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(1e-3))(x)

    # Add batch normalization to stabilize and accelerate training.
    x = layers.BatchNormalization()(x)

    # Add dropout with a rate of 0.5 to further prevent overfitting, using the specified seed for reproducibility.
    x = layers.Dropout(0.5, seed=seed)(x)

    # Finally, add the output layer with a number of units equal to the number of classes, softmax activation for 
    # multi-class classification, and GlorotUniform initialization.
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer)(x)

    # Create the model by specifying the inputs and outputs.
    model = models.Model(inputs, outputs)

    # Return both the model and the base model (EfficientNet) so that we can unfreeze the base model later for fine-tuning.
    return model, base_model