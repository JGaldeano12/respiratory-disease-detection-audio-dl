from keras import layers, models
from keras.applications import VGG19

def create_vgg_19(input_shape=(128, 129, 1), num_classes=4):
    """
    Create a VGG19-based model for multi-class image classification.

    The model uses VGG19 pre-trained on ImageNet as a frozen feature extractor.
    Since the input data contains a single channel, the channel is replicated
    three times to match the three-channel input expected by the pre-trained
    VGG19 model.

    The extracted features are flattened and processed by two fully connected
    layers with dropout before the final softmax classification layer.

    Args:
        input_shape (tuple[int, int, int], optional): Shape of a single input
            sample, excluding the batch dimension. Defaults to (128, 129, 1).
        num_classes (int, optional): Number of output classes. Defaults to 4.

    Returns:
        tf.keras.Model: Uncompiled VGG19-based classification model.
    """
    # Check if the model is downloaded and if not, download it
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape[:-1] + (3,))

    # Freeze the base model layers to prevent them from being updated during training
    base_model.trainable = False

    # Add new layers on top for the specific task
    model = models.Sequential([
        # First, we use the pre-trained VGG19 model
        base_model,

        # Then, flatten the output and add two blocks of fully connected layers.
        layers.Flatten(),

        #  First FC (Fully Connected) with 128 units, ReLU activation, and dropout for regularization.
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        # Second FC with 128 units, ReLU activation, and dropout for regularization.
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        # Output layer with softmax activation for multi-class classification.
        layers.Dense(num_classes, activation='softmax')])

    # Finally, return the created model.
    return model