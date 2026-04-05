from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19

def create_vgg_19(input_shape=(64, 193, 3), num_classes=4):
    # Check if the model is downloaded and if not, download it
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 193, 3))

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
        layers.Dense(4, activation='softmax')])

    # Finally, return the created model.
    return model