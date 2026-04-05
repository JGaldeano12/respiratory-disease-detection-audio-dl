from tensorflow.keras import layers, models

def create_custom_cnn(input_shape = (64, 193, 3), num_classes = 4):
    """
    Function to create a custom Convolutional Neural Network (CNN) architecture for classifying spectrogram images into different classes based on the input shape and number of classes.

    input_shape: the shape of the input data (e.g., (64, 64, 1) for grayscale spectrogram images).
    num_classes: the number of output classes for classification (e.g., 4 for Crackle, Wheeze, Wheeze & Crackle, Healthy).
    """
    model = models.Sequential([
    layers.Input(input_shape),
    
    # First convolutional block with 32 filters, kernel size of (3,3), batch normalization, ReLU activation, and max pooling with strides of (2,4).
    layers.Conv2D(filters = 64, kernel_size = (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'), 
    layers.MaxPooling2D(strides=(2,4)),

    # Second convolutional block with 64 filters, kernel size of (3,3), batch normalization, ReLU activation, and max pooling with strides of (2,4).
    layers.Conv2D(filters = 80, kernel_size = (3,3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'), 
    layers.MaxPooling2D(strides=(3,3)),

    # Third convolutional block with 128 filters, kernel size of (5,5), batch normalization, ReLU activation, and max pooling with strides of (3,3).
    layers.Conv2D(filters = 128, kernel_size = (5,5), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'), 
    layers.MaxPooling2D(strides=(3,3)),

    # Now, I will flatten the output of the convolutional blocks and add fully connected layers with batch normalization, ReLU activation, and dropout for regularization.
    layers.BatchNormalization(),
    layers.Flatten(),

    # First fully connected layer with 1024 units, batch normalization, ReLU activation, and dropout with a rate of 0.4 for regularization.
    layers.Dense(1024, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Second fully connected layer with 4 units (corresponding to the number of classes) and softmax activation for multi-class classification.
    layers.Dense(1024, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Output layer with softmax activation for multi-class classification.
    layers.Dense(4, activation = num_classes)])

    # Finally, I will return the created model.
    return model