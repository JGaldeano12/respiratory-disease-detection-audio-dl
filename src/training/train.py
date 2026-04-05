import tensorflow as tf, numpy as np, os, random

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.training.custom_callback import ICBHI_Score_PrintingCallback
from src.models.custom_cnn import create_custom_cnn

def get_label(file_path):
    """
    Function to extract the label from the file path, specifically from the folder.
    """
    # Split the file path into its components using the OS-specific path separator and extract the relevant part to determine the label.
    elements = tf.strings.split(file_path, os.path.sep)

    # Map the extracted label to a numerical value based on the folder name.
    mapping = {'Healthy': 0, 'Crackle': 1, 'Wheeze': 2, 'Wheeze & Crackle': 3}
    label = mapping.get(elements[5], 4)

    # Return the numerical label corresponding to the class of the image.
    return label

def process_image(file_path):
    """
    Function to process an image file, including reading the image, decoding it, resizing it, and extracting the label.
    """
    # Extract the label from the file path using the get_label function.
    label = get_label(file_path)

    # Convert the label to a one-hot encoded vector with a depth of 4 (for 4 classes).
    label = tf.one_hot(label, depth=4)

    # Read the image file and decode it into a tensor with 3 color channels (RGB).
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)

    # Set the shape of the image tensor to ensure it has 3 channels.
    img.set_shape([None, None, 3])

    # Resize the image to a fixed size of 64x193 pixels.
    img = tf.image.resize(img, [64, 193])

    # Return the processed image and its corresponding one-hot encoded label.
    return img, label

def load_datasets(dir_dataset):
    """
    Function to load the training / testing datasets from the specified directories, process the images and labels, and prepare them for training.
    """
    # Load both training and testing datasets using tf.data.Dataset.list_files.
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle = False)
    test_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'), shuffle = False)

    # Map the process_image function to both datasets:
    train_dataset = train_dataset.map(process_image)
    test_dataset = test_dataset.map(process_image)

    # Make batches and shuffle them:
    train_dataset = train_dataset.shuffle(len(train_dataset)).batch(64, drop_remainder=True)
    test_dataset = test_dataset.shuffle(len(test_dataset)).batch(64, drop_remainder=True)

    # Return the prepared training and testing datasets.
    return train_dataset, test_dataset

def get_class_weights(dataset):
    """
    Function to obtain the class weights for the 'CategoricalFocalLoss' loss function.
    """
    # Get labels from the dataset and concatenate them into a single array.
    y_train = np.concatenate([y.numpy() for x, y in dataset], axis=0)

    # Transform the one-hot encoded labels into integer labels by taking the argmax along the appropriate axis.
    y_train_int = np.argmax(y_train, axis=1)

    # Get the unique class labels from the integer labels to compute class weights.
    classes = np.unique(y_train_int)

    # Calculate the class weights (must return an array).
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)

    # Cast to a numpy array to ensure it is in the correct format for use in the loss function.
    class_weights_array = np.array(class_weights)

    # Return the dictionary and array of class weights.
    return dict(enumerate(class_weights)), class_weights_array

def train(model, train_dataset, val_dataset, epochs=100):
    """
    Function to train a given model using the provided training and validation datasets.

    model: the neural network model to be trained (e.g., a custom CNN, ResNet50, or VGG19).
    train_dataset: the dataset used for training the model.
    val_dataset: the dataset used for validating the model during training.
    class_weights: a dictionary mapping class indices to weights, used to handle class imbalance during training (optional).
    epochs: the number of epochs to train the model (default is 10).
    batch_size: the number of samples per batch during training (default is 32).
    callbacks: a list of Keras callbacks to be applied during training (optional).
    """
    # Obtain the class weights using the get_class_weights function, which computes the weights based on the test dataset.
    alpha_dict, alpha = get_class_weights(val_dataset)

    # Compile the model with the Adam optimizer, categorical cross-entropy loss function, and the defined metrics.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss = 'CategoricalCrossentropy',
                  callbacks=[ICBHI_Score_PrintingCallback()])

    # Train the model using the fit method.
    history = model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, class_weight = alpha_dict, verbose = 2)

    # Save the model after training is complete.
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.h5'))

    # Return the training history, which contains information about the loss and metrics for each epoch.
    return history

# Load the training and validation datasets.
train_dataset, val_dataset = load_datasets('/app/data/processed')

# Create the model using the create_custom_cnn function, which defines a custom CNN architecture.
model = create_custom_cnn()

# Train the model using the defined train function, which includes class weights to handle class imbalance.
history = train(model = model, train_dataset = train_dataset, val_dataset = val_dataset, epochs = 100)