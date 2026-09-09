import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, recall_score

def get_label(file_path):
    """
    Extract the class label from the file path.

    The function maps the class name contained in the file path to its
    corresponding integer label.

    Args:
        file_path (tf.Tensor): Path to the NumPy spectrogram file.

    Returns:
        tf.Tensor: Integer-encoded class label.
    """
    elements = tf.strings.split(file_path, os.path.sep)
    label_str = elements[5]

    keys = tf.constant(['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=4
    )

    label = table.lookup(label_str)
    return label

def load_npy(path):
    """
    Load a NumPy spectrogram from a file.

    The function loads the spectrogram, verifies its expected width,
    and converts it to float32 format.

    Args:
        path (tf.Tensor): Path to the NumPy spectrogram file.

    Returns:
        np.ndarray: Loaded spectrogram with float32 data type.
    """
    path = path.numpy().decode("utf-8")
    spec = np.load(path)
    assert spec.shape[1] == 251, (f"Unexpected spectrogram width: {spec.shape[1]}")

    return spec.astype(np.float32)

def process_npy(file_path):
    """
    Load and preprocess a NumPy spectrogram for model evaluation.

    The function extracts the class label from the file path, converts it
    to one-hot encoding, loads the corresponding spectrogram, and sets
    its expected shape.

    Args:
        file_path (tf.Tensor): Path to the NumPy spectrogram file.

    Returns:
        tuple:
            tf.Tensor: Preprocessed spectrogram.
            tf.Tensor: One-hot encoded class label.
    """
    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)

    spec = tf.py_function(load_npy, [file_path], tf.float32)
    spec.set_shape([128, 251, 1])

    return spec, label

def load_test_dataset(dir_dataset):
    """
    Load and preprocess the test dataset.

    The function retrieves all NumPy spectrogram files from the test
    directory, applies the preprocessing pipeline, batches the samples,
    and enables prefetching for efficient evaluation.

    Args:
        dir_dataset (str): Root directory containing the test dataset.

    Returns:
        tf.data.Dataset: Preprocessed and batched test dataset.
    """
    test_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'), shuffle=False)

    # Ensure deterministic processing and evaluation order.
    options = tf.data.Options()
    options.experimental_deterministic = True

    test_dataset = test_dataset.with_options(options)

    # Load and preprocess the spectrograms.
    test_dataset = test_dataset.map(process_npy, num_parallel_calls=1)

    # Group samples into batches for model inference.
    test_dataset = test_dataset.batch(1, drop_remainder=True)

    # Prefetch batches to improve evaluation efficiency.
    test_dataset = test_dataset.prefetch(1)

    return test_dataset

def create_confusion_matrix(labels, predictions):
    """
    Create and save a confusion matrix based on the true and predicted labels.

    Args:
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.

    Returns:
        str: Confirmation message after the confusion matrix is saved.
    """
    # Compute the confusion matrix.
    cm = confusion_matrix(labels, predictions)

    # Define the class names corresponding to the labels in get_label().
    class_names = ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']

    # Create and save a heatmap of the confusion matrix.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'/app/src/testing/confusion_matrix_' f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
    plt.close()

    return "Confusion matrix saved!"

def calculate_specificity_per_class(labels, predictions):
    """
    Calculate the specificity for each class.

    Specificity is calculated as the proportion of true negative samples
    among all samples that do not belong to the corresponding class.

    Args:
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.

    Returns:
        list: Specificity value for each class.
    """
    # Determine the number of classes from the true labels.
    num_classes = np.max(labels) + 1
    specificities = []

    # Calculate specificity independently for each class.
    for i in range(num_classes):
        true_negatives = np.sum((labels != i) & (predictions != i))
        false_positives = np.sum((labels != i) & (predictions == i))
        specificity = (true_negatives / (true_negatives + false_positives + np.finfo(float).eps))
        specificities.append(specificity)

    return specificities

def compute_recall_specificity_score(labels, predictions):
    """
    Compute recall, specificity, and the ICBHI score.

    The ICBHI score is calculated as the average of the mean recall and
    mean specificity across all classes.

    Args:
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.

    Returns:
        tuple:
            float: Mean recall across all classes.
            float: Mean specificity across all classes.
            float: ICBHI score.
    """
    # Compute recall independently for each class.
    recall_per_class = recall_score(labels, predictions, average=None)

    # Compute specificity independently for each class.
    specificity_per_class = calculate_specificity_per_class(labels, predictions)

    # Compute the mean recall and specificity.
    mean_recall = np.mean(recall_per_class)
    mean_specificity = np.mean(specificity_per_class)

    # Compute the ICBHI score.
    icbhi_score = (mean_recall + mean_specificity) / 2

    return mean_recall, mean_specificity, icbhi_score

def evaluate_model(model_path, dir_test_dataset):
    """
    Evaluate a trained model on the test dataset.

    The function loads the trained model, processes the test dataset,
    generates predictions, and computes the confusion matrix, recall,
    specificity, and ICBHI score.

    Args:
        model_path (str): Path to the trained Keras model.
        dir_test_dataset (str): Root directory containing the test dataset.

    Returns:
        str: Confirmation message after the evaluation is completed.
    """
    # Load the trained model without restoring its compilation configuration.
    model = load_model(model_path, compile=False)

    # Load and preprocess the test dataset.
    test_dataset = load_test_dataset(dir_test_dataset)

    # Initialize lists to store the true and predicted labels.
    predictions = []
    labels = []

    # Generate predictions for each batch in the test dataset.
    for images, batch_labels in test_dataset:
        batch_predictions = model.predict(images, verbose=0)

        # Convert model outputs and one-hot encoded labels to class indices.
        batch_predictions = np.argmax(batch_predictions, axis=1)
        batch_labels = np.argmax(batch_labels, axis=1)

        predictions.append(batch_predictions)
        labels.append(batch_labels)

    # Concatenate all batches into single arrays.
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # Create and save the confusion matrix.
    create_confusion_matrix(labels, predictions)

    # Compute recall, specificity, and the ICBHI score.
    recall, specificity, icbhi_score = (compute_recall_specificity_score(labels, predictions))

    # Display the evaluation metrics.
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ICBHI Score: {icbhi_score:.4f}")

    return "Evaluation complete!"

evaluate_model('/app/models/custom_cnn/best_model_epoch.keras', '/app/data/processed')