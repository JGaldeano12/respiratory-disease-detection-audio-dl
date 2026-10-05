import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from src.data.utils import process_npy
from src.evaluation.metrics import icbhi_official_score

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

    # Load and preprocess the spectrograms. training=False is passed explicitly
    # so that SpecAugment is not applied at evaluation time (process_npy
    # defaults to training=True, which was silently augmenting the test data).
    test_dataset = test_dataset.map(lambda x: process_npy(x, training=False), num_parallel_calls=1)

    # Group samples into batches for model inference.
    test_dataset = test_dataset.batch(1, drop_remainder=True)

    # Prefetch batches to improve evaluation efficiency.
    test_dataset = test_dataset.prefetch(1)

    return test_dataset

def create_confusion_matrix(labels, predictions, output_path):
    """
    Create and save a confusion matrix based on the true and predicted labels.

    Args:
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.
        output_path (str): Path of the PNG file where the figure is saved.

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
    plt.savefig(output_path)
    plt.close()

    return f"Confusion matrix saved to {output_path}"

def evaluate_model(model_path, dir_test_dataset, output_dir):
    """
    Evaluate a trained model on the test dataset.

    The function loads the trained model, processes the test dataset,
    generates predictions, and computes the confusion matrix and the
    official ICBHI sensitivity, specificity, and score.

    Args:
        model_path (str): Path to the trained Keras model.
        dir_test_dataset (str): Root directory containing the test dataset.
        output_dir (str): Directory where the confusion matrix is saved.

    Returns:
        str: Confirmation message after the evaluation is completed.
    """
    # Load the trained model without restoring its compilation configuration.
    model = load_model(model_path, compile=False, safe_mode=False)

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

    # Create and save the confusion matrix, named after the evaluated model.
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(create_confusion_matrix(labels, predictions, os.path.join(output_dir, f"confusion_matrix_{model_name}_{timestamp}.png")))

    # Compute the official ICBHI metrics.
    sensitivity, specificity, icbhi_score = icbhi_official_score(labels, predictions)

    # Display the evaluation metrics.
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ICBHI Score: {icbhi_score:.4f}")

    return "Evaluation complete!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the test dataset.')
    parser.add_argument('--model_path', type=str, default='/app/models/CustomCNN/best_model_CustomCNN.keras', help='Path to the trained Keras model')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed', help='Root directory containing the Test folder')
    parser.add_argument('--output_dir', type=str, default='/app/experiments', help='Directory where the confusion matrix is saved')
    args = parser.parse_args()

    print(evaluate_model(args.model_path, args.data_dir, args.output_dir))