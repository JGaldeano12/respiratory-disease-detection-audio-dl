import seaborn as sns, numpy as np, matplotlib.pyplot as plt, datetime, os

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, recall_score

def evaluate_model(model_path, test_dataset):
    """
    Function to evaluate a trained model on a test dataset and compute the confusion matrix and recall score.
    """
    # Load the trained model from the specified file path.
    model = load_model(model_path)

    # Initialize lists to store the true labels and predicted labels for the test dataset.
    predictions = []
    labels = []

    # Iterate through the test dataset and make predictions using the loaded model.
    for img, label in test_dataset:
        # Predict and obtain the predicted class by taking the argmax of the model's output.
        pred = model.predict(img, verbose = 0)
        pred = np.argmax(pred, axis = 1)
        label = np.argmax(label, axis = 1)

        # Append the predicted and true labels to their respective lists.
        predictions.append(pred)
        labels.append(label)

def create_confusion_matrix(labels, predictions):
    """
    Function to create and save a confusion matrix based on the true labels and predicted labels.
    """
    # Obtain the confusion matrix by comparing the true labels and predicted labels.
    cm = confusion_matrix(np.concatenate(labels), np.concatenate(predictions))

    # Create and save a heatmap of the confusion matrix using seaborn for better visualization.
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Return a log message indicating that the confusion matrix has been saved.
    return "Confusion matrix saved!"

def calculate_specificity_per_class(labels, predictions):
    """
    Function to calculate the specificity for each class based on the true labels and predicted labels.
    """
    # Concatenate the true labels and predicted labels into single arrays for metric computation.
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # Determine the number of classes based on the maximum label value in the true labels, and initialize a list to store the specificity for each class.
    num_classes = np.max(labels) + 1
    specificities = []
    
    # Iterate through each class and calculate the true negatives and false positives to compute the specificity for that class.
    for i in range(num_classes):
        # Calculate true negatives and false positives for the current class by comparing the true labels and predicted labels.
        true_negatives = np.sum((labels != i) & (predictions != i))
        false_positives = np.sum((labels != i) & (predictions == i))
        
        # Calculate specificity for the current class, with a small epsilon added to the denominator to prevent division by zero.
        specificity = true_negatives / (true_negatives + false_positives + np.finfo(float).eps)
        specificities.append(specificity)
    
    # Return the list of specificities for each class.
    return specificities

def compute_recall_specificity_score(labels, predictions):
    """
    Function to compute recall, specificity, and the ICBHI score based on the true labels and predicted labels.
    """
    # Concatenate the true labels and predicted labels into single arrays for metric computation.
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # Compute recall per class:
    recall_per_class = recall_score(labels, predictions, average=None)

    # Compute specificity per class using the previously defined function.
    specificity_per_class = calculate_specificity_per_class(labels, predictions)

    # Compute the ICBHI score as the average of recall and specificity across all classes.
    icbhi_score = ( np.mean(recall_per_class) + np.mean(specificity_per_class) ) / 2

    # Return the computed recall, specificity, and ICBHI score as a log.
    return f"Recall: {np.mean(recall_per_class):.4f}, Specificity: {np.mean(specificity_per_class):.4f}, ICBHI Score: {icbhi_score:.4f}"