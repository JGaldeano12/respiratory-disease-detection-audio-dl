import numpy as np, tensorflow as tf, os

from sklearn.metrics import recall_score, confusion_matrix
from datetime import datetime

def calculate_specificity_per_class(y_true, y_pred):
    """
    Function to calculate the specificity for each class in a multi-class classification problem.
    Specificity is calculated as TN / (TN + FP) for each class, where:
        - TN (True Negatives) is the count of samples that are correctly identified as not belonging to the class.
        - FP (False Positives) is the count of samples that are incorrectly identified as belonging to the class.
    """
    # Determine the number of classes based on the unique labels in y_true.
    num_classes = np.max(y_true) + 1

    # Initialize a list to store the specificity for each class.
    specificities = []
    
    # Loop through each class to calculate its specificity.
    for i in range(num_classes):
        # Calculate true negatives and false positives for the current class.
        true_negatives = np.sum((y_true != i) & (y_pred != i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        
        # Calculate specificity for the current class, adding a small epsilon to the denominator to avoid division by zero.
        specificity = true_negatives / (true_negatives + false_positives + np.finfo(float).eps)

        # Append the calculated specificity for the current class to the list of specificities.
        specificities.append(specificity)
    
    # Return the list of specificities for all classes.
    return specificities

class ICBHI_Score_PrintingCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to calculate and print the ICBHI score, recall per class, specificity per class, and confusion matrix at the end of each epoch during training. 
    It also keeps track of the best ICBHI score achieved during training and saves the model if a new best score is found.
    """
    def __init__(self, val_dataset):
        super().__init__()
        self.best_icbhi_score = 0
        self.dir_logs = f"/app/experiments/experiments_training_logs_{datetime.now()}.txt"
        self.val_dataset = val_dataset

    def on_train_begin(self, logs=None):
        with open(self.dir_logs, 'a') as f:
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            f.write("\n".join(model_summary))

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        labels = []

        for image, label in self.val_dataset:
            pred = self.model(image, training=False)
            pred = np.argmax(pred, axis=1)
            label = np.argmax(label, axis=1)

            predictions.append(pred)
            labels.append(label)

        predictions_np = np.concatenate(predictions)
        labels_np = np.concatenate(labels)

        recall_per_class = recall_score(labels_np, predictions_np, average=None)
        specificity_per_class = calculate_specificity_per_class(labels_np, predictions_np)

        mean_recall = np.mean(recall_per_class)
        mean_specificity = np.mean(specificity_per_class)
        icbhi_score = (mean_recall + mean_specificity) / 2

        # Now, round the calculated recall and specificity to 4 decimal places.
        print("\nRecall per class: ", [round(r, 4) for r in recall_per_class])
        print("Specificity per class:", [round(s, 4) for s in specificity_per_class])
        print(f"ICBHI Score: {round(icbhi_score, 5)} - Sensitivity: {round(mean_recall, 5)} - Specificity: {round(mean_specificity, 5)}")

        if icbhi_score > self.best_icbhi_score:
            self.best_icbhi_score = icbhi_score
            self.model.save("/app/models/custom_cnn/best_model_epoch.keras")

        cm = confusion_matrix(labels_np, predictions_np)

        print("Confusion Matrix:")
        print(cm)
        print(f"Best ICBHI Score found: {round(self.best_icbhi_score, 5)}\n")

        with open(self.dir_logs, 'a') as f:
            f.write(
                f"Epoch {epoch}\n"
                f"Recall per class: {recall_per_class}\n"
                f"Specificity per class: {specificity_per_class}\n"
                f"ICBHI Score: {round(icbhi_score, 5)}\n"
                f"Sensitivity: {round(mean_recall, 5)}\n"
                f"Specificity: {round(mean_specificity, 5)}\n"
                f"Best Score: {round(self.best_icbhi_score, 5)}\n"
                f"{cm}\n"
                "--------------------------------------\n"
            )

        # Al final de on_epoch_end en ICBHI_Score_PrintingCallback:
        logs['icbhi_score'] = icbhi_score
    
    def on_train_end(self, logs=None):
        final_msg = f"\nBest ICBHI Score during training: {round(self.best_icbhi_score, 5)}"
        print(final_msg)
        with open(self.dir_logs, 'a') as f:
            f.write(final_msg + "\n")