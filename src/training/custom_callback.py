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
    def __init__(self):
        super(ICBHI_Score_PrintingCallback, self).__init__()
        self.best_icbhi_score = 0
        self.dir_logs = f"experiments_training_logs_{datetime.now()}.txt"

    def on_train_begin(self, model, logs = None):
        """
        Method called at the beginning of training. It initializes the log file and writes the model architecture to it.
        """
        with open(self.dir_logs, 'a') as f:
            # First, obtain and store the model summary.
            model_summary_str = []
            model.summary(print_fn=lambda x: model_summary_str.append(x))
            model_summary_str = "\n".join(model_summary_str)
            resultado = model_summary_str
            f.write(resultado)   

    def on_test_end(self, test_dataset, model, epoch = None, logs = None):
        """
        Method called at the end of each epoch during training. It calculates and prints the recall per class, specificity per class, ICBHI score, 
        and confusion matrix based on the model's predictions on the validation dataset. 
        It also checks if the current ICBHI score is the best one achieved so far and saves the model if it is.
        """
        # Initialize lists to store predictions and true labels for the validation dataset.
        predictions = []
        labels = []

        # Loop through the validation dataset to get predictions and true labels.
        for image, label in test_dataset:
            # Make predictions using the model and convert them to class indices by taking the argmax along the appropriate axis.
            prediccion = model.predict(image, verbose = 0)
            prediccion = np.argmax(prediccion, axis = 1)
            label = np.argmax(label, axis = 1)

            # Save the predictions and labels for later processing.
            predictions.append(prediccion)
            labels.append(label)

        # Concat predictions and labels into single arrays for further analysis.
        predictions_np = np.concatenate(predictions)
        labels_np = np.concatenate(labels)

        # Obtain recall per class using the recall_score function from sklearn, specifying average=None to get the recall for each class separately.
        # Also, obtain specificity per class using the custom calculate_specificity_per_class function.
        recall_per_class = recall_score(labels_np, predictions_np, average=None)
        specificity_per_class = calculate_specificity_per_class(labels_np, predictions_np)

        # Calculate the total specificity by summing the specificity of each class, which will be used to compute the mean specificity later on.
        total_specificity = 0

        # Loop through the specificity for each class and add it to the total specificity to prepare for calculating the mean specificity.
        for i, specificity in enumerate(specificity_per_class):
            total_specificity += specificity

        # Print both the recall for each class and the specificity for each class to the console to provide insight during training.
        print("\nRecall per class:", recall_per_class)
        print(f"Specificity per class: {specificity_per_class}")

        # Finally, calculate the mean recall and mean specificity across all classes, and use these to compute the ICBHI score, 
        # which is the average of mean recall and mean specificity.
        mean_recall = np.mean(recall_per_class)
        mean_specificity = total_specificity / len(specificity_per_class)
        icbhi_score = (mean_recall + mean_specificity) / 2

        # Print the calculated ICBHI score, mean recall, and mean specificity to the console for monitoring the model's performance during training.
        print("ICBHI Score: ", round((mean_recall + mean_specificity) / 2, 5), " - Sensitivity:", round(mean_recall, 5), " - Specificity: ", round(mean_specificity, 5))

        # Now, check if the current ICBHI score is better than the best ICBHI score found so far during training.
        # If it is, update the best ICBHI score and save the model.
        if icbhi_score > self.best_icbhi_score:
            self.best_icbhi_score = icbhi_score
            self.model.save(f"best_model_epoch.keras")

        # Obtain the confusion matrix using the confusion_matrix function from sklearn.
        cm = confusion_matrix(labels_np, predictions_np)
        
        # Now, print the confusion matrix and the best ICBHI score found so far to the console for further insight into the model's performance.
        print("Confusion Matrix:")
        print(cm, "\n")
        print(f"Best ICBHI Score found: {round(self.best_icbhi_score, 5)}\n")

        # Finally, write the recall per class, specificity per class, ICBHI score, mean recall, mean specificity, best ICBHI score, 
        # and confusion matrix to the log file for record-keeping and analysis after training is complete.
        with open(self.ruta, 'a') as f:
            resultado = (f"Recall per class: {recall_per_class}\nSpecificity per class: {specificity_per_class} "
                        f"\nICBHI Score: {round((mean_recall + mean_specificity) / 2, 5)} "
                        f"\nSensitivity: {round(mean_recall, 5)}\nSpecificity: {round(mean_specificity, 5)}\nICBHI Best Score: {round(self.best_icbhi_score, 5)}\n{cm}\n"
                        "--------------------------------------------------------------------------------------------\n")
            f.write(resultado)
    
    def on_train_end(self, logs=None):
        """
        Method called at the end of training. It prints the best ICBHI score achieved during training to the console.
        """
        print(f"\nBest ICBHI Score during training: {round(self.best_icbhi_score, 5)}")