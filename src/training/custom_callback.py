import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, recall_score
from datetime import datetime

class ICBHI_Score_PrintingCallback(tf.keras.callbacks.Callback):
    """
    Callback to compute and monitor the ICBHI score during training.

    The callback evaluates the model on the validation dataset at the end
    of each epoch, computes recall, specificity, and the ICBHI score,
    prints the corresponding metrics, logs the results, and saves the model
    whenever a new best ICBHI score is achieved.
    """

    def __init__(self, val_dataset):
        """
        Initialize the ICBHI score callback.

        Args:
            val_dataset (tf.data.Dataset): Dataset used to evaluate the model
                after each training epoch.
        """
        super().__init__()

        self.best_icbhi_score = 0.0
        self.val_dataset = val_dataset

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.dir_logs = (
            f"/app/experiments/"
            f"experiments_training_logs_{timestamp}.txt"
        )

    def calculate_specificity_per_class(self, y_true, y_pred):
        """
        Calculate the specificity for each class.

        Args:
            y_true (np.ndarray): True class labels.
            y_pred (np.ndarray): Predicted class labels.

        Returns:
            np.ndarray: Specificity value for each class.
        """
        num_classes = np.max(y_true) + 1
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        specificity_per_class = []

        for i in range(num_classes):
            true_negatives = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            false_positives = np.sum(cm[:, i]) - cm[i, i]
            specificity = true_negatives / (true_negatives + false_positives + np.finfo(float).eps)
            specificity_per_class.append(specificity)

        return np.array(specificity_per_class)

    def on_train_begin(self, logs=None):
        """
        Save the model summary to the training log file.

        Args:
            logs (dict, optional): Dictionary containing training logs.
        """
        model_summary = []
        self.model.summary(print_fn=lambda line: model_summary.append(line))

        with open(self.dir_logs, 'a') as file:
            file.write("\n".join(model_summary))
            file.write("\n\n")

    def on_epoch_end(self, epoch, logs=None):
        """
        Evaluate the model and compute the ICBHI score after each epoch.

        The validation dataset is used to obtain predictions and true labels.
        Recall, specificity, and the ICBHI score are then computed. The model
        is saved whenever a new best ICBHI score is achieved.

        Args:
            epoch (int): Index of the current training epoch.
            logs (dict, optional): Dictionary containing the metrics generated
                during the current epoch.
        """
        predictions = []
        labels = []

        # Generate predictions on the validation dataset.
        for images, batch_labels in self.val_dataset:
            batch_predictions = self.model.predict(images, verbose=0)
            batch_predictions = np.argmax(batch_predictions, axis=1)
            batch_labels = np.argmax(batch_labels, axis=1)

            predictions.append(batch_predictions)
            labels.append(batch_labels)

        # Concatenate all batches into single arrays.
        predictions_np = np.concatenate(predictions)
        labels_np = np.concatenate(labels)

        # Compute recall and specificity for each class.
        recall_per_class = recall_score(labels_np, predictions_np, average=None, zero_division=0)
        specificity_per_class = self.calculate_specificity_per_class(labels_np, predictions_np)

        # Compute mean recall and specificity.
        mean_recall = np.mean(recall_per_class)
        mean_specificity = np.mean(specificity_per_class)

        # Compute the ICBHI score.
        icbhi_score = (mean_recall + mean_specificity) / 2

        # Display per-class and overall metrics.
        print("\nRecall per class:", [round(value, 4) for value in recall_per_class])
        print("\nSpecificity per class:", [round(value, 4) for value in specificity_per_class])
        print(f"\nICBHI Score: {icbhi_score:.5f}" f" - Sensitivity: {mean_recall:.5f}" f" - Specificity: {mean_specificity:.5f}")

        # Save the model when a new best ICBHI score is achieved.
        if icbhi_score > self.best_icbhi_score:
            self.best_icbhi_score = icbhi_score
            self.model.save("/app/models/custom_cnn/best_model_epoch.keras")

        # Compute and display the confusion matrix.
        cm = confusion_matrix(labels_np, predictions_np)

        print("\nConfusion Matrix:")
        print(cm)

        print(f"\nBest ICBHI Score found: " f"{self.best_icbhi_score:.5f}\n")

        # Save the epoch results to the training log.
        with open(self.dir_logs, 'a') as file:
            file.write(
                f"Epoch {epoch + 1}\n"
                f"Recall per class: {recall_per_class}\n"
                f"Specificity per class: {specificity_per_class}\n"
                f"ICBHI Score: {icbhi_score:.5f}\n"
                f"Sensitivity: {mean_recall:.5f}\n"
                f"Specificity: {mean_specificity:.5f}\n"
                f"Best Score: {self.best_icbhi_score:.5f}\n"
                f"Confusion Matrix:\n{cm}\n"
                "--------------------------------------\n"
            )

        print(
            "\n----------------------------------------------------------------------------\n"
        )

        # Store the ICBHI score so that other callbacks can access it.
        if logs is not None:
            logs['icbhi_score'] = icbhi_score

    def on_train_end(self, logs=None):
        """
        Display and save the best ICBHI score obtained during training.

        Args:
            logs (dict, optional): Dictionary containing training logs.
        """
        final_msg = (f"\nBest ICBHI Score during training: " f"{self.best_icbhi_score:.5f}")

        print(final_msg)

        with open(self.dir_logs, 'a') as file:
            file.write(final_msg + "\n")

class ICBHIEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_delta=1e-6):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get('icbhi_score', -np.inf)

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True
                print(f"\nEarly stopping: best ICBHI Score = {self.best_score:.5f}")