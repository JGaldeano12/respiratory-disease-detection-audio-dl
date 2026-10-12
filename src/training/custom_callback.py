import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score

from src.evaluation.metrics import icbhi_official_score

class ICBHI_Score_PrintingCallback(tf.keras.callbacks.Callback):
    """
    Callback to compute and monitor the ICBHI score during training.

    The callback evaluates the model on the validation dataset at the end
    of each epoch, computes the official ICBHI sensitivity, specificity,
    and score,
    prints the corresponding metrics, logs the results, and saves the model
    to a model-specific path whenever a new best selection score is achieved.

    The selection score is the mean validation ICBHI score of the last
    `smoothing_window` epochs. With a window of 1 (the default) it is the
    ICBHI score of the current epoch; larger windows avoid selecting
    checkpoints from isolated peaks of a small validation set. The best
    3-epoch mean is always tracked as well, so that runs with different
    windows can be compared with the same criterion.

    Degenerate epochs, in which the sensitivity or the specificity is 0
    (the model predicts only normal or only abnormal classes), score 0 for
    checkpoint selection and early stopping.
    """

    def __init__(self, val_dataset, model_name, smoothing_window=1):
        """
        Initialize the ICBHI score callback.

        Args:
            val_dataset (tf.data.Dataset): Dataset used to evaluate the model
                after each training epoch.
            model_name (str): Name of the model being trained (e.g. 'ResNet',
                'VGG19', 'EfficientNet', 'CustomCNN'). Used to build a
                model-specific directory and filename for the best checkpoint,
                so that different architectures don't overwrite each other's
                best model.
            smoothing_window (int, optional): Number of epochs averaged to
                compute the selection score. Defaults to 1.
        """
        super().__init__()

        self.best_icbhi_score = 0.0
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.smoothing_window = max(1, int(smoothing_window))
        self.score_history = []
        self.best_smoothed_3 = 0.0

        # Build a model-specific save path, e.g. /app/models/ResNet/best_model_ResNet.keras,
        # and make sure the target directory exists.
        self.save_dir = os.path.join('/app/models', model_name)
        self.save_path = os.path.join(self.save_dir, f'best_model_{model_name}.keras')
        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.dir_logs = (
            f"/app/experiments/"
            f"experiments_training_logs_{timestamp}.txt"
        )

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
        The per-class recall and the official ICBHI metrics are then computed. The model
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

        # Compute recall per class (for diagnostics) and the official ICBHI metrics.
        recall_per_class = recall_score(labels_np, predictions_np, average=None, zero_division=0)
        sensitivity, specificity, icbhi_score = icbhi_official_score(labels_np, predictions_np)

        # Display per-class recall and the official ICBHI metrics.
        print("\nRecall per class:", [round(value, 4) for value in recall_per_class])
        print(f"\nICBHI Score: {icbhi_score:.5f} - Sensitivity: {sensitivity:.5f} - Specificity: {specificity:.5f}")

        # A degenerate epoch (no abnormal cycle detected, or no normal cycle recognised) is a trivial
        # classifier: it scores 0 for checkpoint selection and early stopping, so that predicting a single
        # group (e.g. ICBHI 0.5 when everything is "normal") can never block the training.
        degenerate = sensitivity == 0.0 or specificity == 0.0
        if degenerate:
            print("Degenerate epoch (sensitivity or specificity is 0): not eligible for checkpoint selection.")

        # Selection score: mean ICBHI score of the last `smoothing_window` epochs (the current one if the window is 1).
        self.score_history.append(0.0 if degenerate else icbhi_score)
        selection_score = float(np.mean(self.score_history[-self.smoothing_window:]))

        # The 3-epoch mean is always tracked, so that every run can be compared with the same criterion.
        self.best_smoothed_3 = max(self.best_smoothed_3, float(np.mean(self.score_history[-3:])))

        if self.smoothing_window > 1:
            print(f"Selection score ({self.smoothing_window}-epoch mean): {selection_score:.5f}")

        # Save the model when a new best selection score is achieved.
        if selection_score > self.best_icbhi_score:
            self.best_icbhi_score = selection_score
            self.model.save(self.save_path)

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
                f"ICBHI Score: {icbhi_score:.5f}\n"
                f"Sensitivity: {sensitivity:.5f}\n"
                f"Specificity: {specificity:.5f}\n"
                f"Selection Score: {selection_score:.5f}\n"
                f"Best Score: {self.best_icbhi_score:.5f}\n"
                f"Confusion Matrix:\n{cm}\n"
                "--------------------------------------\n"
            )

        print(
            "\n----------------------------------------------------------------------------\n"
        )

        # Store the selection score (used by early stopping) and the raw ICBHI score for other callbacks.
        if logs is not None:
            logs['icbhi_score'] = selection_score
            logs['icbhi_score_raw'] = icbhi_score

    def on_train_end(self, logs=None):
        """
        Display and save the best ICBHI score obtained during training.

        Args:
            logs (dict, optional): Dictionary containing training logs.
        """
        final_msg = (f"\nBest ICBHI Score during training: {self.best_icbhi_score:.5f}"
                     f"\nBest 3-epoch mean ICBHI Score during training: {self.best_smoothed_3:.5f}")

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