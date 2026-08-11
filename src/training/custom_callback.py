import numpy as np, tensorflow as tf, os

from sklearn.metrics import recall_score, confusion_matrix
from datetime import datetime


def calculate_specificity_per_class(y_true, y_pred, num_classes):
    """
    Calcula la especificidad por clase reconstruida (Healthy, Crackle, Wheeze, Wheeze & Crackle)
    a partir de predicciones multietiqueta [crackle, wheeze].

    Especificidad = TN / (TN + FP) para cada clase.
    """
    specificities = []

    for i in range(num_classes):
        true_negatives  = np.sum((y_true != i) & (y_pred != i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        specificity = true_negatives / (true_negatives + false_positives + np.finfo(float).eps)
        specificities.append(specificity)

    return specificities


def multilabel_to_4class(crackle, wheeze):
    """
    Reconstruye el índice de clase original (0-3) a partir de las dos etiquetas binarias.

    [0, 0] → 0 (Healthy)
    [1, 0] → 1 (Crackle)
    [0, 1] → 2 (Wheeze)
    [1, 1] → 3 (Wheeze & Crackle)
    """
    return crackle * 1 + wheeze * 2


class ICBHI_Score_PrintingCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback adaptado a clasificación multietiqueta [crackle, wheeze].

    Las predicciones sigmoid se umbralean en 0.5, se reconstruyen las 4 clases
    originales y se calcula el ICBHI Score estándar (Se + Sp) / 2.
    """
    def __init__(self, val_dataset, threshold=0.5):
        super().__init__()
        self.best_icbhi_score = 0
        self.threshold = threshold
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
            # Predicciones sigmoid → umbralear → binario
            pred = self.model(image, training=False).numpy()
            pred_binary = (pred >= self.threshold).astype(int)

            # Etiquetas ya son binarias [crackle, wheeze]
            label_binary = label.numpy().astype(int)

            predictions.append(pred_binary)
            labels.append(label_binary)

        predictions_np = np.concatenate(predictions, axis=0)  # (N, 2)
        labels_np      = np.concatenate(labels,      axis=0)  # (N, 2)

        # Reconstruir índices de clase 0-3 para métricas ICBHI estándar
        pred_4class  = multilabel_to_4class(predictions_np[:, 0], predictions_np[:, 1])
        label_4class = multilabel_to_4class(labels_np[:, 0],      labels_np[:, 1])

        # Métricas sobre las 4 clases reconstruidas
        recall_per_class      = recall_score(label_4class, pred_4class, average=None, labels=[0,1,2,3], zero_division=0)
        specificity_per_class = calculate_specificity_per_class(label_4class, pred_4class, num_classes=4)

        mean_recall      = np.mean(recall_per_class)
        mean_specificity = np.mean(specificity_per_class)
        icbhi_score      = (mean_recall + mean_specificity) / 2

        # Imprimir métricas por clase
        class_names = ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']
        print("\nRecall per class:")
        for name, r in zip(class_names, recall_per_class):
            print(f"  {name}: {round(r, 4)}")

        print("\nSpecificity per class:")
        for name, s in zip(class_names, specificity_per_class):
            print(f"  {name}: {round(s, 4)}")

        print(f"\nICBHI Score: {round(icbhi_score, 5)} - Sensitivity: {round(mean_recall, 5)} - Specificity: {round(mean_specificity, 5)}")

        # Guardar mejor modelo
        if icbhi_score > self.best_icbhi_score:
            self.best_icbhi_score = icbhi_score
            self.model.save("/app/models/custom_cnn/best_model_epoch.keras")

        # Matriz de confusión sobre 4 clases reconstruidas
        cm = confusion_matrix(label_4class, pred_4class, labels=[0,1,2,3])
        print(f"\nConfusion Matrix (Healthy / Crackle / Wheeze / Wheeze & Crackle):")
        print(f"{cm}")
        print(f"\nBest ICBHI Score found: {round(self.best_icbhi_score, 5)}\n")

        # Log a fichero
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

        print("\n----------------------------------------------------------------------------\n")

        # Propagar icbhi_score al EarlyStopping
        logs['icbhi_score'] = icbhi_score

    def on_train_end(self, logs=None):
        final_msg = f"\nBest ICBHI Score during training: {round(self.best_icbhi_score, 5)}"
        print(final_msg)
        with open(self.dir_logs, 'a') as f:
            f.write(final_msg + "\n")