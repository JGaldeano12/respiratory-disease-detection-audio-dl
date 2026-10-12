import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.utils import get_spectrogram_width, process_npy
from src.evaluation.metrics import icbhi_official_score

DEVICES = ['AKGC417L', 'Meditron', 'LittC2SE', 'Litt3200']

def load_split(dir_dataset, split):
    """
    Build the dataset of one split with an explicit, sorted file list.

    Args:
        dir_dataset (str): Root directory containing the split directories.
        split (str): Name of the split directory (e.g. Val or Test).

    Returns:
        tuple[tf.data.Dataset, np.ndarray, np.ndarray]: Preprocessed dataset,
        true class labels and recording device of each spectrogram.
    """
    paths = sorted(glob.glob(os.path.join(dir_dataset, split, '*', '*.npy')))
    devices = np.array([os.path.basename(path).split('_')[4] for path in paths])

    width = get_spectrogram_width(dir_dataset)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda x: process_npy(x, width, training=False), num_parallel_calls=1)
    dataset = dataset.batch(16, drop_remainder=False).prefetch(1)

    labels = np.concatenate([np.argmax(batch_labels, axis=1) for _, batch_labels in dataset])
    return dataset, labels, devices

def predict_probabilities(model, dataset):
    """
    Predict the class probabilities of every sample of a dataset.

    Args:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset to predict.

    Returns:
        np.ndarray: Predicted probabilities (N x 4).
    """
    return np.concatenate([model.predict(images, verbose=0) for images, _ in dataset])

def metrics_row(name, labels, predictions):
    """
    Format the official ICBHI metrics of a set of predictions.

    Args:
        name (str): Row name.
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.

    Returns:
        str: Formatted table row.
    """
    se, sp, score = icbhi_official_score(labels, predictions)
    return f"{name:<28}{se:>9.4f}{sp:>9.4f}{score:>9.4f}"

def evaluate_ensemble(model_paths, data_dir, output_dir, name):
    """
    Evaluate each model and the average of their predicted probabilities on Val and Test.

    All the models are fixed beforehand and weighted equally, so nothing is
    selected or tuned on the test set.

    Args:
        model_paths (list[str]): Paths to the trained Keras models.
        data_dir (str): Root directory containing the Val and Test directories.
        output_dir (str): Directory where the report is saved.
        name (str): Name of the ensemble, used for the report filename.

    Returns:
        str: Report with the metrics of every model and of the ensemble.
    """
    # Splits without spectrograms (e.g. no validation set in the oracle protocol) are skipped.
    splits = {split: load_split(data_dir, split) for split in ['Val', 'Test']
              if glob.glob(os.path.join(data_dir, split, '*', '*.npy'))}
    probabilities = {split: [] for split in splits}

    # Predict with one model at a time to keep the memory usage low.
    for model_path in model_paths:
        model = load_model(model_path, compile=False, safe_mode=False)
        for split, (dataset, _, _) in splits.items():
            probabilities[split].append(predict_probabilities(model, dataset))
        del model
        tf.keras.backend.clear_session()

    lines = [f"Ensemble: {name} ({len(model_paths)} models)"]
    for split, (_, labels, devices) in splits.items():
        ensemble_predictions = np.argmax(np.mean(probabilities[split], axis=0), axis=1)

        lines += ["", f"[{split}]", f"{'Model':<28}{'Se':>9}{'Sp':>9}{'ICBHI':>9}"]
        for model_path, model_probabilities in zip(model_paths, probabilities[split]):
            lines.append(metrics_row(os.path.splitext(os.path.basename(model_path))[0][-28:], labels, np.argmax(model_probabilities, axis=1)))
        lines.append(metrics_row('ENSEMBLE (mean)', labels, ensemble_predictions))

        # Per-device breakdown of the ensemble (devices with both normal and abnormal cycles only).
        lines.append("  Ensemble per device:")
        for device in DEVICES:
            mask = devices == device
            if mask.any() and (labels[mask] == 0).any() and (labels[mask] != 0).any():
                lines.append("  " + metrics_row(device, labels[mask], ensemble_predictions[mask]))

    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"ensemble_{name}.txt"), 'w') as file:
        file.write(report + "\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate an equally weighted ensemble of trained models.')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='Paths to the trained Keras models')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed', help='Root directory containing the Val and Test folders')
    parser.add_argument('--output_dir', type=str, default='/app/experiments/ensemble', help='Directory where the report is saved')
    parser.add_argument('--name', type=str, default='ensemble', help='Name of the ensemble')
    args = parser.parse_args()

    print(evaluate_ensemble(args.model_paths, args.data_dir, args.output_dir, args.name))
