import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.utils import get_spectrogram_width, process_npy
from src.evaluation.metrics import icbhi_official_score

DEVICES = ['AKGC417L', 'Meditron', 'LittC2SE', 'Litt3200']

def predict_split(model, dir_dataset, split):
    """
    Predict every spectrogram of a split and return the predictions with their devices.

    The file list is built explicitly, so that each prediction can be matched
    with the recording device stored in its filename.

    Args:
        model (tf.keras.Model): Trained model.
        dir_dataset (str): Root directory containing the split directories.
        split (str): Name of the split directory (e.g. Val or Test).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: True labels, predicted
        labels and recording device of each spectrogram.
    """
    paths = sorted(glob.glob(os.path.join(dir_dataset, split, '*', '*.npy')))

    # The device is the fifth field of the recording identifier at the start of the filename.
    devices = np.array([os.path.basename(path).split('_')[4] for path in paths])

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    width = get_spectrogram_width(dir_dataset)
    dataset = dataset.map(lambda x: process_npy(x, width, training=False), num_parallel_calls=1)
    dataset = dataset.batch(16, drop_remainder=False).prefetch(1)

    labels, predictions = [], []
    for images, batch_labels in dataset:
        predictions.append(np.argmax(model.predict(images, verbose=0), axis=1))
        labels.append(np.argmax(batch_labels, axis=1))

    return np.concatenate(labels), np.concatenate(predictions), devices

def format_row(name, labels, predictions):
    """
    Format the number of cycles and the official ICBHI metrics of a subset.

    Sensitivity (or specificity) is shown as '-' when the subset has no
    abnormal (or normal) cycles.

    Args:
        name (str): Name of the subset.
        labels (np.ndarray): True class labels.
        predictions (np.ndarray): Predicted class labels.

    Returns:
        str: Formatted table row.
    """
    n_normal = int(np.sum(labels == 0))
    n_abnormal = int(np.sum(labels != 0))

    if n_normal == 0 or n_abnormal == 0:
        se = np.mean(predictions[labels != 0] == labels[labels != 0]) if n_abnormal else None
        sp = np.mean(predictions[labels == 0] == 0) if n_normal else None
        score = None
    else:
        se, sp, score = icbhi_official_score(labels, predictions)

    fmt = lambda value: f"{value:>9.4f}" if value is not None else f"{'-':>9}"
    return f"{name:<10}{n_normal:>8}{n_abnormal:>10}{fmt(se)}{fmt(sp)}{fmt(score)}"

def evaluate_by_device(model_path, data_dir, output_dir):
    """
    Report the official ICBHI metrics per recording device on the Val and Test sets.

    This is a diagnostic of an already selected model: nothing is tuned here.

    Args:
        model_path (str): Path to the trained Keras model.
        data_dir (str): Root directory containing the Val and Test directories.
        output_dir (str): Directory where the report is saved.

    Returns:
        str: Report with one table per split.
    """
    model = load_model(model_path, compile=False, safe_mode=False)

    lines = [f"Model: {model_path}"]
    for split in ['Val', 'Test']:
        # Skip a split without spectrograms (e.g. no validation set in the oracle protocol).
        if not glob.glob(os.path.join(data_dir, split, '*', '*.npy')):
            continue
        labels, predictions, devices = predict_split(model, data_dir, split)

        lines += ["", f"[{split}]", f"{'Device':<10}{'Normal':>8}{'Abnormal':>10}{'Se':>9}{'Sp':>9}{'ICBHI':>9}"]
        for device in DEVICES:
            mask = devices == device
            if mask.any():
                lines.append(format_row(device, labels[mask], predictions[mask]))
        lines.append(format_row('All', labels, predictions))

    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    with open(os.path.join(output_dir, f"by_device_{model_name}.txt"), 'w') as file:
        file.write(report + "\n")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Report the official ICBHI metrics per recording device.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained Keras model')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed', help='Root directory containing the Val and Test folders')
    parser.add_argument('--output_dir', type=str, default='/app/experiments/by_device', help='Directory where the report is saved')
    args = parser.parse_args()

    print(evaluate_by_device(args.model_path, args.data_dir, args.output_dir))
