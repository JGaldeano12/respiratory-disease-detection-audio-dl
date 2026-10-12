import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.utils import get_spectrogram_width, process_npy
from src.evaluation.metrics import icbhi_official_score

def load_split(dir_dataset, split):
    """
    Load and preprocess one split (e.g. Val or Test) without shuffling or augmentation.

    Args:
        dir_dataset (str): Root directory containing the split directories.
        split (str): Name of the split directory to load.

    Returns:
        tf.data.Dataset: Preprocessed and batched dataset.
    """
    options = tf.data.Options()
    options.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, split, '*/*'), shuffle=False).with_options(options)
    width = get_spectrogram_width(dir_dataset)
    dataset = dataset.map(lambda x: process_npy(x, width, training=False), num_parallel_calls=1)
    return dataset.batch(16, drop_remainder=False).prefetch(1)

def predict_split(model, dataset):
    """
    Obtain the predicted class probabilities and true labels of a dataset.

    Args:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Dataset to predict.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predicted probabilities (N x 4) and
        true class labels (N).
    """
    probabilities, labels = [], []

    for images, batch_labels in dataset:
        probabilities.append(model.predict(images, verbose=0))
        labels.append(np.argmax(batch_labels, axis=1))

    return np.concatenate(probabilities), np.concatenate(labels)

def apply_offset(probabilities, offset, normal_class=0):
    """
    Shift the decision towards or away from the normal class.

    The offset is added to the log-probability of the normal class before
    taking the argmax. A negative offset favours the abnormal classes
    (higher sensitivity), and a positive one favours the normal class
    (higher specificity). An offset of 0 reproduces the original predictions.

    Args:
        probabilities (np.ndarray): Predicted probabilities (N x 4).
        offset (float): Value added to the normal-class log-probability.
        normal_class (int, optional): Label of the normal class. Defaults to 0.

    Returns:
        np.ndarray: Predicted class labels after applying the offset.
    """
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    log_probabilities[:, normal_class] += offset
    return np.argmax(log_probabilities, axis=1)

def search_offset(probabilities, labels, offsets):
    """
    Find the offset that maximises the official ICBHI score.

    Ties are resolved in favour of the offset closest to 0, i.e. the smallest
    change with respect to the original model.

    Args:
        probabilities (np.ndarray): Predicted probabilities (N x 4).
        labels (np.ndarray): True class labels.
        offsets (np.ndarray): Candidate offsets.

    Returns:
        tuple[float, float]: Best offset and its ICBHI score.
    """
    scores = np.array([icbhi_official_score(labels, apply_offset(probabilities, o))[2] for o in offsets])
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    best_offset = offsets[best[np.argmin(np.abs(offsets[best]))]]
    return float(best_offset), float(scores.max())

def calibrate(model_paths, data_dir, output_dir, name=None):
    """
    Calibrate the normal-class decision offset on the validation set and report its effect on the test set.

    With several models, their predicted probabilities are averaged with equal
    weights (ensemble) before calibrating. The offset is chosen exclusively on
    the validation set. The test set is evaluated only twice: with the
    original decision (offset 0) and with the selected offset.

    Args:
        model_paths (list[str]): Paths to the trained Keras models.
        data_dir (str): Root directory containing the Val and Test directories.
        output_dir (str): Directory where the calibration summary is saved.
        name (str, optional): Name used for the summary file. Defaults to the
            filename of the first model.

    Returns:
        str: Calibration summary.
    """
    val_dataset, test_dataset = load_split(data_dir, 'Val'), load_split(data_dir, 'Test')
    val_runs, test_runs = [], []

    # Predict with one model at a time and average the probabilities of all the models.
    for model_path in model_paths:
        model = load_model(model_path, compile=False, safe_mode=False)
        val_runs.append(predict_split(model, val_dataset))
        test_runs.append(predict_split(model, test_dataset))
        del model
        tf.keras.backend.clear_session()

    val_probabilities, val_labels = np.mean([p for p, _ in val_runs], axis=0), val_runs[0][1]
    test_probabilities, test_labels = np.mean([p for p, _ in test_runs], axis=0), test_runs[0][1]

    # Select the offset on the validation set only.
    offsets = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
    best_offset, _ = search_offset(val_probabilities, val_labels, offsets)

    lines = [f"Models ({len(model_paths)}, probabilities averaged):"] + [f"  {path}" for path in model_paths]
    lines += [f"Selected offset (on validation): {best_offset:+.2f}", ""]
    lines.append(f"{'Set':<6}{'Offset':>9}{'Se':>9}{'Sp':>9}{'ICBHI':>9}")

    for split, probabilities, labels in [('Val', val_probabilities, val_labels), ('Test', test_probabilities, test_labels)]:
        for offset in [0.0, best_offset]:
            se, sp, score = icbhi_official_score(labels, apply_offset(probabilities, offset))
            lines.append(f"{split:<6}{offset:>+9.2f}{se:>9.4f}{sp:>9.4f}{score:>9.4f}")

    summary = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    name = name or os.path.splitext(os.path.basename(model_paths[0]))[0]
    with open(os.path.join(output_dir, f"calibration_{name}.txt"), 'w') as file:
        file.write(summary + "\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate the normal-class decision offset on the validation set.')
    parser.add_argument('--model_path', '--model_paths', dest='model_paths', type=str, nargs='+', required=True, help='Path(s) to the trained Keras model(s); several paths are averaged as an ensemble')
    parser.add_argument('--data_dir', type=str, default='/app/data/processed', help='Root directory containing the Val and Test folders')
    parser.add_argument('--output_dir', type=str, default='/app/experiments/calibration', help='Directory where the calibration summary is saved')
    parser.add_argument('--name', type=str, default=None, help='Name used for the summary file')
    args = parser.parse_args()

    print(calibrate(args.model_paths, args.data_dir, args.output_dir, args.name))