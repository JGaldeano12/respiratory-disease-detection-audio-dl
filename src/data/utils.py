import glob
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

def get_label(file_path):
    """
    Extract the class label from the file path.

    The function maps the class name contained in the file path to its
    corresponding integer label.

    Args:
        file_path (tf.Tensor): Path to the NumPy spectrogram file.

    Returns:
        tf.Tensor: Integer-encoded class label.
    """
    # The class label is the name of the parent directory of the file.
    label_str = tf.strings.split(file_path, os.path.sep)[-2]

    keys = tf.constant(['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=4
    )

    label = table.lookup(label_str)
    return label

def get_spectrogram_width(dir_dataset):
    """
    Read the spectrogram width (number of time frames) from the dataset.

    The width depends on the segment duration used to generate the dataset
    (e.g. 251 frames for 8 s at 8000 Hz with a hop length of 256), so it is
    read from the first spectrogram found instead of being hard-coded.

    Args:
        dir_dataset (str): Root directory containing the split directories.

    Returns:
        int: Width of the spectrograms.
    """
    paths = sorted(glob.glob(os.path.join(dir_dataset, '*', '*', '*.npy')))
    if not paths:
        raise FileNotFoundError(f"No spectrograms found in {dir_dataset}")
    return int(np.load(paths[0], mmap_mode='r').shape[1])

def load_npy(path, width):
    """
    Load a spectrogram from a NumPy file.

    The function loads the spectrogram from the provided file path,
    verifies its expected width, and converts it to a 32-bit floating-point
    NumPy array.

    Args:
        path (tf.Tensor): Path to the NumPy spectrogram file.
        width (tf.Tensor): Expected spectrogram width.

    Returns:
        np.ndarray: Loaded spectrogram as a 32-bit floating-point array.
    """
    path = path.numpy().decode("utf-8")
    width = int(width.numpy())
    spec = np.load(path)
    assert spec.shape[1] == width, f"Unexpected spectrogram width: {spec.shape[1]} (expected {width})"
    return spec.astype(np.float32)

def spec_augment_tf(spec, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    """
    Apply SpecAugment to a spectrogram using frequency and time masking.

    The function randomly masks frequency bands and temporal regions of
    the input spectrogram to augment the training data.

    Args:
        spec (tf.Tensor): Input spectrogram to augment.
        time_mask_param (int): Maximum size of each time mask.
        freq_mask_param (int): Maximum size of each frequency mask.
        num_time_masks (int): Number of time masks to apply.
        num_freq_masks (int): Number of frequency masks to apply.

    Returns:
        tf.Tensor: Augmented spectrogram.
    """
    H = tf.shape(spec)[0]
    W = tf.shape(spec)[1]

    for _ in range(num_freq_masks):
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, H - f + 1, dtype=tf.int32)
        mask = tf.ones_like(spec)

        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.reshape(tf.range(f0, f0 + f), (-1, 1)),
            updates=tf.zeros((f, W, tf.shape(spec)[2]))
        )

        spec = spec * mask

    for _ in range(num_time_masks):
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, W - t + 1, dtype=tf.int32)

        time_mask = tf.concat([
            tf.ones((H, t0, tf.shape(spec)[2])),
            tf.zeros((H, t, tf.shape(spec)[2])),
            tf.ones((H, W - t0 - t, tf.shape(spec)[2]))
        ], axis=1)

        spec = spec * time_mask

    return spec

# Mel bins of the 1.5-2 kHz band (n_mels=128, fmin=50 Hz, fmax=2500 Hz; row 0 is the lowest frequency).
# Recordings from the Littmann 3200 stethoscope have no energy in this band.
BLANK_BAND_ROWS = (93, 113)

def mask_blank_band(spec, probability=0.3):
    """
    Randomly blank the 1.5-2 kHz band of a spectrogram.

    With the given probability, the mel bins of the 1.5-2 kHz band are set to
    the minimum value of the spectrogram, simulating the empty band of the
    Littmann 3200 recordings. This teaches the model that an empty band is a
    property of the device and not a sign of pathology.

    Args:
        spec (tf.Tensor): Spectrogram of shape (128, width, 1).
        probability (float, optional): Probability of blanking the band.
            Defaults to 0.3.

    Returns:
        tf.Tensor: Spectrogram, with the band blanked or unchanged.
    """
    low, high = BLANK_BAND_ROWS
    rows = tf.range(tf.shape(spec)[0])
    band = tf.reshape(tf.logical_and(rows >= low, rows < high), (-1, 1, 1))
    blanked = tf.where(band, tf.reduce_min(spec), spec)

    return tf.cond(tf.random.uniform([]) < probability, lambda: blanked, lambda: spec)

def get_class_weights_from_paths(dir_dataset):
    """
    Compute class weights from the training dataset file paths.

    The function extracts the class labels from the training directory
    structure and computes balanced class weights to account for class
    imbalance.

    Args:
        dir_dataset (str): Path to the root directory containing the
            training dataset.

    Returns:
        tuple:
            A tuple containing a dictionary mapping class indices to their
            corresponding weights and a NumPy array containing the class
            weights.
    """
    train_paths = sorted(glob.glob(os.path.join(dir_dataset, 'Train/*/*')))

    label_map = {'Healthy': 0, 'Crackle': 1, 'Wheeze': 2, 'Wheeze & Crackle': 3}

    labels = []

    for path in train_paths:
        folder = path.split(os.path.sep)[-2]
        labels.append(label_map[folder])

    labels = np.array(labels)
    classes = np.unique(labels)

    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)

    return dict(enumerate(class_weights)), np.array(class_weights)

def process_npy(file_path, width, training=True, band_mask=False):
    """
    Load and preprocess a NumPy spectrogram and its corresponding label.

    The function extracts and one-hot encodes the class label, loads the
    spectrogram from the NumPy file, and optionally applies SpecAugment
    during training.

    Args:
        file_path (tf.Tensor): Path to the NumPy spectrogram file.
        width (int): Spectrogram width, as returned by get_spectrogram_width().
        training (bool): Whether the spectrogram is being processed for
            training. If True, SpecAugment may be applied.
        band_mask (bool): Whether to randomly blank the 1.5-2 kHz band
            during training (see mask_blank_band). Defaults to False.

    Returns:
        tuple:
            A tuple containing the processed spectrogram and its one-hot
            encoded class label.
    """
    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)

    spec = tf.py_function(load_npy, [file_path, width], tf.float32)
    spec.set_shape([128, width, 1])

    if training:
        spec = tf.cond(
            tf.random.uniform([]) < 0.8,
            lambda: spec_augment_tf(spec, time_mask_param=20, freq_mask_param=15, num_time_masks=2, num_freq_masks=2),
            lambda: spec
        )

    if training and band_mask:
        spec = mask_blank_band(spec)

    return spec, label

def load_datasets(dir_dataset, seed=12345, band_mask=False):
    """
    Load and prepare the training, validation and testing datasets.

    The function creates TensorFlow data pipelines for the training,
    validation and testing datasets, including shuffling, preprocessing,
    batching, and prefetching. Only the training dataset is shuffled (using
    the provided seed) and augmented with SpecAugment, while deterministic
    execution is enabled for all datasets.

    Args:
        dir_dataset (str): Path to the root directory containing the
            Train, Val and Test directories.
        seed (int): Random seed used for shuffling the training dataset.
        band_mask (bool): Whether to randomly blank the 1.5-2 kHz band of the
            training spectrograms. Defaults to False.

    Returns:
        tuple:
            A tuple containing the prepared training, validation and
            testing TensorFlow datasets. A split without spectrograms is
            returned as None.
    """
    options = tf.data.Options()
    options.experimental_deterministic = True

    # Spectrogram width, which depends on the segment duration used to generate the dataset.
    width = get_spectrogram_width(dir_dataset)

    # Training dataset: shuffled, augmented, and batched dropping the last incomplete batch.
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle=False).with_options(options)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: process_npy(x, width, training=True, band_mask=band_mask), num_parallel_calls=1)
    train_dataset = train_dataset.batch(16, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)

    # Validation and testing datasets: neither shuffled nor augmented, and every sample is kept.
    eval_datasets = []
    for split in ['Val', 'Test']:
        # A split without spectrograms (e.g. no validation set in the oracle protocol) is returned as None.
        if not glob.glob(os.path.join(dir_dataset, split, '*', '*')):
            eval_datasets.append(None)
            continue
        dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, split, '*/*'), shuffle=False).with_options(options)
        dataset = dataset.map(lambda x: process_npy(x, width, training=False), num_parallel_calls=1)
        dataset = dataset.batch(16, drop_remainder=False)
        dataset = dataset.prefetch(1)
        eval_datasets.append(dataset)

    val_dataset, test_dataset = eval_datasets

    return train_dataset, val_dataset, test_dataset
