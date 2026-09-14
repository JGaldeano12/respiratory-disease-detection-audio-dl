import os, numpy as np, tensorflow as tf, glob
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
    elements = tf.strings.split(file_path, os.path.sep)
    label_str = elements[5]

    keys = tf.constant(['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=4
    )

    label = table.lookup(label_str)
    return label

def load_npy(path):
    """
    Load a spectrogram from a NumPy file.

    The function loads the spectrogram from the provided file path,
    verifies its expected width, and converts it to a 32-bit floating-point
    NumPy array.

    Args:
        path (tf.Tensor): Path to the NumPy spectrogram file.

    Returns:
        np.ndarray: Loaded spectrogram as a 32-bit floating-point array.
    """
    path = path.numpy().decode("utf-8")
    spec = np.load(path)
    assert spec.shape[1] == 251, f"Unexpected spectrogram width: {spec.shape[1]}"
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

def process_npy(file_path, training=True):
    """
    Load and preprocess a NumPy spectrogram and its corresponding label.

    The function extracts and one-hot encodes the class label, loads the
    spectrogram from the NumPy file, and optionally applies SpecAugment
    during training.

    Args:
        file_path (tf.Tensor): Path to the NumPy spectrogram file.
        training (bool): Whether the spectrogram is being processed for
            training. If True, SpecAugment may be applied.

    Returns:
        tuple:
            A tuple containing the processed spectrogram and its one-hot
            encoded class label.
    """
    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)

    spec = tf.py_function(load_npy, [file_path], tf.float32)
    spec.set_shape([128, 251, 1])

    if training:
        spec = tf.cond(
            tf.random.uniform([]) < 0.8,
            lambda: spec_augment_tf(spec, time_mask_param=20, freq_mask_param=15, num_time_masks=2, num_freq_masks=2),
            lambda: spec
        )

    return spec, label

def load_datasets(dir_dataset, seed=12345):
    """
    Load and prepare the training and testing datasets.

    The function creates TensorFlow data pipelines for the training and
    testing datasets, including shuffling, preprocessing, batching, and
    prefetching. The training dataset is shuffled using the provided seed,
    while deterministic execution is enabled for both datasets.

    Args:
        dir_dataset (str): Path to the root directory containing the
            training and testing datasets.
        seed (int): Random seed used for shuffling the training dataset.

    Returns:
        tuple:
            A tuple containing the prepared training and testing
            TensorFlow datasets.
    """
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle=False)
    test_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'), shuffle=False)

    options = tf.data.Options()
    options.experimental_deterministic = True

    train_dataset = train_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: process_npy(x, training=True), num_parallel_calls=1)
    train_dataset = train_dataset.batch(128, drop_remainder=True)
    train_dataset = train_dataset.prefetch(1)

    test_dataset = test_dataset.map(lambda x: process_npy(x, training=False), num_parallel_calls=1)
    test_dataset = test_dataset.batch(128, drop_remainder=True)
    test_dataset = test_dataset.prefetch(1)

    return train_dataset, test_dataset