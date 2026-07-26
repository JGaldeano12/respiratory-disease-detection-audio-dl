import tensorflow as tf, numpy as np, os, random, glob, argparse

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.models.efficientnet import create_efficientnet_model
from src.training.custom_callback import ICBHI_Score_PrintingCallback

class ICBHIEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, min_delta=1e-6):
        super().__init__()

        # Init the early stopping parameters: patience (number of epochs to wait for an improvement before stopping) 
        # and min_delta (minimum improvement required to reset the patience counter).
        self.patience = patience
        self.min_delta = min_delta

        # Initialize the best score to negative infinity, the wait counter to 0, and the best weights to None.
        self.best_score = -np.inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Get the current ICBHI Score from the logs.
        current_score = logs.get('icbhi_score', -np.inf)

        # Save weights at first epoch
        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        # If the current ICBHI Score is better than the best score by at least min_delta, update the best score, 
        # reset the wait counter, and save the current model weights.
        if current_score > self.best_score + self.min_delta:
            # Update the best score and reset the wait counter
            self.best_score = current_score
            self.wait = 0

            # Save the current model weights as the best weights
            self.best_weights = self.model.get_weights()

        # If the current ICBHI Score does not improve sufficiently, increment the wait counter. If the wait counter exceeds the patience,
        # restore the model weights to the best weights and stop training.
        else:
            # Increment the wait counter if there is no sufficient improvement
            self.wait += 1

            # If the wait counter exceeds the patience, restore the best weights and stop training
            if self.wait >= self.patience:
                # Restore the model weights to the best weights and stop training
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True

                # Print the best ICBHI Score achieved during training when early stopping is triggered.
                print(f"\nEarly stopping: " f"best ICBHI Score = {self.best_score:.5f}")

def focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)
        if alpha is not None:
            alpha_t = tf.reduce_sum(
                y_true * tf.constant(alpha, dtype=tf.float32),
                axis=-1, keepdims=True
            )
            focal_weight = alpha_t * focal_weight
        return tf.reduce_sum(focal_weight * ce, axis=-1)
    return loss

def get_label(file_path):
    """
    Function to extract the label from the file path, specifically from the folder.
    """
    # Split the file path into its components and extract the label string from the appropriate position (5th index).
    elements = tf.strings.split(file_path, os.path.sep)
    label_str = elements[5]

    # Create a lookup table to convert the label strings into numerical labels.
    keys = tf.constant(['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    # Create a static hash table for the label lookup, with a default value of 4 for any unknown labels.
    table = tf.lookup.StaticHashTable( tf.lookup.KeyValueTensorInitializer(keys, values), default_value=4)

    # Use the lookup table to convert the label string into its corresponding numerical label.
    label = table.lookup(label_str)

    # Return the numerical label corresponding to the class of the image.
    return label

def spec_augment_tf(spec, time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2):
    """
    Apply SpecAugment to a spectrogram using frequency and time masking.

    IMPORTANT:
    This function relies on TensorFlow random operations. To ensure
    reproducibility, a global TensorFlow seed must be fixed beforehand
    using tf.random.set_seed(...).
    """

    # Extract spectrogram dimensions.
    H = tf.shape(spec)[0]
    W = tf.shape(spec)[1]

    # ============================================================
    # FREQUENCY MASKING
    # Randomly masks frequency bands in the spectrogram.
    # ============================================================
    for _ in range(num_freq_masks):

        # Randomly select mask size and starting frequency index.
        f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, H - f + 1, dtype=tf.int32)

        # Create an initial mask filled with ones.
        mask = tf.ones_like(spec)

        # Replace selected frequency region with zeros.
        mask = tf.tensor_scatter_nd_update(
            mask,
            indices=tf.reshape(tf.range(f0, f0 + f), (-1, 1)),
            updates=tf.zeros((f, W, tf.shape(spec)[2]))
        )

        # Apply frequency mask.
        spec = spec * mask

    # ============================================================
    # TIME MASKING
    # Randomly masks temporal regions in the spectrogram.
    # ============================================================
    for _ in range(num_time_masks):

        # Randomly select mask size and starting time index.
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, W - t + 1, dtype=tf.int32)

        # Build a deterministic time mask:
        # ones -> zeros -> ones
        time_mask = tf.concat([
            tf.ones((H, t0, tf.shape(spec)[2])),
            tf.zeros((H, t, tf.shape(spec)[2])),
            tf.ones((H, W - t0 - t, tf.shape(spec)[2]))
        ], axis=1)

        # Apply time mask.
        spec = spec * time_mask

    # Return augmented spectrogram.
    return spec

def get_class_weights_from_paths(dir_dataset):
    """
    Compute class weights directly from the training file paths without using the TensorFlow pipeline.
    """
    # Retrieve all training file paths and sort them to ensure
    # deterministic ordering across runs and environments.
    train_paths = sorted(glob.glob(os.path.join(dir_dataset, 'Train/*/*')))

    # Map class names to integer labels.
    label_map = {'Healthy': 0, 'Crackle': 1, 'Wheeze': 2, 'Wheeze & Crackle': 3}
    
    # Extract labels from parent folder names.
    labels = []

    # Iterate through each training file path, extract the class label from the parent folder name, 
    # and convert it to a numeric label using the label_map.
    for path in train_paths:
        # Parent folder corresponds to class name.
        folder = path.split(os.path.sep)[-2]

        # Convert class name into numeric label.
        labels.append(label_map[folder])

    # Convert labels to NumPy array.
    labels = np.array(labels)

    # Retrieve unique class indices.
    classes = np.unique(labels)

    # Compute balanced class weights.
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)

    # Return both dictionary and array representations.
    return dict(enumerate(class_weights)), np.array(class_weights)

def load_npy(path):
    path = path.numpy().decode("utf-8")
    spec = np.load(path)
    assert spec.shape[1] == 129, f"Unexpected spectrogram width: {spec.shape[1]}"
    return spec.astype(np.float32)

def process_npy(file_path, training=True):
    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)
    spec = tf.py_function(load_npy, [file_path], tf.float32)
    spec.set_shape([128, 129, 1])

    if training:
        spec = tf.cond(
            tf.random.uniform([]) < 0.8,
            lambda: spec_augment_tf(spec, time_mask_param=10, freq_mask_param=8, num_time_masks=2, num_freq_masks=2),
            lambda: spec
        )

    return spec, label

def load_datasets(dir_dataset, seed=12345):
    """
    Load and prepare the training and testing datasets.

    IMPORTANT:
    To ensure full reproducibility:
    - Dataset shuffling uses a fixed seed.
    - Parallel processing with AUTOTUNE is disabled.
    - Deterministic tf.data execution is explicitly enabled.
    """
    # ============================================================
    # CREATE DATASET FILE LISTS
    # ============================================================
    # Load training and testing file paths without internal shuffling.
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle=False)
    test_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'), shuffle=False)

    # ============================================================
    # ENABLE DETERMINISTIC EXECUTION
    # ============================================================
    # Force deterministic ordering and execution in tf.data.
    options = tf.data.Options()
    options.experimental_deterministic = True

    # Apply the deterministic options to both datasets.
    train_dataset = train_dataset.with_options(options)
    test_dataset = test_dataset.with_options(options)

    # ============================================================
    # TRAINING DATASET PIPELINE
    # ============================================================
    # Shuffle dataset using a fixed seed to ensure reproducibility.
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=seed, reshuffle_each_iteration=True)

    # Process spectrograms sequentially to avoid non-deterministic
    # execution caused by parallel workers.
    train_dataset = train_dataset.map(lambda x: process_npy(x, training=True), num_parallel_calls=1)

    # Create batches while dropping incomplete final batches.
    train_dataset = train_dataset.batch(128, drop_remainder=True)

    # Prefetch a single batch deterministically.
    train_dataset = train_dataset.prefetch(1)

    # ============================================================
    # TEST DATASET PIPELINE
    # ============================================================
    # Process test samples sequentially for deterministic execution.
    test_dataset = test_dataset.map(lambda x: process_npy(x, training=False), num_parallel_calls=1)

    # Create test batches.
    test_dataset = test_dataset.batch(128, drop_remainder=False)

    # Deterministic prefetching.
    test_dataset = test_dataset.prefetch(1)

    # Return prepared datasets.
    return train_dataset, test_dataset

def train(model, base_model, train_dataset, val_dataset):
    """
    Train the model in three stages:
    1. Initial training with frozen EfficientNet backbone.
    2. Partial fine-tuning of deeper EfficientNet layers.
    3. Full fine-tuning of the entire backbone.

    IMPORTANT:
    Reproducibility depends on:
    - Global TensorFlow seed configuration.
    - Deterministic tf.data pipeline execution.
    - Deterministic TensorFlow operations enabled beforehand.
    """

    # ============================================================
    # CUSTOM VALIDATION CALLBACK
    # ============================================================

    # Compute and display the ICBHI score during validation.
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset)

    # ============================================================
    # PHASE 1 — INITIAL TRAINING
    # ============================================================

    print("PHASE 1: INITIAL TRAINING (lr=1e-3)")

    # Early stopping based on validation ICBHI score.
    early_stopping_1 = ICBHIEarlyStopping(patience=15)

    # Compile model using Adam optimizer and focal loss.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0), loss=focal_loss(gamma=1.5))

    # Train only the classification head while the backbone remains frozen.
    model.fit(train_dataset, epochs=1, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping_1])

    # ============================================================
    # PHASE 2 — PARTIAL FINE-TUNING
    # ============================================================

    print("PHASE 2: REFINE TRAINING (lr=1e-4)")

    # Unfreeze the EfficientNet backbone.
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # for layer in base_model.layers:
    #     if isinstance(layer, tf.keras.layers.BatchNormalization):
    #         layer.trainable = False

    # Early stopping for refinement phase.
    early_stopping_2 = ICBHIEarlyStopping(patience=50)

    # Recompile model after changing trainable layers.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0), loss=focal_loss(gamma=1.5))

    # Fine-tune deeper layers of the backbone.
    model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping_2])

    # ============================================================
    # PHASE 3 — FULL FINE-TUNING
    # ============================================================

    print("PHASE 3: FINE-TUNING (lr=1e-5)")

    # Unfreeze the complete backbone for final fine-tuning.
    base_model.trainable = True

    # # Freeze all BatchNormalization layers to maintain their learned statistics during fine-tuning.
    # for layer in base_model.layers:
    #     if isinstance(layer, tf.keras.layers.BatchNormalization):
    #         layer.trainable = False

    # Early stopping for final training stage.
    early_stopping_3 = ICBHIEarlyStopping(patience=100)

    # Recompile model with lower learning rate.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss=focal_loss(gamma=1.5))
    
    # Perform full fine-tuning.
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=2,callbacks=[icbhi_callback, early_stopping_3])

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================

    # Save the best trained model using a timestamped filename
    # to avoid overwriting previous experiments.
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.keras'))

    # Return trained model.
    return model

# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12345)
args = parser.parse_args()

# Store seed value in a dedicated variable.
SEED = args.random_seed

# Set environment variables before importing TensorFlow.
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set Python random seed.
random.seed(SEED)

# Set NumPy random seed.
np.random.seed(SEED)

# Set TensorFlow random seed.
tf.random.set_seed(SEED)

# Enable deterministic TensorFlow operations.
tf.config.experimental.enable_op_determinism()

print("Loading datasets...")

# Load reproducible training and validation datasets.
train_dataset, val_dataset = load_datasets('/app/data/processed', seed=SEED)

print("Creating model...")

# Create EfficientNet-based model with deterministic initialization.
model, base_model = create_efficientnet_model(input_shape=(128, 129, 1), num_classes=4, seed=SEED)

print("Starting training...")

# Train model using deterministic pipeline configuration.
model = train(model, base_model, train_dataset, val_dataset)