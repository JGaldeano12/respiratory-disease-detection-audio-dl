import argparse
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.data.utils import get_class_weights_from_paths, load_datasets
from src.models.custom_cnn import create_custom_cnn
from src.training.custom_callback import ICBHI_Score_PrintingCallback, ICBHIEarlyStopping
from src.training.losses import focal_loss

# Training hyperparameters of the single-phase schedule.
INITIAL_LEARNING_RATE = 3e-4
FINAL_LEARNING_RATE_FRACTION = 0.01
MAX_EPOCHS = 100
PATIENCE = 20

def add_frequency_normalization(model, input_shape):
    """
    Prepend a per-band mean normalization to the model.

    For each spectrogram, the mean over time of every mel band is subtracted
    (cepstral mean normalization). This removes the constant spectral
    signature of the recording device or channel. It is built with standard
    Keras layers, so it is saved inside the model and applied automatically
    at evaluation time.

    Args:
        model (tf.keras.Model): Model that receives the normalized spectrograms.
        input_shape (tuple): Input shape (128, width, 1).

    Returns:
        tf.keras.Model: Model with the normalization as its first step.
    """
    inputs = tf.keras.Input(shape=input_shape)
    band_means = tf.keras.layers.AveragePooling2D(pool_size=(1, input_shape[1]), name='band_time_mean')(inputs)
    normalized = tf.keras.layers.Subtract(name='frequency_mean_normalization')([inputs, band_means])
    return tf.keras.Model(inputs, model(normalized), name=f'{model.name}_freq_norm')

def train(model, train_dataset, val_dataset, smoothing_window=1, alpha_mode='balanced'):
    """
    Train the model in a single phase with a cosine-decaying learning rate.

    The learning rate starts at INITIAL_LEARNING_RATE and decays following a
    cosine schedule down to FINAL_LEARNING_RATE_FRACTION of its initial value
    over MAX_EPOCHS epochs. The model is compiled only once, so the optimizer
    state is never reset. Early stopping is based on the validation ICBHI
    score, and the best weights are restored before saving the final model
    with a timestamped filename.

    Args:
        model (tf.keras.Model): Model to be trained.
        train_dataset (tf.data.Dataset): Training dataset used to update
            the model parameters.
        val_dataset (tf.data.Dataset): Validation dataset used to monitor
            the model performance during training.
        smoothing_window (int, optional): Number of epochs averaged to select
            the checkpoint and drive early stopping. Defaults to 1.
        alpha_mode (str, optional): Class weighting of the focal loss:
            "balanced" (balanced class weights) or "sqrt" (their square root,
            a milder compensation). Defaults to "balanced".

    Returns:
        tf.keras.Model: Trained model with the best weights found.
    """
    # Initialize the ICBHI score callback to monitor the model's performance on the validation dataset.
    # A distinct model name keeps its checkpoint separate from the three-phase Custom CNN.
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset, model_name='CustomCNNCosine', smoothing_window=smoothing_window)

    # Get the class weights based on the training dataset to handle class imbalance.
    _, class_weights = get_class_weights_from_paths('/app/data/processed')
    if alpha_mode == 'sqrt':
        class_weights = np.sqrt(class_weights)
    print(f"Focal loss class weights ({alpha_mode}): {np.round(class_weights, 3)}")

    # Early stopping on the validation ICBHI score over the whole run.
    early_stopping = ICBHIEarlyStopping(patience=PATIENCE)

    # Cosine decay over the maximum number of training steps.
    steps_per_epoch = int(train_dataset.cardinality().numpy())
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=MAX_EPOCHS * steps_per_epoch,
        alpha=FINAL_LEARNING_RATE_FRACTION
    )

    # Single training phase: the model is compiled once and trained until early stopping or MAX_EPOCHS.
    print(f"SINGLE PHASE: COSINE DECAY (lr={INITIAL_LEARNING_RATE:g} -> {INITIAL_LEARNING_RATE * FINAL_LEARNING_RATE_FRACTION:g}, "
          f"max {MAX_EPOCHS} epochs, patience {PATIENCE})")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss=focal_loss(gamma=1.0, alpha=class_weights))
    model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping])

    # Restore the best weights found before saving, so the timestamped checkpoint matches the best
    # model, i.e. the same one saved as best_model_<model_name>.keras by icbhi_callback.
    if early_stopping.best_weights is not None:
        model.set_weights(early_stopping.best_weights)

    # Save the final trained model with a timestamped filename.
    model.save(os.path.join('models/custom_cnn', 'CustomCNNCosine - ' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.keras'))
    return model

# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12345)
parser.add_argument('--select_on', type=str, default='val', choices=['val', 'test'], help='Set used to select the checkpoint and drive early stopping (test = oracle protocol)')
parser.add_argument('--smoothing_window', type=int, default=1, help='Epochs averaged to select the checkpoint (1 = no smoothing)')
parser.add_argument('--freq_norm', action='store_true', help='Subtract the time mean of each mel band (per spectrogram)')
parser.add_argument('--band_mask', action='store_true', help='Randomly blank the 1.5-2 kHz band during training')
parser.add_argument('--alpha', type=str, default='balanced', choices=['balanced', 'sqrt'], help='Class weighting of the focal loss')
args = parser.parse_args()
print(f"Options: select_on={args.select_on}, smoothing_window={args.smoothing_window}, freq_norm={args.freq_norm}, band_mask={args.band_mask}, alpha={args.alpha}")

# Set the random seed for reproducibility across various libraries and TensorFlow operations.
SEED = args.random_seed

# Set environment variables and seeds for reproducibility.
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set random seeds for Python's random module, NumPy, and TensorFlow.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Load the training and validation datasets.
print("Loading datasets...")
train_dataset, val_dataset, test_dataset = load_datasets('/app/data/processed', seed=SEED, band_mask=args.band_mask)
# Checkpoint selection and early stopping use the validation set, or the testing set in the oracle protocol.
if args.select_on == 'test':
    print("WARNING: oracle protocol - the checkpoint is selected on the TEST set.")
    val_dataset = test_dataset
if val_dataset is None:
    raise ValueError("No validation set found; use --select_on test (oracle protocol) or generate a validation set.")
# The input shape (128, width, 1) is taken from the dataset, since the width depends on the segment duration.
input_shape = tuple(int(d) for d in train_dataset.element_spec[0].shape[1:])

# Create the custom plain CNN model.
print(f"Creating model with input shape {input_shape}...")
model = create_custom_cnn(input_shape=input_shape, num_classes=4, seed=SEED)
if args.freq_norm:
    model = add_frequency_normalization(model, input_shape)

# Start the training process for the model using the loaded datasets.
print("Starting training...")
model = train(model, train_dataset, val_dataset, smoothing_window=args.smoothing_window, alpha_mode=args.alpha)
