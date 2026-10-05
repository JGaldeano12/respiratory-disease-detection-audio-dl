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

def train(model, train_dataset, val_dataset):
    """
    Train the model using a three-phase training strategy.

    The training process consists of three consecutive phases with
    progressively adjusted learning rates and early stopping based on
    the validation ICBHI score. The final trained model is saved with
    a timestamped filename.

    Args:
        model (tf.keras.Model): Model to be trained.
        train_dataset (tf.data.Dataset): Training dataset used to update
            the model parameters.
        val_dataset (tf.data.Dataset): Validation dataset used to monitor
            the model performance during training.

    Returns:
        tf.keras.Model: Trained model after completing the three training
            phases.
    """
    # Initialize the ICBHI score callback to monitor the model's performance on the validation dataset.
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset, model_name='CustomCNN')

    # Get the class weights based on the training dataset to handle class imbalance.
    _, class_weights = get_class_weights_from_paths('/app/data/processed')

    # Single early stopping instance reused across all three phases, so that
    # best_score/best_weights track the best model across the entire training
    # run rather than resetting at the start of each phase.
    early_stopping = ICBHIEarlyStopping(patience=5)

    # First phase of training with a higher learning rate and early stopping.
    print("PHASE 1: INITIAL TRAINING (lr=3e-4)")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0), loss=focal_loss(gamma=1.0, alpha=class_weights))
    model.fit(train_dataset, epochs=15, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping])

    # Second phase of training with a reduced learning rate and early stopping.
    print("PHASE 2: REFINE TRAINING (lr=1e-5)")
    early_stopping.patience = 15
    early_stopping.wait = 0
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss=focal_loss(gamma=1.0, alpha=class_weights))
    model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping])

    # Third and final phase of training with a slightly higher learning rate and early stopping.
    print("PHASE 3: FINE-TUNING (lr=5e-5)")
    early_stopping.patience = 20
    early_stopping.wait = 0
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss=focal_loss(gamma=1.0, alpha=class_weights))
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping])

    # Restore the best weights found across the entire training run (all three
    # phases) before saving, so the timestamped checkpoint matches the best
    # model, i.e. the same one saved as best_model_epoch.keras by icbhi_callback.
    if early_stopping.best_weights is not None:
        model.set_weights(early_stopping.best_weights)

    # Save the final trained model with a timestamped filename.
    model.save(os.path.join('models/custom_cnn', 'CustomCNN - ' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.keras'))
    return model

# Parse command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=12345)
args = parser.parse_args()

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
train_dataset, val_dataset = load_datasets('/app/data/processed', seed=SEED)

# Create the custom plain CNN model.
print("Creating model...")
model = create_custom_cnn(input_shape=(128, 251, 1), num_classes=4, seed=SEED)

# Start the training process for the model using the loaded datasets.
print("Starting training...")
model = train(model, train_dataset, val_dataset)