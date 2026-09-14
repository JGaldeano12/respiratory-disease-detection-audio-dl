import tensorflow as tf, numpy as np, os, random, glob, argparse

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.training.custom_callback import ICBHI_Score_PrintingCallback, ICBHIEarlyStopping
from src.training.losses import focal_loss
from src.data.utils import load_datasets
from src.models.vgg_19 import create_vgg_19_model

def train(model, base_model, train_dataset, val_dataset):
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
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset)

    # First phase of training with a higher learning rate and early stopping.
    print("PHASE 1: INITIAL TRAINING (lr=1e-3)")
    early_stopping_1 = ICBHIEarlyStopping(patience=5)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0), loss=focal_loss(gamma=1.0))
    model.fit(train_dataset, epochs=15, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping_1])

    # Second phase of training with a reduced learning rate and early stopping.
    print("PHASE 2: REFINE TRAINING (lr=1e-4)")

    # Unfreeze the VGG-19 backbone.
    base_model.trainable = True

    # Freeze the first 30 layers of the VGG-19 backbone to retain learned features.
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Early stopping for refinement phase.
    early_stopping_2 = ICBHIEarlyStopping(patience=15)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss=focal_loss(gamma=1.0))
    model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping_2])

    # Third and final phase of training with an even lower learning rate and early stopping.
    print("PHASE 3: FINE-TUNING (lr=1e-5)")

    # Unfreeze the EfficientNet backbone.
    base_model.trainable = True

    # Early stopping for final training stage.
    early_stopping_3 = ICBHIEarlyStopping(patience=20)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss=focal_loss(gamma=1.0))
    model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=2, callbacks=[icbhi_callback, early_stopping_3])

    # Save the final trained model with a timestamped filename.
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.keras'))
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

# Create VGG-19-based model with deterministic initialization.
print("Creating model...")
model, base_model = create_vgg_19_model(input_shape=(128, 251, 1), num_classes=4, seed=SEED)

# Start the training process for the model using the loaded datasets.
print("Starting training...")
model = train(model, base_model, train_dataset, val_dataset)