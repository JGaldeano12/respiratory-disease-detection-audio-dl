import tensorflow as tf, numpy as np, os, random
import glob

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from src.training.custom_callback import ICBHI_Score_PrintingCallback
from src.models.custom_cnn import create_custom_cnn

class ICBHIEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=10):
        super().__init__()
        self.patience = patience
        self.best_score = 0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_score = logs.get('icbhi_score', -np.inf)
        
        # Guarda pesos siempre en época 0, independientemente del score
        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True
                print(f"\nEarly stopping: mejor ICBHI Score = {self.best_score:.5f}")

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

def process_image(file_path):
    """
    Function to process an image file, including reading the image, decoding it, resizing it, and extracting the label.
    """
    # Extract the label from the file path using the get_label function.
    label = get_label(file_path)

    # Convert the label to a one-hot encoded vector with a depth of 4 (for 4 classes).
    label = tf.one_hot(label, depth=4)

    # Read the image file and decode it into a tensor with 3 color channels (RGB).
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)

    # Set the shape of the image tensor to ensure it has 3 channels.
    img.set_shape([None, None, 3])

    # Resize the image to a fixed size of 64x193 pixels.
    img = tf.image.resize(img, [64, 193])

    # Return the processed image and its corresponding one-hot encoded label.
    return img, label

def spec_augment_tf(spec, time_mask_param=20, freq_mask_param=10,
                    num_time_masks=2, num_freq_masks=2):

    H = tf.shape(spec)[0]
    W = tf.shape(spec)[1]

    # 🔹 FREQUENCY MASK
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

    # 🔹 TIME MASK
    for _ in range(num_time_masks):
        t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, W - t + 1, dtype=tf.int32)

        mask = tf.ones_like(spec)

        # construir máscara segura
        time_mask = tf.concat([
            tf.ones((H, t0, tf.shape(spec)[2])),
            tf.zeros((H, t, tf.shape(spec)[2])),
            tf.ones((H, W - t0 - t, tf.shape(spec)[2]))
        ], axis=1)

        spec = spec * time_mask

    return spec

def get_class_weights_from_paths(dir_dataset):
    """
    Calcula class weights directamente desde los paths del train set,
    sin pasar por el pipeline de TF (sin shuffle, batch ni augmentation).
    """
    # Obtener todos los paths del train set
    train_paths = glob.glob(os.path.join(dir_dataset, 'Train/*/*'))

    # Extraer labels desde el nombre de la carpeta (igual que get_label)
    label_map = {'Healthy': 0, 'Crackle': 1, 'Wheeze': 2, 'Wheeze & Crackle': 3}
    
    labels = []
    for path in train_paths:
        folder = path.split(os.path.sep)[-2]  # carpeta padre = clase
        labels.append(label_map[folder])
    
    labels = np.array(labels)
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    
    return dict(enumerate(class_weights)), np.array(class_weights)

CHANNEL = 0  # 0=Mel, 1=MFCC, 2=Delta, 3=Delta-Delta

# def load_npy(path):
#     path = path.numpy().decode("utf-8")
#     spec = np.load(path)

#     target_width = 501
#     current_width = spec.shape[1]

#     if current_width > target_width:
#         start = (current_width - target_width) // 2
#         spec = spec[:, start:start + target_width, :]
#     elif current_width < target_width:
#         pad = target_width - current_width
#         spec = np.pad(spec, ((0,0), (0,pad), (0,0)), mode='constant')

#     return spec[:, :, CHANNEL:CHANNEL+1].astype(np.float32)

def load_npy(path):
    path = path.numpy().decode("utf-8")
    spec = np.load(path)
    assert spec.shape[1] == 501, f"Unexpected spectrogram width: {spec.shape[1]}"
    return spec[:, :, CHANNEL:CHANNEL+1].astype(np.float32)

def process_npy(file_path, training=True):

    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)
    spec = tf.py_function(load_npy, [file_path], tf.float32)

    # 🔥 AJUSTAR AQUÍ
    spec.set_shape([128, 501, 1])  # ← cambia según tus features

    if training:
        spec = tf.cond(
            tf.random.uniform([]) < 0.6,
            lambda: spec_augment_tf(spec, time_mask_param=12, freq_mask_param=6, num_time_masks=1, num_freq_masks=1),
            lambda: spec
        )

    return spec, label

def load_datasets(dir_dataset):
    """
    Function to load the training / testing datasets from the specified directories, process the images and labels, and prepare them for training.
    """
    # Load both training and testing datasets using tf.data.Dataset.list_files.
    train_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Train/*/*'), shuffle = False)
    test_dataset = tf.data.Dataset.list_files(os.path.join(dir_dataset, 'Test/*/*'), shuffle = False)

    # Shuffle ANTES del map
    train_dataset = train_dataset.shuffle(buffer_size= len(train_dataset), reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: process_npy(x, training=True), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the training dataset with a batch size of 32 and drop any remaining samples that do not fit into a full batch.
    train_dataset = train_dataset.batch(16, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(lambda x: process_npy(x, training=False), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(16, drop_remainder=True)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Return the prepared training and testing datasets.
    return train_dataset, test_dataset

def get_class_weights(dataset):
    """
    Function to obtain the class weights for the 'CategoricalFocalLoss' loss function.
    """
    # Get labels from the dataset and concatenate them into a single array.
    y_train = np.concatenate([y.numpy() for x, y in dataset], axis=0)

    # Transform the one-hot encoded labels into integer labels by taking the argmax along the appropriate axis.
    y_train_int = np.argmax(y_train, axis=1)

    # Get the unique class labels from the integer labels to compute class weights.
    classes = np.unique(y_train_int)

    # Calculate the class weights (must return an array).
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_int)

    # Cast to a numpy array to ensure it is in the correct format for use in the loss function.
    class_weights_array = np.array(class_weights)

    # Return the dictionary and array of class weights.
    return dict(enumerate(class_weights)), class_weights_array

def train(model, train_dataset, val_dataset, epochs=35):
    """
    Function to train a given model using the provided training and validation datasets.

    model: the neural network model to be trained (e.g., a custom CNN, ResNet50, or VGG19).
    train_dataset: the dataset used for training the model.
    val_dataset: the dataset used for validating the model during training.
    class_weights: a dictionary mapping class indices to weights, used to handle class imbalance during training (optional).
    epochs: the number of epochs to train the model (default is 10).
    batch_size: the number of samples per batch during training (default is 32).
    callbacks: a list of Keras callbacks to be applied during training (optional).
    """
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5, decay_steps=1000, decay_rate=0.95, staircase=True)

    # # Early stopping callback to prevent overfitting by monitoring the validation loss and stopping training if it does not improve for a certain number of epochs.
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    # Custom callback to print the ICBHI Score at the end of each epoch, which is a specific metric relevant to the task at hand.
    icbhi_callback = ICBHI_Score_PrintingCallback(val_dataset)
    early_stopping = ICBHIEarlyStopping(patience=5)

    # # Get the class weights for the training dataset to handle class imbalance.
    # class_weights_dict, class_weights_array = get_class_weights(train_dataset)
    
    # Calculate class weights directly from the file paths of the training set, without going through the TF pipeline (without shuffle, batch, or augmentation).
    _, class_weights_array = get_class_weights_from_paths('/app/data/processed')

    # Compile the model with the Adam optimizer, categorical cross-entropy loss function, and the defined metrics.
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0), loss = focal_loss(gamma=1.0, alpha=class_weights_array))
    
    # Train the model using the fit method.
    # history = model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, verbose = 2, callbacks=[ICBHI_Score_PrintingCallback(val_dataset), icbhi_callback, early_stopping])
    history = model.fit(train_dataset, epochs = epochs, validation_data = val_dataset, verbose = 2, callbacks=[icbhi_callback, early_stopping])

    # Save the model after training is complete.
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.h5'))

    # Return the training history, which contains information about the loss and metrics for each epoch.
    return history

# Set warning level to ignore to suppress TensorFlow warnings during execution.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading datasets...")

# Load the training and validation datasets.
train_dataset, val_dataset = load_datasets('/app/data/processed')

print("Creating model...")

# Create the model using the create_custom_cnn function, which defines a custom CNN architecture.
model = create_custom_cnn(input_shape=(128, 501, 1), num_classes=4)

print("Training model...")

# Train the model using the defined train function, which includes class weights to handle class imbalance.
history = train(model = model, train_dataset = train_dataset, val_dataset = val_dataset, epochs = 100)