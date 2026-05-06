# This script allows a trained model to be fine-tuned on a specific stetoscope type, which can be useful for improving performance on that type.
# We'll separate the training data into each stetoscope type:

import os, shutil, numpy as np, tensorflow as tf

from sklearn.metrics import confusion_matrix, recall_score
from src.models.custom_cnn import create_custom_cnn
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# Don't show warnings...
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_directory_structure(data_path = '/app/data/processed'):
    """
    Function to create the directory structure for the training and testing datasets for a specific stetoscope type.
    """
    # Create the main directory for the given stetoscope type
    stetoscope_path = os.path.join(data_path, 'Stetoscope')
    os.makedirs(stetoscope_path, exist_ok=True)

    # Create subdirectories for training and testing datasets
    training_stetoscope_path = os.path.join(stetoscope_path, 'Train')
    testing_stetoscope_path = os.path.join(stetoscope_path, 'Test')
    
    # Create directories for training and testing datasets for the given stetoscope type
    os.makedirs(training_stetoscope_path, exist_ok=True)
    os.makedirs(testing_stetoscope_path, exist_ok=True)

    # Create each stetoscope type subdirectory in the training and testing directories
    stetoscope_types = ['AKGC417L', 'Meditron', 'LittC2SE', 'Litt3200']

    # Now, create each subdirectory for each stetoscope type in the training and testing directories
    for stetoscope in stetoscope_types:
        os.makedirs(os.path.join(training_stetoscope_path, stetoscope), exist_ok=True)
        os.makedirs(os.path.join(testing_stetoscope_path, stetoscope), exist_ok=True)

    # Finally, create every category for each stetoscope folder in the training and testing directories:
    categories = ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']

    # Create each category subdirectory for each stetoscope type in the training and testing directories
    for category in categories:
        for stetoscope in stetoscope_types:
            os.makedirs(os.path.join(training_stetoscope_path, stetoscope, category), exist_ok=True)
            os.makedirs(os.path.join(testing_stetoscope_path, stetoscope, category), exist_ok=True)

    return 'Directory structure created successfully!'

def get_and_copy_file(input_path, output_path = '/app/data/processed/Stetoscope', train_test = None, label = None):
    """
    Function used to list all files on a folder and copy them.
    """
    # Get all files from the directory
    list_data = os.listdir(os.path.join(input_path, train_test, label))

    # Copy each file to the corresponding output folder:
    for file in list_data:
        # Get the stetoscope for the file:
        stetoscope_type = file.split('_')[4]

        # Copy the file:
        shutil.copy(os.path.join(input_path, train_test, label, file), os.path.join(output_path, train_test, stetoscope_type, label))

# 0=Mel, 1=MFCC, 2=Delta, 3=Delta-Delta
CHANNEL = 0

def get_label(file_path):
    """
    Function to extract the label from the file path, specifically from the folder.
    """
    # Split the file path into its components and extract the label string from the appropriate position (5th index).
    elements = tf.strings.split(file_path, os.path.sep)
    label_str = elements[7]

    # Create a lookup table to convert the label strings into numerical labels.
    keys = tf.constant(['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'])
    values = tf.constant([0, 1, 2, 3], dtype=tf.int32)

    # Create a static hash table for the label lookup, with a default value of 4 for any unknown labels.
    table = tf.lookup.StaticHashTable( tf.lookup.KeyValueTensorInitializer(keys, values), default_value=4)

    # Use the lookup table to convert the label string into its corresponding numerical label.
    label = table.lookup(label_str)

    # Return the numerical label corresponding to the class of the image.
    return label

def load_npy(path):
    path = path.numpy().decode("utf-8")
    spec = np.load(path)

    target_width = 376
    current_width = spec.shape[1]

    if current_width > target_width:
        start = (current_width - target_width) // 2
        spec = spec[:, start:start + target_width, :]
    elif current_width < target_width:
        pad = target_width - current_width
        spec = np.pad(spec, ((0,0), (0,pad), (0,0)), mode='constant')

    return spec[:, :, CHANNEL:CHANNEL+1].astype(np.float32)

def process_npy(file_path):

    label = get_label(file_path)
    label = tf.one_hot(label, depth=4)
    spec = tf.py_function(load_npy, [file_path], tf.float32)
    spec.set_shape([128, 376, 1])

    return spec, label

def load_datasets_for_specific_stetoscope(training_path, testing_path, stetoscope, batch_size = 16):
    """
    Function to load the dataset asociated with a specific device:
    """
    # Obtain the directory path for each stetoscope path:
    stetoscope_training = os.path.join(training_path, stetoscope)
    stetoscope_testing = os.path.join(testing_path, stetoscope)

    # Load both training and testing datasets using tf.data.Dataset.list_files.
    train_dataset = tf.data.Dataset.list_files(os.path.join(stetoscope_training, '*/*'), shuffle = False)
    test_dataset = tf.data.Dataset.list_files(os.path.join(stetoscope_testing, '*/*'), shuffle = False)

    # Shuffle ANTES del map
    train_dataset = train_dataset.shuffle(buffer_size= len(train_dataset), reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(lambda x: process_npy(x), num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the training dataset with a batch size of 32 and drop any remaining samples that do not fit into a full batch.
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    test_dataset = test_dataset.map(lambda x: process_npy(x), num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Return the prepared training and testing datasets.
    return train_dataset, test_dataset

def obtain_preds_and_labels(model, dataset):
    """
    Function to predict the labels for a given dataset using the provided model and return both the predicted labels and the true labels for further evaluation.
    """
    # Initialize the lists to store predictions and labels for the validation dataset.
    y_pred = []
    y_true = []

    # Iterate through the validation dataset, make predictions using the trained model, and store the predicted and true labels in the respective lists.
    for image, label in dataset:
        # Make predictions using the trained model on the input image
        pred = model(image, training=False)
        label = np.argmax(label, axis=1)

        # Append the predicted labels and true labels to the respective lists for later evaluation.
        y_pred.append(np.argmax(pred, axis = 1))
        y_true.append(label)

    # Concatenate the lists of predictions and true labels into single arrays for easier evaluation.
    y_pred_np = np.concatenate(y_pred)
    y_true_np = np.concatenate(y_true)

    # Return the predicted labels and true labels as numpy arrays for further evaluation.
    return y_pred_np, y_true_np

def calculate_specificity_per_class(y_true, y_pred):
    """
    Function to calculate the specificity for each class in a multi-class classification problem.
    Specificity is calculated as TN / (TN + FP) for each class, where:
    - TN (True Negatives) is the count of samples that are correctly identified as not belonging to the class.
    - FP (False Positives) is the count of samples that are incorrectly identified as belonging to the class.
    
    y_true: the true labels for the samples.
    y_pred: the predicted labels for the samples.
    """
    # Get the unique classes from the true and predicted labels to ensure we calculate specificity for all classes present in the data.
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Initialize a list to store the specificity for each class.
    specificities = []

    # Loop through each class to calculate its specificity.
    for i in classes:
        # Calculate true negatives (TN) and false positives (FP) for the current class.
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        
        # Calculate specificity for the current class, adding a small epsilon to the denominator to avoid division by zero.
        specificity = tn / (tn + fp + 1e-8)
        specificities.append(specificity)

    # Return the list of specificities for all classes.
    return specificities

def obtain_ICBHI_Score(y_true, y_pred):
    """
    Function to calculate the ICBHI score, which is the average of recall and specificity, for a multi-class classification problem.

    y_true: the true labels for the samples.
    y_pred: the predicted labels for the samples.
    """
    # We obtain recall per class and specificity:
    recall_per_class = recall_score(y_true, y_pred, average=None)
    specificity_per_class = calculate_specificity_per_class(y_true, y_pred)

    # Then, we calculate the mean recall and mean specificity across all classes.
    mean_recall = np.mean(recall_per_class)
    mean_specificity = np.mean(specificity_per_class)

    # Finally, we calculate the ICBHI score as the average of mean recall and mean specificity, and return it along with the recall and specificity per class.
    icbhi_score = (mean_recall + mean_specificity) / 2

    # Return the ICBHI score, recall per class, and specificity per class.
    return icbhi_score, recall_per_class, specificity_per_class

def obtain_confusion_matrix(y_true, y_pred):
    """
    Function to get the confusion matrix:
    """
    # Calculate the matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Return the matrix:
    print(f"Confussion Matrix:\n{conf_matrix}")

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

def fine_tuning_model_per_stetoscope(model_path, stetoscope):
    """
    Function to fine-tune a model for a specific stetoscope
    """
    # Load model and weights:
    model = create_custom_cnn()
    model.load_weights(model_path)

    # Load datasets:
    train, test = load_datasets_for_specific_stetoscope(training_path = '/app/data/processed/Stetoscope/Train', testing_path = '/app/data/processed/Stetoscope/Test', stetoscope = stetoscope, batch_size = 16)

    # Get the class weights for the training dataset to handle class imbalance.
    class_weights_dict, class_weights_array = get_class_weights(train)
    
    # Compile the model with the Adam optimizer, categorical cross-entropy loss function, and the defined metrics.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0),
                  loss = focal_loss(gamma=1.5, alpha=class_weights_array))
    
    # Train the model using the fit method.
    history = model.fit(train, epochs = 10, validation_data = test, verbose = 2)

    # Save the model after training is complete.
    model.save(os.path.join('models', datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.h5'))

    # Return the trained model
    return model

##########################################################################################################################################################
# # Exec the creation and copy of files:
# create_directory_structure()

# # Finally, move all files:
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Train', label = 'Healthy')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Train', label = 'Crackle')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Train', label = 'Wheeze')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Train', label = 'Wheeze & Crackle')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Test', label = 'Healthy')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Test', label = 'Crackle')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Test', label = 'Wheeze')
# get_and_copy_file(input_path = '/app/data/processed', output_path = '/app/data/processed/Stetoscope', train_test = 'Test', label = 'Wheeze & Crackle')

print("Creating and loading model weights...")

# After copying the files, we load an specific dataset and evaluate a model:
model = create_custom_cnn()

# Then, load the .h5 file of the best model obtained during training to use it for making predictions and evaluating the ICBHI score with different thresholds.
model.load_weights("/app/models/2026-05-03_15-31-24.h5")

print("Loading datasets...")

# Now, load an specific dataset
train, test = load_datasets_for_specific_stetoscope(training_path = '/app/data/processed/Stetoscope/Train', testing_path = '/app/data/processed/Stetoscope/Test', stetoscope = 'Meditron', batch_size = 16)

print("Evaluating the model on the testing set...")

# Evaluate the model:
y_pred, y_true = obtain_preds_and_labels(model=model, dataset=test)
icbhi_score, recall_per_class, specificity_per_class = obtain_ICBHI_Score(y_true, y_pred)

# Print the score:
print(f"\nICBHI Score: {icbhi_score * 100:.2f} // Recall: {[f'{recall * 100:.2f}' for recall in recall_per_class]} // Specificity: {[f'{specif * 100:.2f}' for specif in specificity_per_class]}\n")

# Get the confusion matrix:
obtain_confusion_matrix(y_true, y_pred)

# ##########################################################################################################################################################
# # Now, exec a bit of fine-tuning and then obtain metrics:
# from datetime import datetime
# from sklearn.utils.class_weight import compute_class_weight

# print("\nStarting fine-tuning...")

# model = fine_tuning_model_per_stetoscope(model_path = '/app/models/2026-05-03_15-31-24.h5', stetoscope = 'AKGC417L')

# print("Now, checking the new metrics...\n")

# # Evaluate the model:
# y_pred, y_true = obtain_preds_and_labels(model=model, dataset=test)
# icbhi_score, recall_per_class, specificity_per_class = obtain_ICBHI_Score(y_true, y_pred)

# # Print the score:
# print(f"\nICBHI Score: {icbhi_score * 100:.2f} // Recall: {[f'{recall * 100:.2f}' for recall in recall_per_class]} // Specificity: {[f'{specif * 100:.2f}' for specif in specificity_per_class]}\n")

# # Get the confusion matrix:
# obtain_confusion_matrix(y_true, y_pred)

# ##########################################################################################################################################################
# # Now, exec a bit of fine-tuning and then obtain metrics:
# print("\nStarting fine-tuning...")

# model = fine_tuning_model_per_stetoscope(model_path = '/app/models/2026-05-03_15-31-24.h5', stetoscope = 'LittC2SE')

# print("Now, checking the new metrics...\n")

# # Evaluate the model:
# y_pred, y_true = obtain_preds_and_labels(model=model, dataset=test)
# icbhi_score, recall_per_class, specificity_per_class = obtain_ICBHI_Score(y_true, y_pred)

# # Print the score:
# print(f"\nICBHI Score: {icbhi_score * 100:.2f} // Recall: {[f'{recall * 100:.2f}' for recall in recall_per_class]} // Specificity: {[f'{specif * 100:.2f}' for specif in specificity_per_class]}\n")

# # Get the confusion matrix:
# obtain_confusion_matrix(y_true, y_pred)

##########################################################################################################################################################
# Now, exec a bit of fine-tuning and then obtain metrics:
print("\nStarting fine-tuning...")

model = fine_tuning_model_per_stetoscope(model_path = '/app/models/2026-05-03_15-31-24.h5', stetoscope = 'Meditron')

print("Now, checking the new metrics...\n")

# Evaluate the model:
y_pred, y_true = obtain_preds_and_labels(model=model, dataset=test)
icbhi_score, recall_per_class, specificity_per_class = obtain_ICBHI_Score(y_true, y_pred)

# Print the score:
print(f"\nICBHI Score: {icbhi_score * 100:.2f} // Recall: {[f'{recall * 100:.2f}' for recall in recall_per_class]} // Specificity: {[f'{specif * 100:.2f}' for specif in specificity_per_class]}\n")

# Get the confusion matrix:
obtain_confusion_matrix(y_true, y_pred)