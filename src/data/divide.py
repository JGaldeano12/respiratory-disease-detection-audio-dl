import os
import shutil

import numpy as np

# Class and split directory names used across the dataset.
CLASSES = ["Crackle", "Wheeze", "Wheeze & Crackle", "Healthy"]
SPLITS = ["Train", "Val", "Test"]

def create_directories(input_path):
    """
    Create the class directories required to store spectrogram data.

    The function ensures that directories for the Crackle, Wheeze,
    Wheeze & Crackle, and Healthy classes exist inside the specified
    input path. Existing directories are preserved.

    Args:
        input_path (str): Base directory where the class directories
            will be created.

    Returns:
        None
    """
    # Create the directory of each class if it does not already exist.
    for class_name in CLASSES:
        os.makedirs(os.path.join(input_path, class_name), exist_ok=True)

def check_and_create_directories(input_path):
    """
    Create the directory structure required for the training, validation and testing sets.

    The function ensures that the Train, Val and Test directories exist,
    together with a subdirectory for each respiratory sound class: Crackle,
    Wheeze, Wheeze & Crackle, and Healthy. Existing directories are preserved.

    Args:
        input_path (str): Base directory where the directory structure
            will be created.

    Returns:
        None
    """
    # Create one directory per split and class if it does not already exist.
    for split in SPLITS:
        for class_name in CLASSES:
            os.makedirs(os.path.join(input_path, split, class_name), exist_ok=True)

def split_patients_by_train_val_test(aux_file, seed, train_test_split, val_split):
    """
    Split patients and their respiratory cycles into training, validation and testing sets.

    The respiratory cycle data is loaded from the provided NumPy file, and
    unique patient identifiers are randomly shuffled using the specified seed.
    The first portion of patients, determined by `train_test_split`, forms the
    development block, while the remaining patients form the testing set. A
    fraction of the development block, determined by `val_split`, is then held
    out as the validation set, and the rest is used for training. All sets are
    patient-wise disjoint, and the testing set does not depend on `val_split`.

    Args:
        aux_file (str): Path to the NumPy file containing labeled respiratory
            cycle data, with patient identifiers stored in the first column.
        seed (int): Random seed used to reproducibly shuffle patient IDs.
        train_test_split (float): Percentage of patients assigned to the
            development block (training + validation).
        val_split (float): Percentage of the development block assigned to
            the validation set.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Respiratory cycle data
        corresponding to the training, validation and testing patients,
        respectively.
    """
    # Load the file with the labeled respiratory cycles.
    aux_file = np.load(aux_file)

    # Obtain the unique patient IDs from the respiratory cycles data.
    list_id_patients = np.unique(aux_file[:, 0])

    # Use a local random generator so we don't reset the global NumPy RNG state.
    rng = np.random.default_rng(seed)
    rng.shuffle(list_id_patients)

    # The first n_dev shuffled patients form the development block and the rest the testing set.
    n_dev = int(len(list_id_patients) * train_test_split / 100)
    n_val = int(n_dev * val_split / 100)

    # The last n_val patients of the development block form the validation set.
    random_train = list_id_patients[:n_dev - n_val]
    random_val = list_id_patients[n_dev - n_val:n_dev]
    random_test = list_id_patients[n_dev:]

    # Get the respiratory cycles corresponding to the patients of each set.
    cycles_train = aux_file[np.isin(aux_file[:, 0], random_train)]
    cycles_val = aux_file[np.isin(aux_file[:, 0], random_val)]
    cycles_test = aux_file[np.isin(aux_file[:, 0], random_test)]

    return cycles_train, cycles_val, cycles_test

def move_spectrograms_by_split(input_path, cycles_train, cycles_val, cycles_test):
    """
    Move spectrogram images into their corresponding training, validation or testing sets.

    The function retrieves spectrogram files from each respiratory sound class
    directory, determines the patient identifier associated with each file, and
    moves the spectrogram to the corresponding Train, Val or Test directory
    according to the set the patient belongs to.

    Args:
        input_path (str): Base directory containing the class-specific
            spectrogram directories.
        cycles_train (np.ndarray): Training respiratory cycle data used to
            identify patients assigned to the training set.
        cycles_val (np.ndarray): Validation respiratory cycle data used to
            identify patients assigned to the validation set.
        cycles_test (np.ndarray): Testing respiratory cycle data used to
            identify patients assigned to the testing set.

    Returns:
        None
    """
    # List every spectrogram file inside the class directories.
    full_data = [os.path.join(input_path, class_name, f) for class_name in CLASSES for f in os.listdir(os.path.join(input_path, class_name))]

    # Check and create the necessary directories for every split if they do not exist.
    check_and_create_directories(input_path)

    # Map each patient ID (first column) to its split for exact and fast lookups.
    patient_split = {}
    for split, cycles in zip(SPLITS, [cycles_train, cycles_val, cycles_test]):
        for id_patient in np.unique(cycles[:, 0]):
            patient_split[id_patient] = split

    print("Starting to move spectrograms into Train, Val and Test directories based on patient IDs...")

    for spectrogram in full_data:
        # Get the label from the parent directory and the patient ID from the filename.
        label = os.path.basename(os.path.dirname(spectrogram))
        id_patient = os.path.basename(spectrogram).split("_")[0]

        # Check which set the patient belongs to.
        split = patient_split.get(id_patient)
        if split is None:
            print(f"Patient ID {id_patient} not found in any set.")
            continue

        # Move the spectrogram to its split. The full destination path is given so that an existing file is overwritten.
        shutil.move(spectrogram, os.path.join(input_path, split, label, os.path.basename(spectrogram)))

    print("Spectrograms moved into Train, Val and Test directories.")