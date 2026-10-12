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

def load_official_split(official_file):
    """
    Load the official ICBHI 2017 challenge train/test partition.

    Each line of the file contains a recording identifier and its assigned
    set ("train" or "test"), separated by whitespace. Windows line endings
    are handled transparently.

    Args:
        official_file (str): Path to the official partition file
            (ICBHI_challenge_train_test.txt).

    Returns:
        dict: Mapping from recording identifier to "train" or "test".
    """
    with open(official_file) as file:
        return {line.split()[0]: line.split()[1] for line in file if line.strip()}

def split_patients_official(aux_file, official_file, seed, val_split):
    """
    Split respiratory cycles using the official ICBHI 2017 train/test partition.

    The official partition is defined per recording. Every respiratory cycle is
    assigned to the set of its recording. A recording missing from the file is
    matched by its patient, recording index, chest location and acquisition
    mode, ignoring the device name (the official file lists one recording of
    patient 226 with a different device than the dataset). The validation set
    is then drawn, patient-wise and using the given seed, from the patients
    whose recordings all belong to the official training set, so that no
    validation patient appears in the testing set. The official testing set
    is kept unchanged.

    Args:
        aux_file (str): Path to the NumPy file containing labeled respiratory
            cycle data, with patient identifiers in the first column and
            recording identifiers in the second column.
        official_file (str): Path to the official partition file.
        seed (int): Random seed used to select the validation patients.
        val_split (float): Percentage of the training-only patients assigned
            to the validation set.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Respiratory cycle data
        corresponding to the training, validation and testing sets,
        respectively.
    """
    # Load the labeled respiratory cycles and the official partition.
    aux_file = np.load(aux_file)
    official = load_official_split(official_file)

    # Fallback lookup ignoring the device name (last field of the recording identifier).
    official_no_device = {"_".join(recording.split("_")[:4]): split for recording, split in official.items()}

    # Assign the official set to every respiratory cycle based on its recording.
    cycle_sets = []
    unmatched = set()
    for recording in aux_file[:, 1]:
        split = official.get(recording)
        if split is None:
            split = official_no_device["_".join(recording.split("_")[:4])]
            unmatched.add((recording, split))
        cycle_sets.append(split)
    cycle_sets = np.array(cycle_sets)

    for recording, split in sorted(unmatched):
        print(f"Recording {recording} not found in the official file; matched ignoring the device ({split}).")

    # Report the patients that the official partition places in both sets.
    train_patients = set(aux_file[cycle_sets == "train", 0])
    test_patients = set(aux_file[cycle_sets == "test", 0])
    shared = sorted(train_patients & test_patients)
    if shared:
        print(f"Warning: the official partition places patients {shared} in both the training and testing sets.")

    # Draw the validation patients only among the patients that never appear in the testing set.
    candidates = np.array(sorted(train_patients - test_patients))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    n_val = int(len(train_patients) * val_split / 100)
    val_patients = candidates[:n_val]

    is_val = np.isin(aux_file[:, 0], val_patients)

    cycles_train = aux_file[(cycle_sets == "train") & ~is_val]
    cycles_val = aux_file[is_val]
    cycles_test = aux_file[cycle_sets == "test"]

    return cycles_train, cycles_val, cycles_test

def split_patients_respirenet(aux_file, folds_file, test_fold, seed, val_split):
    """
    Split respiratory cycles using the patient-wise 5-fold partition published with RespireNet.

    Each line of the folds file contains a patient identifier and its fold
    (0-4). The patients of `test_fold` form the testing set (RespireNet uses
    fold 4 for its 80-20 results) and the rest form the development block. A
    fraction of the development patients, drawn with the given seed, is held
    out as the validation set.

    Args:
        aux_file (str): Path to the NumPy file containing labeled respiratory
            cycle data, with patient identifiers in the first column.
        folds_file (str): Path to RespireNet's patient_list_foldwise.txt.
        test_fold (int): Fold used as the testing set.
        seed (int): Random seed used to select the validation patients.
        val_split (float): Percentage of the development patients assigned to
            the validation set (0 for no validation set).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Respiratory cycle data
        corresponding to the training, validation and testing sets.
    """
    aux_file = np.load(aux_file)

    with open(folds_file) as file:
        folds = {line.split()[0]: int(line.split()[1]) for line in file if line.strip()}

    missing = sorted(set(aux_file[:, 0]) - set(folds))
    if missing:
        raise ValueError(f"Patients missing from the folds file: {missing}")

    # Development patients (all folds except the test fold), sorted for reproducibility.
    patients = np.unique(aux_file[:, 0])
    test_patients = np.array([p for p in patients if folds[p] == test_fold])
    dev_patients = np.array([p for p in patients if folds[p] != test_fold])

    # The validation patients are drawn from the development block with the given seed.
    rng = np.random.default_rng(seed)
    rng.shuffle(dev_patients)
    n_val = int(len(dev_patients) * val_split / 100)
    val_patients, train_patients = dev_patients[:n_val], dev_patients[n_val:]

    cycles_train = aux_file[np.isin(aux_file[:, 0], train_patients)]
    cycles_val = aux_file[np.isin(aux_file[:, 0], val_patients)]
    cycles_test = aux_file[np.isin(aux_file[:, 0], test_patients)]

    return cycles_train, cycles_val, cycles_test

def get_split(aux_file, seed, train_test_split, val_split, split_mode="random", official_file=None, folds_file=None, test_fold=4):
    """
    Obtain the training, validation and testing respiratory cycles.

    Args:
        aux_file (str): Path to the NumPy file containing labeled respiratory
            cycle data.
        seed (int): Random seed used for the patient-wise selection.
        train_test_split (float): Percentage of patients assigned to the
            training + validation block (only used in "random" mode).
        val_split (float): Percentage of training patients assigned to the
            validation set.
        split_mode (str, optional): "random" for a seeded patient-wise split,
            "official" for the official ICBHI 2017 partition, or "respirenet"
            for RespireNet's patient-wise folds. Defaults to "random".
        official_file (str, optional): Path to the official partition file
            (only used in "official" mode).
        folds_file (str, optional): Path to RespireNet's folds file (only used
            in "respirenet" mode).
        test_fold (int, optional): Fold used as the testing set in
            "respirenet" mode. Defaults to 4.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Respiratory cycle data
        corresponding to the training, validation and testing sets.
    """
    if split_mode == "official":
        return split_patients_official(aux_file, official_file, seed, val_split)
    if split_mode == "random":
        return split_patients_by_train_val_test(aux_file, seed, train_test_split, val_split)
    if split_mode == "respirenet":
        return split_patients_respirenet(aux_file, folds_file, test_fold, seed, val_split)
    raise ValueError(f"Unknown split_mode: {split_mode}")

def move_spectrograms_by_split(input_path, cycles_train, cycles_val, cycles_test):
    """
    Move spectrogram images into their corresponding training, validation or testing sets.

    The function retrieves spectrogram files from each respiratory sound class
    directory, determines the recording identifier associated with each file,
    and moves the spectrogram to the corresponding Train, Val or Test directory
    according to the set the recording belongs to.

    Args:
        input_path (str): Base directory containing the class-specific
            spectrogram directories.
        cycles_train (np.ndarray): Training respiratory cycle data used to
            identify recordings assigned to the training set.
        cycles_val (np.ndarray): Validation respiratory cycle data used to
            identify recordings assigned to the validation set.
        cycles_test (np.ndarray): Testing respiratory cycle data used to
            identify recordings assigned to the testing set.

    Returns:
        None
    """
    # List every spectrogram file inside the class directories.
    full_data = [os.path.join(input_path, class_name, f) for class_name in CLASSES for f in os.listdir(os.path.join(input_path, class_name))]

    # Check and create the necessary directories for every split if they do not exist.
    check_and_create_directories(input_path)

    # Map each recording ID (second column) to its split. Recordings are used instead of
    # patients because the official ICBHI partition is defined per recording.
    recording_split = {}
    for split, cycles in zip(SPLITS, [cycles_train, cycles_val, cycles_test]):
        for recording in np.unique(cycles[:, 1]):
            recording_split[recording] = split

    print("Starting to move spectrograms into Train, Val and Test directories based on recording IDs...")

    for spectrogram in full_data:
        # Get the label from the parent directory and the recording ID (first five fields) from the filename.
        label = os.path.basename(os.path.dirname(spectrogram))
        recording = "_".join(os.path.basename(spectrogram).split("_")[:5])

        # Check which set the recording belongs to.
        split = recording_split.get(recording)
        if split is None:
            print(f"Recording {recording} not found in any set.")
            continue

        # Move the spectrogram to its split. The full destination path is given so that an existing file is overwritten.
        shutil.move(spectrogram, os.path.join(input_path, split, label, os.path.basename(spectrogram)))

    print("Spectrograms moved into Train, Val and Test directories.")