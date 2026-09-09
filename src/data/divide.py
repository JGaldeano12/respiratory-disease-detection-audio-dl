import shutil, os, numpy as np

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
    # Create directories for each class if they do not already exist
    for dir_path in [os.path.join(input_path, "Crackle"),
                     os.path.join(input_path, "Wheeze"), 
                     os.path.join(input_path, "Wheeze & Crackle"), 
                     os.path.join(input_path, "Healthy")]:
        
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

def check_and_create_directories(input_path):
    """
    Create the directory structure required for the training and testing sets.

    The function ensures that the Train and Test directories exist, together
    with a subdirectory for each respiratory sound class: Crackle, Wheeze,
    Wheeze & Crackle, and Healthy. Existing directories are preserved.

    Args:
        input_path (str): Base directory where the training and testing
            directory structure will be created.

    Returns:
        None
    """
    # Create directories for Train and Test sets, along with subdirectories for each class
    for dir_path in [os.path.join(input_path, "Train"), 
                     os.path.join(input_path, "Test"), 
                     os.path.join(input_path, "Train/Crackle"),
                     os.path.join(input_path, "Train/Wheeze"), 
                     os.path.join(input_path, "Train/Wheeze & Crackle"), 
                     os.path.join(input_path, "Train/Healthy"), 
                     os.path.join(input_path, "Test/Crackle"), 
                     os.path.join(input_path, "Test/Wheeze"), 
                     os.path.join(input_path, "Test/Wheeze & Crackle"),
                     os.path.join(input_path, "Test/Healthy")]:
        
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

def split_patients_by_train_test(aux_file, seed, train_test_split):
    """
    Split patients and their respiratory cycles into training and testing sets.

    The respiratory cycle data is loaded from the provided NumPy file, and
    unique patient identifiers are randomly shuffled using the specified seed.
    The first portion of patients, determined by `train_test_split`, is
    assigned to the training set, while the remaining patients are assigned
    to the testing set. The corresponding respiratory cycles are then returned
    for each set.

    Args:
        aux_file (str): Path to the NumPy file containing labeled respiratory
            cycle data, with patient identifiers stored in the first column.
        seed (int): Random seed used to reproducibly shuffle patient IDs.
        train_test_split (float): Percentage of the 112 patients assigned to
            the training set.

    Returns:
        tuple[np.ndarray, np.ndarray]: Respiratory cycle data corresponding to
        the training patients and testing patients, respectively.
    """
    # First, load file with the labeled respiratory cycles
    aux_file = np.load(aux_file)

    # Then, obtain the unique patient IDs from the respiratory cycles data
    list_id_patients = np.unique(aux_file[:, 0])

    # Use a local random generator for the train/test split so we don't reset the global NumPy RNG state.
    rng = np.random.default_rng(seed)
    rng.shuffle(list_id_patients)

    # Split the patient IDs into training and testing sets based on the specified train_test_split ratio.
    # The first portion of the shuffled patient IDs will be assigned to the training set, while the remaining portion will be assigned to the testing set.
    random_train = list_id_patients[:int(112 * train_test_split / 100)]
    random_test = list_id_patients[int(112 * train_test_split / 100):]

    # Finally, get the respiratory cycles corresponding to the training and testing patient IDs by filtering the original respiratory cycles data based on the assigned patient IDs for each set.
    cycles_train = aux_file[np.isin(aux_file[:, 0], random_train)]
    cycles_test = aux_file[np.isin(aux_file[:, 0], random_test)]

    # Finally, return the lists.
    return cycles_train, cycles_test

def move_spectrograms_by_train_test(input_path, cycles_train, cycles_test):
    """
    Copy spectrogram images into their corresponding training or testing sets.

    The function retrieves spectrogram files from each respiratory sound class
    directory, determines the patient identifier associated with each file, and
    copies the spectrogram to the corresponding Train or Test directory
    according to whether the patient belongs to `cycles_train` or
    `cycles_test`.

    Args:
        input_path (str): Base directory containing the class-specific
            spectrogram directories.
        cycles_train (np.ndarray): Training respiratory cycle data used to
            identify patients assigned to the training set.
        cycles_test (np.ndarray): Testing respiratory cycle data used to
            identify patients assigned to the testing set.

    Returns:
        None
    """
    # List every spectrogram image in the directory and its subdirectories, and categorize them based on their corresponding patient IDs
    crackle_data = [os.path.join(os.path.join(input_path, "Crackle"), f) for f in os.listdir(os.path.join(input_path, "Crackle"))]
    wheeze_data = [os.path.join(os.path.join(input_path, "Wheeze"), f) for f in os.listdir(os.path.join(input_path, "Wheeze"))]
    crackles_and_wheeze_data = [os.path.join(os.path.join(input_path, "Wheeze & Crackle"), f) for f in os.listdir(os.path.join(input_path, "Wheeze & Crackle"))]
    healthy_data = [os.path.join(os.path.join(input_path, "Healthy"), f) for f in os.listdir(os.path.join(input_path, "Healthy"))]

    # Check and create the necessary directories for training and testing sets if they do not exist
    check_and_create_directories(input_path)
    
    # Create a list with every spectrogrma.
    full_data = [crackle_data, wheeze_data, crackles_and_wheeze_data, healthy_data]

    print("Starting to move spectrograms into Train and Test directories based on patient IDs...")

    # Iterate through the list of spectrogram images for each pathology, and move each image to the corresponding training or testing directory 
    # based on the patient ID extracted from the filename and its presence in the random_train or random_test lists.
    for data in full_data:
        # Iterate through each spectrogram image in the current pathology's list of images.
        for spectrogram in data:
            # Separate the path of the spectrogram image to extract the patient ID and label information from the filename and directory structure.
            path_elements = spectrogram.split("/")

            # Extract the patient ID and label information from the filename and directory structure to determine the appropriate destination 
            # directory for the spectrogram image based on its corresponding patient ID and assigned training or testing set.
            filename = path_elements[5]

            # Get the label from the directory structure.
            label = path_elements[4]
            
            # Extract the patient ID from the filename.
            filename_data = filename.split("_")
            id_patient = filename_data[0]

            # Consulto a qué conjunto pertenece...
            if id_patient in cycles_train:
                shutil.copy(spectrogram, os.path.join(input_path, 'Train', label))
                print(f"Moved spectrogram {spectrogram} to Train/{label}")
            elif id_patient in cycles_test:
                shutil.copy(spectrogram, os.path.join(input_path, 'Test', label))
                print(f"Moved spectrogram {spectrogram} to Test/{label}")
            else:
                print(f"Patient ID {id_patient} not found in either training or testing sets.")