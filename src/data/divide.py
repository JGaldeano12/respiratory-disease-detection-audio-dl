import shutil, os, numpy as np

def create_directories(input_path):
    """
    Function to check if the necessary directories for training and testing sets exist, and create them if they do not exist. 
    This includes directories for each class (Crackle, Wheeze, Wheeze & Crackle, Healthy) within both the Train and Test directories.
    """
    for dir_path in [os.path.join(input_path, "Crackle"),
                     os.path.join(input_path, "Wheeze"), 
                     os.path.join(input_path, "Wheeze & Crackle"), 
                     os.path.join(input_path, "Healthy")]:
        
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

def check_and_create_directories(input_path):
    """
    Function to check if the necessary directories for training and testing sets exist, and create them if they do not exist. 
    This includes directories for each class (Crackle, Wheeze, Wheeze & Crackle, Healthy) within both the Train and Test directories.
    """
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
    Function to split patients into training and testing sets based on the provided respiratory cycles and random training patient IDs.
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
    Function to move the spectrogram images into the corresponding training and testing directories based on the assigned patient IDs for each set.
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
            elif id_patient in cycles_test:
                shutil.copy(spectrogram, os.path.join(input_path, 'Test', label))
            else:
                print(f"Patient ID {id_patient} not found in either training or testing sets.")