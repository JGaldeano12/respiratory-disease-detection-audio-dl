import nlpaug.augmenter.audio as naa
import cv2, cmapy, os, gc, numpy as np, librosa, multiprocessing

from scipy.signal import butter, lfilter
from src.data.preprocess import butter_bandpass_filter, standardize_audio, check_length_and_padding
from src.features.extract_features import extract_features, save_features
from src.data.divide import split_patients_by_train_test

import warnings
warnings.filterwarnings("ignore")

def gen_augmented(original, sample_rate):
    """
    Function to generate augmented versions of the original audio segment.
    """
    # Define a list of audio augmentation techniques to be applied to the original audio segment.
    # These techniques include adding noise, changing loudness, and applying vocal tract length perturbation (VTLP).
    augment_list = [
        naa.NoiseAug(),
        naa.LoudnessAug(factor=(0.5, 2)),
        naa.VtlpAug(sampling_rate=sample_rate, zone=(0.0, 1.0)),
    ]

    # List to store the augmented audio segments generated from the original audio segment.
    augmented_audio_list = []

    # Apply each augmentation technique from the list to the original audio segment and store the augmented versions in the audios_aumentados list.
    for tecnica in augment_list:
        augmented = tecnica.augment(original)
        augmented_audio_list.append(augmented)

    # Return the list of augmented audio segments generated from the original audio segment.
    return augmented_audio_list

def check_balanced_dataset(data_path):
    """
    Since each type of augmentation will be applied for a specific purpose, we need to calculate
    the number of augmented spectrograms that we need to generate for each class in order to balance the dataset.
    """
    # Obtain the list of spectrograms for each class.
    healthy = [os.path.join(data_path, f) for f in os.listdir(os.path.join(data_path, 'Healthy'))]
    crackle = [os.path.join(data_path, f) for f in os.listdir(os.path.join(data_path, 'Crackle'))]
    wheeze = [os.path.join(data_path, f) for f in os.listdir(os.path.join(data_path, 'Wheeze'))]
    both = [os.path.join(data_path, f) for f in os.listdir(os.path.join(data_path, 'Wheeze & Crackle'))]

    # Calculate the number of images that I need to generate to balance the classes, which will be equal to the max number of spectrograms for any class.
    num_max_reg = np.max([len(healthy), len(crackle), len(wheeze), len(both)])

    # Calculate the number of records that need to be augmented for each class.
    num_reg_aug_healthy = num_max_reg - len(healthy)
    num_reg_aug_crackle = num_max_reg - len(crackle)
    num_reg_aug_wheeze = num_max_reg - len(wheeze)
    num_reg_aug_both = num_max_reg - len(both)

    # Return the number of records that need to be augmented for each class to achieve a balanced dataset.
    return num_reg_aug_healthy, num_reg_aug_crackle, num_reg_aug_wheeze, num_reg_aug_both

def select_random_respiratory_cycles(category, control_file_path = '/app/src/data/ciclos_respiratorios_train.npy'):
    """
    Function to select two random respiratory cycles.
    """
    # Load the file with the labeled cycles
    cycles = np.load(control_file_path)

    # Filter the respiratory cycles based on the specified class and a 
    # duration threshold of 4.5 seconds to ensure that only relevant cycles are selected for augmentation.
    cycles_healthy = cycles[ (cycles[:, 6] == 'Healthy') & (cycles[:, 11].astype(float) < 4.5)]
    cycles_crackle = cycles[ (cycles[:, 6] == 'Crackle') & (cycles[:, 11].astype(float) < 4.5)]
    cycles_wheeze= cycles[ (cycles[:, 6] == 'Wheeze') & (cycles[:, 11].astype(float) < 4.5)]
    cycles_both= cycles[ (cycles[:, 6] == 'Wheeze & Crackle') & (cycles[:, 11].astype(float) < 4.5)]

    # Depending on the specified class, select two random respiratory cycles from the corresponding filtered list of cycles.
    if category == 'Healthy':
        # Select two random respiratory cycles from the healthy class and return their indices and records for augmentation.
        # Since the healthy class needs to be augmented by only two healthy cycles, we only sample healthy ones.
        indices = np.random.choice(len(cycles_healthy), size=2, replace=False)
        regs = cycles_healthy[indices]

    # After that, for "Crackle" and "Wheeze" classes, we can either select two random cycles from the same class or a mix with one healthy cycle.
    elif category == 'Crackle':
        # With a 50% chance, select one random cycle from the crackle class and one random cycle from the healthy class to create a mixed augmentation, 
        # or select two random cycles from the crackle class for augmentation.
        if np.random.rand() < 0.5:
            # Select one random cycle with crackle and one random cycle from healthy class
            id_crackle = np.random.choice(len(cycles_crackle))
            id_healthy = np.random.choice(len(cycles_healthy))

            # Select the records corresponding to the chosen indices for augmentation
            registro_crackle = cycles_crackle[id_crackle]
            registro_healthy = cycles_healthy[id_healthy]

            # Combine the selected crackle and healthy records into a list for augmentation
            regs = [registro_crackle, registro_healthy]

            # Store the indices of the selected crackle and healthy cycles for reference during augmentation
            indices = [id_crackle, id_healthy]
        
        # As mentioned, the alternative is to select two random cycles from the crackle class for augmentation.
        else:
            # Select the two random cycles from crackle class.
            indices = np.random.choice(len(cycles_crackle), size=2, replace=False)
            regs = cycles_crackle[indices]

    # Same thing with "Wheeze" class, where we can either select two random cycles from the wheeze class or a mix with one healthy cycle for augmentation.
    elif category == 'Wheeze':
        # First, mix between Wheeze and Healthy class.
        if np.random.rand() < 0.5:
            # Select one random cycle with wheeze and one random cycle from healthy class
            id_wheeze = np.random.choice(len(cycles_wheeze))
            id_healthy = np.random.choice(len(cycles_healthy))

            # Select the records corresponding to the chosen indices for augmentation
            registro_wheeze = cycles_wheeze[id_wheeze]
            registro_healthy = cycles_healthy[id_healthy]

            # Combine the selected wheeze and healthy records into a list for augmentation
            regs = [registro_wheeze, registro_healthy]
            
            # Store the indices of the selected wheeze and healthy cycles for reference during augmentation
            indices = [id_wheeze, id_healthy]
        else:
            # Select two random cycles from the wheeze class
            indices = np.random.choice(len(cycles_wheeze), size=2, replace=False)
            regs = cycles_wheeze[indices]

    # Finally, for the "Wheeze & Crackle" class, we can generate a random combination of cycles from the crackle, wheeze, healthy, and both classes for augmentation,
    # or select two random cycles from the "Wheeze & Crackle" class for augmentation, depending on the available cycles in each class and a random choice.
    elif category == 'Wheeze & Crackle':
        
        # Generate a list of possible combinations.
        opciones = []

        # Depending on the availability of cycles in each class, add different combinations of classes to the list of options for augmentation.
        if len(cycles_both) > 0:
            if len(cycles_crackle) > 0:
                opciones.append('wheeze_crackle + crackle')
            if len(cycles_wheeze) > 0:
                opciones.append('wheeze_crackle + wheeze')
            if len(cycles_healthy) > 0:
                opciones.append('wheeze_crackle + healthy')
            if len(cycles_both) > 1:
                opciones.append('wheeze_crackle + wheeze_crackle')

        # If there are available options for augmentation, randomly select one of the combinations from the list of options and 
        # proceed with selecting the corresponding respiratory cycles for augmentation based on the chosen combination.
        if len(opciones) > 0:
            opcion = np.random.choice(opciones)

            # Depending on the randomly selected combination, select the corresponding respiratory cycles from the appropriate classes for augmentation 
            # and store their indices and records for reference during augmentation.
            if opcion == 'wheeze_crackle + crackle':
                # Mix between "Wheeze & Crackle" and "Crackle" classes by selecting one random cycle from each class for augmentation.
                id_both = np.random.choice(len(cycles_both))
                id_crackle = np.random.choice(len(cycles_crackle))

                # Select the records corresponding to the chosen indices for augmentation
                reg_wheeze_crackle = cycles_both[id_both]
                reg_crackle = cycles_crackle[id_crackle]

                # Combine the selected "Wheeze & Crackle" and "Crackle" records into a list for augmentation
                regs = [reg_wheeze_crackle, reg_crackle]

                # Store the indices of the selected "Wheeze & Crackle" and "Crackle" cycles for reference during augmentation
                indices = [id_both, id_crackle]

            # Same thing with "Wheeze & Crackle" and "Wheeze" classes, where we can mix between them by selecting one random cycle from each class for augmentation.
            elif opcion == 'wheeze_crackle + wheeze':
                # Mix between "Wheeze & Crackle" and "Wheeze" classes by selecting one random cycle from each class for augmentation.
                id_both = np.random.choice(len(cycles_both))
                id_wheeze = np.random.choice(len(cycles_wheeze))

                # Select the records corresponding to the chosen indices for augmentation
                reg_wheeze_crackle = cycles_both[id_both]
                reg_wheeze = cycles_wheeze[id_wheeze]

                # Combine the selected "Wheeze & Crackle" and "Wheeze" records into a list for augmentation
                regs = [reg_wheeze_crackle, reg_wheeze]

                # Store the indices of the selected "Wheeze & Crackle" and "Wheeze" cycles for reference during augmentation
                indices = [id_both, id_wheeze]

            # Same thing with "Wheeze & Crackle" and "Healthy" classes, where we can mix between them by selecting one random cycle from each class for augmentation.
            elif opcion == 'wheeze_crackle + healthy':
                # Mix between "Wheeze & Crackle" and "Healthy" classes by selecting one random cycle from each class for augmentation.
                id_both = np.random.choice(len(cycles_both))
                id_healthy = np.random.choice(len(cycles_healthy))

                # Select the records corresponding to the chosen indices for augmentation
                reg_wheeze_crackle = cycles_both[id_both]
                reg_healthy = cycles_healthy[id_healthy]

                # Combine the selected "Wheeze & Crackle" and "Healthy" records into a list for augmentation
                regs = [reg_wheeze_crackle, reg_healthy]

                # Store the indices of the selected "Wheeze & Crackle" and "Healthy" cycles for reference during augmentation
                indices = [id_both, id_healthy]

            elif opcion == 'wheeze_crackle + wheeze_crackle':
                # Randomly select two cycles from the "Wheeze & Crackle" class for augmentation.
                indices = np.random.choice(len(cycles_both), size=2, replace=False)
                regs = cycles_both[indices]
        else:
            raise ValueError("Not enough cycles available for 'Wheeze & Crackle' augmentation")
    
    # Finally, if an invalid category is specified, return an error message indicating that the category is not valid and providing the valid options for augmentation.
    else:
        return "Error: Invalid category specified. Please choose from 'Healthy', 'Crackle', 'Wheeze', or 'Wheeze & Crackle'."

    # Return the indices and records of the selected respiratory cycles for augmentation based on the specified class and random selection criteria.
    return indices, regs

def CBA(category, duration, input_path, output_path):
    """
    Function to perform Class-Based Augmentation (CBA) for a specified class by selecting random respiratory cycles, applying augmentation techniques, 
    and generating augmented spectrograms for the selected class.
    """
    # Obtain the indices and records for the selected respiratory cycles
    indices, regs = select_random_respiratory_cycles(category)

    # Create list to store the audio segments.
    audios = []

    # Read the audio files and add them to the list.
    for cycle in regs:
        # Load the audio file corresponding to the current respiratory cycle using librosa, specifying the sampling rate (sr) for consistent processing.
        audio, sr = librosa.load(os.path.join(input_path, cycle[1]) + ".wav", sr = 4096)
        
        # Define start and end of the cycle.
        start = int(float(cycle[2]) * sr)
        end = int(float(cycle[3]) * sr)

        # Segment the audio.
        segment = audio[start:end]

        # Apply a Butterworth bandpass filter to the segmented audio.
        audio_segment = butter_bandpass_filter(segment, 50, 2000, 4096, order=5)
        
        # Apply standardization to the filtered audio segment.
        audio_segment = standardize_audio(audio_segment)

        # Add the processed audio segment to the list of audios for augmentation.
        audios.append(audio_segment)

    # Now, create the index for the concatenated audio
    index_generated = str(indices[0]) + "_" + str(indices[1])

    # Concat the two audio segments.
    concat_audio = np.concatenate((audios[0], audios[1]))

    # Defino la duración objetivo
    target_length = int(duration * sr)

    # Check wether the audio segment is shorter than the target duration. If so, we will apply padding to reach the desired length.
    # Why do we use 0 as the start sample? Mainly, because we want to keep the entire augmented audio segment.
    concat_audio_padded = check_length_and_padding(concat_audio, start_sample = 0, target_length = target_length)

    # Generate the Mel Spectrogram for the audio segment:
    extract_features(concat_audio_padded, output_path, label = category, patient = index_generated, index_cycle = index_generated, type = "CBA")

def apply_CBA_by_category(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 6, number = 1, category = 'Default'):
    """
    Function to apply Class-Based Augmentation (CBA) for a specified class by calling the CBA function a certain number of times 
    to generate augmented spectrograms for the selected class.
    """
    # For the specified number of times, call the CBA function to perform Class-Based Augmentation (CBA) for the specified class and 
    # generate augmented spectrograms for that class.
    for i in range(number):
        # Aplico el aumento para la clase 'Healthy'
        CBA(category = category, duration = duration, input_path = raw_audio_path, output_path = output_path)

def apply_CBA(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 6):
    """
    Function to apply Class-Based Augmentation (CBA) for all classes by calling the apply_CBA_by_category function for each class with the corresponding parameters.
    """
    # First, obtain the number of spectrograms that need to be augmented for each class.
    num_healthy, num_crackle, num_wheeze, num_both = check_balanced_dataset(output_path)

    # Now, apply Class-Based Augmentation (CBA) for each class by calling the apply_CBA_by_category function:
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_healthy, category = 'Healthy')
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_crackle, category = 'Crackle')
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_wheeze, category = 'Wheeze')
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_both, category = 'Wheeze & Crackle')

    return "CBA applied for all classes. Augmented spectrograms generated and saved to the specified output path."

# Now, we want to apply traditional augmentation.
# First, we need to check how many spectrograms we need to augment to keep the classes balanced. It will be the minimum number of original respiratory cycles in training.
# After that, we will sample the .WAV files without repeating them until we reach the number of spectrograms to be augmented for each class.
# Finally, we apply the traditional augmentation techniques.
def select_balanced_and_random_respiratory_cycles(control_file_path = '/app/src/data/ciclos_respiratorios.npy', seed = 20251231, train_test_split = 80):
    """
    Function to select random respiratory cycles for traditional augmentation while ensuring a balanced selection across classes.
    """
    # Filter the respiratory cycles used for training.
    cycles_train, _ = split_patients_by_train_test(aux_file = control_file_path, seed = seed, train_test_split = train_test_split)

    # Count the number of rows for each label
    unique, counts = np.unique(cycles_train[:, 6], return_counts=True)
    
    # # Print the distribution:
    # print(f"Distribution of respiratory cycles in training set: {dict(zip(unique, counts))}")

    # Now, calculate the minimun number of cycles in training for every class, so we can apply the same number of traditional augmentations for every class to balance the dataset.
    num_augmented_per_class = min(counts)

    # print(f"Number of respiratory cycles to be augmented for each class: {num_augmented_per_class}")

    # Depending on the specified class, select random respiratory cycles for each class.
    cycles_healthy = cycles_train[ (cycles_train[:, 6] == 'Healthy')]
    indices = np.random.choice(len(cycles_healthy), size=num_augmented_per_class, replace=False)
    cycles_healthy = cycles_healthy[indices]

    # print(f'Selected respiratory cycles for Healthy class: {len(cycles_healthy)}')

    cycles_crackle = cycles_train[ (cycles_train[:, 6] == 'Crackle')]
    indices = np.random.choice(len(cycles_crackle), size=num_augmented_per_class, replace=False)
    cycles_crackle = cycles_crackle[indices]

    # print(f'Selected respiratory cycles for Crackle class: {len(cycles_crackle)}')

    cycles_wheeze= cycles_train[ (cycles_train[:, 6] == 'Wheeze')]
    indices = np.random.choice(len(cycles_wheeze), size=num_augmented_per_class, replace=False)
    cycles_wheeze = cycles_wheeze[indices]

    # print(f'Selected respiratory cycles for Wheeze class: {len(cycles_wheeze)}')

    cycles_both= cycles_train[ (cycles_train[:, 6] == 'Wheeze & Crackle')]
    indices = np.random.choice(len(cycles_both), size=num_augmented_per_class, replace=False)
    cycles_both = cycles_both[indices]

    # print(f'Selected respiratory cycles for Wheeze & Crackle class: {len(cycles_both)}')

    # Concat all the selected cycles into a single numpy array:
    cycles_selected = np.concatenate((cycles_healthy, cycles_crackle, cycles_wheeze, cycles_both), axis=0)

    # print(f'Total number of respiratory cycles selected for traditional augmentation: {len(cycles_selected)}')

    # Finally, return the selected respiratory cycles for each class to be used for traditional augmentation.
    return cycles_selected

def apply_traditional_augmentation_process_file(cycle, input_path, output_path, length):
    """
    Function to process a single .WAV file, previously selected for traditional augmentation.
    Input must be a single row from the list.
    """
    try:
        # Get unique record ID from the second column of the row.
        base_filename = cycle[1]

        # Load the audio file using librosa
        raw_audio, sr = librosa.load(os.path.join(input_path, base_filename + ".wav"), sr = 4096)

        # Proceed to audio segmentation
        apply_traditional_augmentation(raw_audio = raw_audio, cycle = cycle, length = length, sample_rate = sr, output_path = output_path)
        
    except Exception as e:
        print(f"Error while processing {input_path}: {e}")

def apply_traditional_augmentation_lectura_datos_parallel(input_path, output_path, length, cycles):
    """
    Function to read the data and process .wav files in parallel.
    """
    # Create tuples of parameters for each file
    argumentos = [(cycle, input_path, output_path, length) for cycle in cycles]

    # Use multiprocessing to parallelize the processing of files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(apply_traditional_augmentation_process_file, argumentos)

    return "Processing completed for all files."

def apply_traditional_augmentation(raw_audio, sample_rate = 4096, length = 6, cycle = None, output_path = '/app/data/processed/Train'):
    """
    Function to apply traditional audio augmentation techniques to the input audio segment.
    It will only apply to the patients used for training.
    """
    # For every respiratory cycle, I will apply some traditional audio augmentation techniques:
    target_length = int(length * sample_rate)

    # Also, auxiliar variable to keep track of the augmentation technique:
    aux_augmentation_technique = 0

    # Define the start and end of the segment in terms of samples
    start = int(float(cycle[2]) * sample_rate)
    end = int(float(cycle[3]) * sample_rate)

    # Before augmenting, we divide the audio
    segm = raw_audio[start:end]

    # Now, we apply the traditional augmentation techniques to the segmented audio and generate augmented versions of the original audio segment.
    augmented_audios = gen_augmented(segm, sample_rate)

    # For each augmented audio segment generated from the original audio segment, we will apply the same preprocessing steps
    for augmented_audio in augmented_audios:
        # First, check the augmentation techique:
        if aux_augmentation_technique == 0:
            augmentation_type = "NoiseAug"
        elif aux_augmentation_technique == 1:
            augmentation_type = "LoudnessAug"
        elif aux_augmentation_technique == 2:
            augmentation_type = "VtlpAug"
        else:
            augmentation_type = "UnknownAug"

        # Apply the Butterworth bandpass filter and standardize the audio segment
        segmented_audio = butter_bandpass_filter(augmented_audio[0], 50, 2000, sample_rate, order=5)
        segmented_audio = standardize_audio(segmented_audio)

        # Check wether the audio segment is shorter than the target duration. If so, we will apply padding to reach the desired length.
        segmented_audio = check_length_and_padding(segmented_audio, start, target_length)

        # Generate the Mel Spectrogram for the audio segment:
        extract_features(segmented_audio, 
                         output_path = output_path, 
                         label = cycle[6], 
                         patient = cycle[1], 
                         index_cycle = str.replace(cycle[2], '.', '-') + '_' + str.replace(cycle[3], '.', '-'),
                         type = augmentation_type)

        # Increment the augmentation technique index for the next iteration
        aux_augmentation_technique += 1

    # Free memory after processing the audio file
    gc.collect()

# # Example usage of the apply_CBA function to perform Class-Based Augmentation (CBA) for all classes and generate augmented spectrograms.
# num_healthy, num_crackle, num_wheeze, num_both = check_balanced_dataset('/app/data/processed/Train')
# print(f"Number of spectrograms to be augmented for each class: Healthy: {num_healthy}, Crackle: {num_crackle}, Wheeze: {num_wheeze}, Wheeze & Crackle: {num_both}")

# # Now, exec the function to select two random respiratory cycles.
# indices, regs = select_random_respiratory_cycles(category = 'Crackle')
# print(f"Selected indices for augmentation: {indices}; Selected records for augmentation: {regs}")

# # Finally, apply Class-Based Augmentation (CBA) for all classes and generate augmented spectrograms.
# CBA(category = 'Crackle', duration = 6, input_path = '/app/data/raw', output_path = '/app/data/processed/Train')

# # Finally, apply Class-Based Augmentation (CBA) for all classes and generate augmented spectrograms.
# apply_CBA(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 6)

# # After that, we apply traditional augmentation techniques to augment the training set. 
# # We start by selecting the respiratory cycles to be augmented, which will be the same number for each class to keep the dataset balanced.
# cycles_selected = select_balanced_and_random_respiratory_cycles(control_file_path = '/app/src/data/ciclos_respiratorios.npy', seed = 20251231, train_test_split = 80)

# # Now, we apply the traditional augmentation techniques to the selected respiratory cycles in parallel.
# apply_traditional_augmentation_lectura_datos_parallel(input_path = '/app/data/raw', output_path = '/app/data/processed/Train', length = 6, cycles = cycles_selected)