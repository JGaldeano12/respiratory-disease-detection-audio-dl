import nlpaug.augmenter.audio as naa
import cv2, cmapy, os, gc, numpy as np, librosa, multiprocessing, argparse

from scipy.signal import butter, lfilter
from src.data.preprocess import butter_bandpass_filter, check_length_and_padding
from src.features.extract_features import extract_features, save_features
from src.data.divide import split_patients_by_train_test

import warnings
warnings.filterwarnings("ignore")

def gen_augmented(original, sample_rate, seed = 20260131):
    """
    Generates augmented versions of the original audio segment. Includes one sample per technique + one combined sample.
    """
    # To ensure reproducibility, we set a random seed before applying the augmentation techniques, 
    # so that the same random transformations are applied each time the function is called with the same seed.
    rng = np.random.default_rng(seed)

    # Define the list of augmentation techniques to be applied.
    augment_types = ["time_shift", "time_stretch", "vtlp", "noise"]
    augmented_audio_list = []

    # For each augmentation technique, applied it to the original audio segment and store the augmented version in a list.
    for aug_type in augment_types:
        # Make a copy of the original audio segment to apply the augmentation technique, ensuring that the original audio remains unchanged for subsequent augmentations.
        augmented = original.copy()

        # Depending on the specified augmentation technique, apply the corresponding transformation to the audio segment:
        if aug_type == "time_shift":
            shift = rng.integers(len(augmented))
            augmented = np.roll(augmented, shift)

        elif aug_type == "time_stretch":
            rate = rng.uniform(0.97, 1.03)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)

        elif aug_type == "vtlp":
            np.random.seed(seed)
            vtlp = naa.VtlpAug(sampling_rate=sample_rate, zone=(0.4, 0.6), coverage=0.1)
            augmented = vtlp.augment(augmented)
            augmented = np.array(augmented).squeeze()

        elif aug_type == "noise":
            noise = rng.normal(0, 0.0005, len(augmented))
            augmented = augmented + noise

        # Ensure that the augmented audio segment has the same length as the original audio segment by either truncating or padding it as necessary.
        if len(augmented) != len(original):
            if len(augmented) > len(original):
                augmented = augmented[:len(original)]
            else:
                augmented = np.pad(augmented, (0, len(original) - len(augmented)))

        # After applying the augmentation technique and adjusting the length of the augmented audio segment, store the resulting 
        # augmented audio segment in the list of augmented audio segments for further processing or saving.
        augmented_audio_list.append(augmented.astype(np.float32))

    # Finally, create a combined augmented audio segment by applying a random combination of the augmentation techniques to the original audio segment,
    # and store the resulting combined augmented audio segment in the list of augmented audio segments for further processing or saving.
    combined = original.copy()

    # Apply a random combination of the augmentation techniques to the original audio segment to create a combined augmented audio segment,
    # where each augmentation technique is applied with a certain probability to introduce variability in the combined augmentation.
    if rng.random() < 0.7:
        shift = rng.integers(len(combined))
        combined = np.roll(combined, shift)

    if rng.random() < 0.5:
        rate = rng.uniform(0.97, 1.03)
        combined = librosa.effects.time_stretch(combined, rate=rate)

    if rng.random() < 0.3:
        np.random.seed(seed)
        vtlp = naa.VtlpAug(sampling_rate=sample_rate, zone=(0.4, 0.6), coverage=0.1)
        combined = vtlp.augment(combined)
        combined = np.array(combined).squeeze()

    if rng.random() < 0.3:
        noise = rng.normal(0, 0.0005, len(combined))
        combined = combined + noise

    # Ensure that the combined augmented audio segment has the same length as the original audio segment by either truncating or padding it as necessary.
    if len(combined) != len(original):
        if len(combined) > len(original):
            combined = combined[:len(original)]
        else:
            combined = np.pad(combined, (0, len(original) - len(combined)))

    # Finally store the resulting combined augmented audio segment in the list of augmented audio segments for further processing or saving, 
    # along with the individual augmented audio segments generated from each augmentation technique.
    augmented_audio_list.append(combined.astype(np.float32))

    # Return the list of augmented audio segments, which includes the individual augmented versions generated from each augmentation technique 
    # as well as the combined augmented version created by applying a random combination of the augmentation techniques to the original audio segment.
    return augmented_audio_list

def check_balanced_dataset(data_path):
    """
    Calculates the number of augmented spectrograms needed per class to balance the dataset.
    """
    # Define the classes used in the dataset.
    classes = ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']
    
    # Count the number of samples per class.
    counts = {c: len(os.listdir(os.path.join(data_path, c))) for c in classes}

    # Get the maximum number of samples among the classes to determine how many augmented spectrograms are needed for each class to balance the dataset,
    num_max = max(counts.values())
    
    # Return a tuple containing the number of augmented spectrograms needed for each class to balance the dataset, 
    # calculated as the difference between the maximum number of samples and the current count for each class.
    return tuple(num_max - counts[c] for c in classes)

def select_random_respiratory_cycles(category, rng, seed, train_test_split, control_file_path='/app/src/data/ciclos_respiratorios.npy'):
    """
    Function to select two random respiratory cycles for augmentation based on the specified category.
    """
    # Filter the respiratory cycles used for training based on the seed and train-test split.
    train_cycles, _ = split_patients_by_train_test(control_file_path, seed, train_test_split)

    # Precompute the duration mask once and reuse it for all class filters (threshold: 4.5 seconds).
    mask = train_cycles[:, 11].astype(float) < 4.5

    # Filter cycles by class and duration threshold, storing them in a dictionary for easy access.
    cycles = {
        'Healthy':          train_cycles[(train_cycles[:, 6] == 'Healthy')          & mask],
        'Crackle':          train_cycles[(train_cycles[:, 6] == 'Crackle')          & mask],
        'Wheeze':           train_cycles[(train_cycles[:, 6] == 'Wheeze')           & mask],
        'Wheeze & Crackle': train_cycles[(train_cycles[:, 6] == 'Wheeze & Crackle') & mask],
    }

    if category == 'Healthy':
        # For the Healthy class, select two random cycles exclusively from the healthy pool.
        indices = rng.choice(len(cycles['Healthy']), size=2, replace=False)
        regs = cycles['Healthy'][indices]

    elif category in ('Crackle', 'Wheeze'):
        main = cycles[category]

        # With 50% probability, mix one cycle from the target class with one healthy cycle.
        # Otherwise, select two cycles from the target class only.
        if rng.random() < 0.5:
            id_main    = rng.choice(len(main))
            id_healthy = rng.choice(len(cycles['Healthy']))
            indices = [id_main, id_healthy]
            regs    = [main[id_main], cycles['Healthy'][id_healthy]]
        else:
            indices = rng.choice(len(main), size=2, replace=False)
            regs    = main[indices]

    elif category == 'Wheeze & Crackle':
        both = cycles['Wheeze & Crackle']

        # Ensure there are enough cycles available for augmentation.
        if len(both) == 0:
            raise ValueError("Not enough cycles available for 'Wheeze & Crackle' augmentation")

        # Build the list of valid combination options based on the availability of cycles in each class.
        opciones = [k for k, v in {
            'wheeze_crackle + crackle':        len(cycles['Crackle']) > 0,
            'wheeze_crackle + wheeze':         len(cycles['Wheeze']) > 0,
            'wheeze_crackle + healthy':        len(cycles['Healthy']) > 0,
            'wheeze_crackle + wheeze_crackle': len(both) > 1,
        }.items() if v]

        # Randomly select one of the valid combinations.
        opcion = rng.choice(opciones)

        if opcion == 'wheeze_crackle + wheeze_crackle':
            # Select two cycles from the Wheeze & Crackle pool.
            indices = rng.choice(len(both), size=2, replace=False)
            regs    = both[indices]
        else:
            # Map the selected combination to its corresponding secondary class pool.
            label_map = {
                'wheeze_crackle + crackle': 'Crackle',
                'wheeze_crackle + wheeze':  'Wheeze',
                'wheeze_crackle + healthy': 'Healthy',
            }
            secondary = cycles[label_map[opcion]]

            # Select one cycle from the Wheeze & Crackle pool and one from the secondary class pool.
            id_both      = rng.choice(len(both))
            id_secondary = rng.choice(len(secondary))
            indices = [id_both, id_secondary]
            regs    = [both[id_both], secondary[id_secondary]]

    else:
        raise ValueError(f"Invalid category '{category}'. Choose from: 'Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'.")

    # Return the indices and records of the selected respiratory cycles.
    return indices, regs

def CBA(category, duration, input_path, output_path, seed = 20260131, train_test_split = 80, control_file_path = '/app/src/data/ciclos_respiratorios.npy'):
    """
    Function to perform Class-Based Augmentation (CBA) for a specified class by selecting random respiratory cycles, applying augmentation techniques, 
    and generating augmented spectrograms for the selected class.
    """
    # Obtain the indices and records for the selected respiratory cycles
    indices, regs = select_random_respiratory_cycles(category, seed = seed, train_test_split = train_test_split, control_file_path = control_file_path)

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
        audio_segment = butter_bandpass_filter(segment, 50, 2500, 4096, order=5)

        # Add the processed audio segment to the list of audios for augmentation.
        audios.append(audio_segment)

    # Now, create the index for the concatenated audio
    index_generated = str(indices[0]) + "_" + str(indices[1])

    # Concat the two audio segments.
    concat_audio = np.concatenate((audios[0], audios[1]))

    # Defino la duración objetivo
    target_length = int(duration * sr)

    # # Check wether the audio segment is shorter than the target duration. If so, we will apply padding to reach the desired length.
    # # Why do we use 0 as the start sample? Mainly, because we want to keep the entire augmented audio segment.
    # concat_audio_padded = check_length_and_padding(concat_audio, start_sample = 0, target_length = target_length)
    concat_audio_padded = check_length_and_padding(concat_audio, target_length)

    # Generate the Mel Spectrogram for the audio segment:
    extract_features(concat_audio_padded, output_path, label = category, patient = index_generated, index_cycle = index_generated, type = "CBA")

def apply_CBA_by_category(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 8, number = 1, category = 'Default', seed = 20260131, train_test_split = 80, control_file_path = '/app/src/data/ciclos_respiratorios.npy'):
    """
    Function to apply Class-Based Augmentation (CBA) for a specified class by calling the CBA function a certain number of times 
    to generate augmented spectrograms for the selected class.
    """
    # For the specified number of times, call the CBA function to perform Class-Based Augmentation (CBA) for the specified class and 
    # generate augmented spectrograms for that class.
    for i in range(number):
        # Apply Class-Based Augmentation (CBA) for the specified class
        CBA(category, duration, raw_audio_path, output_path, seed, train_test_split, control_file_path)

    return f"CBA applied for {category} class. Augmented spectrograms generated and saved to the specified output path."

def apply_CBA(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 8, seed = 20260131, train_test_split = 80, control_file_path = '/app/src/data/ciclos_respiratorios.npy'):
    """
    Function to apply Class-Based Augmentation (CBA) for all classes by calling the apply_CBA_by_category function for each class with the corresponding parameters.
    """
    # First, obtain the number of spectrograms that need to be augmented for each class.
    num_healthy, num_crackle, num_wheeze, num_both = check_balanced_dataset(output_path)

    # Now, apply Class-Based Augmentation (CBA) for each class by calling the apply_CBA_by_category function:
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_healthy, category = 'Healthy', seed = seed, train_test_split = train_test_split, control_file_path = control_file_path)
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_crackle, category = 'Crackle', seed = seed, train_test_split = train_test_split, control_file_path = control_file_path)
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_wheeze, category = 'Wheeze', seed = seed, train_test_split = train_test_split, control_file_path = control_file_path)
    apply_CBA_by_category(raw_audio_path, output_path, duration, num_both, category = 'Wheeze & Crackle', seed = seed, train_test_split = train_test_split, control_file_path = control_file_path)

    return "CBA applied for all classes. Augmented spectrograms generated and saved to the specified output path."

# Now, we want to apply traditional augmentation.
# First, we need to check how many spectrograms we need to augment to keep the classes balanced. It will be the minimum number of original respiratory cycles in training.
# After that, we will sample the .WAV files without repeating them until we reach the number of spectrograms to be augmented for each class.
# Finally, we apply the traditional augmentation techniques.
def select_balanced_and_random_respiratory_cycles(control_file_path = '/app/src/data/ciclos_respiratorios.npy', seed = 20260131, train_test_split = 80):
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

def apply_traditional_augmentation(raw_audio, sample_rate = 4096, length = 7, cycle = None, output_path = '/app/data/processed/Train'):
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
            augmentation_type = "TimeShift"
        elif aux_augmentation_technique == 1:
            augmentation_type = "TimeStretch"
        elif aux_augmentation_technique == 2:
            augmentation_type = "VtlpAug"
        elif aux_augmentation_technique == 3:
            augmentation_type = "Noise"
        elif aux_augmentation_technique == 4:
            augmentation_type = "Combined"
        else:
            augmentation_type = "UnknownAug"

        # Apply the Butterworth bandpass filter and standardize the audio segment
        segmented_audio = butter_bandpass_filter(augmented_audio, 50, 2000, sample_rate, order=5)

        # Check wether the audio segment is shorter than the target duration. If so, we will apply padding to reach the desired length.
        segmented_audio = check_length_and_padding(segmented_audio, target_length)

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

# # # Example usage of the apply_CBA function to perform Class-Based Augmentation (CBA) for all classes and generate augmented spectrograms.
# # num_healthy, num_crackle, num_wheeze, num_both = check_balanced_dataset('/app/data/processed/Train')
# # print(f"Number of spectrograms to be augmented for each class: Healthy: {num_healthy}, Crackle: {num_crackle}, Wheeze: {num_wheeze}, Wheeze & Crackle: {num_both}")

# # # Finally, apply Class-Based Augmentation (CBA) for all classes and generate augmented spectrograms.
# # apply_CBA(raw_audio_path = '/app/data/raw', output_path = '/app/data/processed/Train', duration = 7, seed = 20260131, train_test_split = 80, control_file_path = '/app/src/data/ciclos_respiratorios.npy')

# # After that, we apply traditional augmentation techniques to augment the training set. 
# # We start by selecting the respiratory cycles to be augmented, which will be the same number for each class to keep the dataset balanced.
# cycles_selected = select_balanced_and_random_respiratory_cycles(control_file_path = '/app/src/data/ciclos_respiratorios.npy', seed = 20260131, train_test_split = 80)

# # Now, we apply the traditional augmentation techniques to the selected respiratory cycles in parallel.
# apply_traditional_augmentation_lectura_datos_parallel(input_path = '/app/data/raw', output_path = '/app/data/processed/Train', length = 8, cycles = cycles_selected)

parser = argparse.ArgumentParser(description='Augment dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=20260119, help='Random seed for train-test split')
args = parser.parse_args()

cycles_selected = select_balanced_and_random_respiratory_cycles(control_file_path='/app/src/data/ciclos_respiratorios.npy', seed=args.seed, train_test_split=80)
apply_traditional_augmentation_lectura_datos_parallel(input_path='/app/data/raw', output_path='/app/data/processed/Train', length=8, cycles=cycles_selected)