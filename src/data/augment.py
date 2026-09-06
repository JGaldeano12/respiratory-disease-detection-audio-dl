import nlpaug.augmenter.audio as naa
import cv2, cmapy, os, gc, numpy as np, librosa, multiprocessing, argparse

from scipy.signal import butter, lfilter
from src.data.preprocess import butter_bandpass_filter, check_length_and_padding
from src.features.extract_features import extract_features, save_features
from src.data.divide import split_patients_by_train_test

import warnings
warnings.filterwarnings("ignore")

def gen_augmented(original, sample_rate, rng, seed):
    """
    Generates augmented versions of the original audio segment. Includes one sample per technique + one combined sample.
    """
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

##########################################################################################
# The following functions are used to apply Class-Based Augmentation (CBA).
##########################################################################################

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

def CBA(category, duration, input_path, output_path, rng, seed=202506, train_test_split=80, control_file_path='/app/src/data/ciclos_respiratorios.npy'):
    """
    Performs Class-Based Augmentation (CBA) for a specified class by selecting two random respiratory 
    cycles, concatenating them, and generating an augmented spectrogram.
    """
    # Select two random respiratory cycles from the training set for the specified category.
    indices, regs = select_random_respiratory_cycles(category, rng=rng, seed=seed, train_test_split=train_test_split, control_file_path=control_file_path)

    # Load, segment, and filter each selected respiratory cycle.
    audios = []
    for cycle in regs:
        # Load the audio file at the specified sampling rate.
        audio, sr = librosa.load(os.path.join(input_path, cycle[1]) + ".wav", sr=8000)

        # Extract the respiratory cycle segment based on its start and end timestamps.
        start = int(float(cycle[2]) * sr)
        end   = int(float(cycle[3]) * sr)
        segment = audio[start:end]

        # Apply a Butterworth bandpass filter to remove out-of-band noise.
        audio_segment = butter_bandpass_filter(segment, 50, 2000, 8000, order=5)
        audios.append(audio_segment)

    # Build a unique identifier from the indices of the two selected cycles.
    index_generated = f"{indices[0]}_{indices[1]}"

    # Concatenate the two audio segments and adjust to the target duration.
    concat_audio = np.concatenate((audios[0], audios[1]))
    target_length = int(duration * sr)
    concat_audio_padded = check_length_and_padding(concat_audio, target_length)

    # Extract and save the Mel Spectrogram for the concatenated audio segment.
    extract_features(concat_audio_padded, output_path, label=category, patient=index_generated, index_cycle=index_generated, type="CBA")

def apply_CBA_by_category(rng, category, number, raw_audio_path='/app/data/raw', output_path='/app/data/processed/Train', duration=6, seed=202506, train_test_split=80, control_file_path='/app/src/data/ciclos_respiratorios.npy'):
    """
    Applies Class-Based Augmentation (CBA) for a specified class a given number of times.
    """
    for _ in range(number):
        CBA(category, duration, raw_audio_path, output_path, rng=rng, seed=seed, train_test_split=train_test_split, control_file_path=control_file_path)

def apply_CBA(rng, raw_audio_path='/app/data/raw', output_path='/app/data/processed/Train', duration=6, seed=202506, train_test_split=80, control_file_path='/app/src/data/ciclos_respiratorios.npy'):
    """
    Applies Class-Based Augmentation (CBA) for all classes, generating the number of spectrograms
    needed to balance the dataset.
    """
    # Calculate the number of augmented spectrograms needed per class to balance the dataset.
    num_healthy, num_crackle, num_wheeze, num_both = check_balanced_dataset(output_path)

    # Apply CBA for each class with the corresponding number of augmentations needed.
    for category, number in zip(
        ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle'],
        [num_healthy, num_crackle, num_wheeze, num_both]
    ):
        apply_CBA_by_category(rng, category, number, raw_audio_path, output_path, duration, seed, train_test_split, control_file_path)

##########################################################################################
# The following functions are used to apply traditional audio augmentation techniques.
##########################################################################################
def select_balanced_and_random_respiratory_cycles(rng, control_file_path='/app/src/data/ciclos_respiratorios.npy', seed=202506, train_test_split=80):
    """
    Selects a balanced random subset of respiratory cycles for traditional augmentation,
    sampling the same number of cycles from each class.
    """
    # Filter the respiratory cycles used for training.
    cycles_train, _ = split_patients_by_train_test(aux_file=control_file_path, seed=seed, train_test_split=train_test_split)

    # Calculate the minimum number of cycles across all classes to ensure a balanced selection.
    _, counts = np.unique(cycles_train[:, 6], return_counts=True)
    num_augmented_per_class = min(counts)

    # For each class, select a random subset of cycles of size num_augmented_per_class.
    classes = ['Healthy', 'Crackle', 'Wheeze', 'Wheeze & Crackle']
    selected = [cycles_train[cycles_train[:, 6] == c][rng.choice(np.sum(cycles_train[:, 6] == c), size=num_augmented_per_class, replace=False)]
        for c in classes
    ]

    # Concatenate the selected cycles from all classes into a single array.
    return np.concatenate(selected, axis=0)

def apply_traditional_augmentation_process_file(cycle, input_path, output_path, length, rng, seed):
    """
    Processes a single .WAV file by loading, segmenting, and applying traditional augmentation techniques.
    Input must be a single row from the cycles array.
    """
    try:
        # Load the audio file at the specified sampling rate.
        raw_audio, sr = librosa.load(os.path.join(input_path, cycle[1] + ".wav"), sr=4096)

        # Apply traditional augmentation to the loaded audio segment.
        apply_traditional_augmentation(raw_audio=raw_audio, cycle=cycle, length=length, sample_rate=sr, output_path=output_path, rng=rng, seed=seed)

    except Exception as e:
        print(f"Error while processing {cycle[1]}: {e}")

def apply_traditional_augmentation_lectura_datos_parallel(input_path, output_path, length, cycles, rng, seed):
    """
    Processes all selected .WAV files in parallel, applying traditional augmentation techniques to each.
    """
    # Build the argument tuples, passing the shared seed to each worker.
    argumentos = [(cycle, input_path, output_path, length, rng, seed) for cycle in cycles]

    # Launch a pool of workers to process files in parallel.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(apply_traditional_augmentation_process_file, argumentos)

def apply_traditional_augmentation(raw_audio, rng, seed, sample_rate=4096, length=8, cycle=None, output_path='/app/data/processed/Train'):
    """
    Applies all traditional augmentation techniques to a single respiratory cycle and saves the resulting spectrograms.
    """
    # Define the target length and the augmentation type labels in order.
    target_length = int(length * sample_rate)
    augmentation_types = ["TimeShift", "TimeStretch", "VtlpAug", "Noise", "Combined"]

    # Extract the respiratory cycle segment from the raw audio.
    start = int(float(cycle[2]) * sample_rate)
    end   = int(float(cycle[3]) * sample_rate)
    segm  = raw_audio[start:end]

    # Generate all augmented versions of the segment.
    augmented_audios = gen_augmented(segm, sample_rate, rng=rng, seed=seed)

    # Build the cycle index string from the start and end timestamps.
    index_cycle = f"{cycle[2].replace('.', '-')}_{cycle[3].replace('.', '-')}"

    # For each augmented version, apply filtering, padding, and feature extraction.
    for augmented_audio, augmentation_type in zip(augmented_audios, augmentation_types):

        # Apply the Butterworth bandpass filter to remove out-of-band noise.
        segmented_audio = butter_bandpass_filter(augmented_audio, 50, 2000, sample_rate, order=5)

        # Adjust the audio segment to the target duration via truncation or padding.
        segmented_audio = check_length_and_padding(segmented_audio, target_length)

        # Extract and save the Mel Spectrogram for the augmented audio segment.
        extract_features(segmented_audio, output_path=output_path, label=cycle[6], patient=cycle[1], index_cycle=index_cycle, type=augmentation_type)

    # Free memory after processing the audio file.
    gc.collect()

##########################################################################################
# CBA is first applied to balance the dataset. Then, a balanced random subset of respiratory 
# cycles is selected for traditional augmentation, which is applied in parallel to speed up the process.
##########################################################################################
parser = argparse.ArgumentParser(description='Augment dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=202506, help='Random seed for reproducibility.')
args = parser.parse_args()

# Create a single rng from the provided seed, shared across all augmentation steps.
rng = np.random.default_rng(args.seed)

# # Apply Class-Based Augmentation (CBA) for all classes.
# apply_CBA(rng=rng, raw_audio_path='/app/data/raw', output_path='/app/data/processed/Train', duration=8, seed=args.seed, train_test_split=80, control_file_path='/app/src/data/ciclos_respiratorios.npy')

# Select a balanced set of respiratory cycles for traditional augmentation.
cycles_selected = select_balanced_and_random_respiratory_cycles(rng=rng, control_file_path='/app/src/data/ciclos_respiratorios.npy', seed=args.seed, train_test_split=80)

# Apply traditional augmentation techniques to the selected cycles in parallel.
apply_traditional_augmentation_lectura_datos_parallel(input_path='/app/data/raw', output_path='/app/data/processed/Train', length=8, cycles=cycles_selected, rng=rng, seed=args.seed)