import nlpaug.augmenter.audio as naa, os, gc, numpy as np, librosa, multiprocessing, argparse, warnings

from src.data.preprocess import butter_bandpass_filter, check_length_and_padding
from src.features.extract_features import extract_features
from src.data.divide import split_patients_by_train_test

warnings.filterwarnings("ignore")

def gen_augmented(original, sample_rate, rng, seed):
    """
    Generate multiple augmented versions of an audio segment.

    The function creates one augmented sample for each supported technique:
    time shifting, time stretching, vocal tract length perturbation (VTLP),
    and additive Gaussian noise. It also generates an additional sample by
    randomly combining these techniques. All generated samples are adjusted
    to match the length of the original audio segment.

    Args:
        original (np.ndarray): Original audio segment to augment.
        sample_rate (int): Sampling rate of the audio signal in Hz.
        rng (np.random.Generator): Random number generator used to sample
            augmentation parameters and probabilities.
        seed (int): Random seed used when applying VTLP augmentation.

    Returns:
        list[np.ndarray]: List containing the four individually augmented
        audio segments and one combined augmented segment.
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
    Calculate the number of additional samples required to balance each class.

    The function counts the files contained in the Healthy, Crackle, Wheeze,
    and Wheeze & Crackle directories and uses the largest class size as the
    target size. For each class, it returns the difference between the target
    size and the current number of samples.

    Args:
        data_path (str): Path containing the directories for all dataset
            classes.

    Returns:
        tuple[int, int, int, int]: Number of additional samples required for
        the Healthy, Crackle, Wheeze, and Wheeze & Crackle classes,
        respectively.
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

# Define all functions needed to apply traditional augmentation techniques to the dataset, 
# including selecting a balanced random subset of respiratory cycles, processing individual audio
# files, and applying augmentation techniques in parallel across multiple files.

def select_balanced_and_random_respiratory_cycles(rng, control_file_path='/app/src/resources/aux_respiratory_cycles.npy', seed=202506, train_test_split=80):
    """
    Select an equal number of respiratory cycles from each class for augmentation.

    The function first obtains the respiratory cycles assigned to the training
    set. It then determines the size of the smallest class and randomly selects
    that same number of cycles from each of the four classes, producing a
    balanced set of respiratory cycles.

    Args:
        rng (np.random.Generator): Random number generator used to select the
            cycles from each class.
        control_file_path (str, optional): Path to the NumPy file containing
            the respiratory cycle annotations. Defaults to
            '/app/src/resources/aux_respiratory_cycles.npy'.
        seed (int, optional): Random seed used for the patient-level
            train/test split. Defaults to 202506.
        train_test_split (float, optional): Percentage of patients assigned to
            the training set. Defaults to 80.

    Returns:
        np.ndarray: Concatenated array containing a balanced random selection
        of respiratory cycles from all classes.
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

def apply_traditional_augmentation_process_file(cycle, input_path, output_path, sample_rate, length, rng, seed):
    """
    Load an audio file and apply traditional augmentation to one respiratory cycle.

    The function receives a single respiratory cycle annotation, loads the
    corresponding WAV recording at 8000 Hz, and applies the traditional
    augmentation pipeline to the audio segment defined by that cycle.

    Args:
        cycle (np.ndarray): Single row containing the respiratory cycle
            annotation and associated metadata.
        input_path (str): Directory containing the original WAV recordings.
        output_path (str): Directory where the augmented features are saved.
        length (float): Target duration of the processed audio segments in
            seconds.
        rng (np.random.Generator): Random number generator used during
            augmentation.
        seed (int): Random seed used by augmentation operations that require
            deterministic behaviour.

    Returns:
        None
    """
    try:
        # Load the audio file at the specified sampling rate.
        raw_audio, sr = librosa.load(os.path.join(input_path, cycle[1] + ".wav"), sr=sample_rate)

        # Apply traditional augmentation to the loaded audio segment.
        apply_traditional_augmentation(raw_audio=raw_audio, cycle=cycle, length=length, sample_rate=sr, output_path=output_path, rng=rng, seed=seed)

    except Exception as e:
        print(f"Error while processing {cycle[1]}: {e}")

def apply_traditional_augmentation_lectura_datos_parallel(input_path, output_path, sample_rate, length, cycles, rng, seed):
    """
    Apply traditional augmentation to multiple respiratory cycles in parallel.

    A multiprocessing pool is created using all available CPU cores. Each
    respiratory cycle is processed independently by loading its corresponding
    audio file and executing the traditional augmentation pipeline.

    Args:
        input_path (str): Directory containing the original WAV recordings.
        output_path (str): Directory where the augmented features are saved.
        length (float): Target duration of the processed audio segments in
            seconds.
        cycles (np.ndarray): Array containing the respiratory cycles to
            process.
        rng (np.random.Generator): Random number generator passed to each
            processing task.
        seed (int): Random seed used during augmentation.

    Returns:
        None
    """
    # Build the argument tuples, passing the shared seed to each worker.
    argumentos = [(cycle, input_path, output_path, sample_rate, length, rng, seed) for cycle in cycles]

    # Launch a pool of workers to process files in parallel.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(apply_traditional_augmentation_process_file, argumentos)

def apply_traditional_augmentation(raw_audio, rng, seed, sample_rate=8000, length=8, cycle=None, output_path='/app/data/processed/Train'):
    """
    Generate and process augmented versions of a single respiratory cycle.

    The respiratory cycle is extracted from the input recording according to
    its start and end timestamps. Five augmented versions are generated,
    consisting of four individual augmentation techniques and one combined
    augmentation. Each version is filtered, adjusted to the target length,
    converted into features, and saved with the corresponding augmentation
    type.

    Args:
        raw_audio (np.ndarray): Original audio recording containing the
            respiratory cycle.
        rng (np.random.Generator): Random number generator used to sample
            augmentation parameters.
        seed (int): Random seed used by augmentation operations that require
            deterministic behaviour.
        sample_rate (int, optional): Sampling rate of the audio signal in Hz.
            Defaults to 4096.
        length (float, optional): Target duration of the processed audio
            segments in seconds. Defaults to 8.
        cycle (np.ndarray, optional): Respiratory cycle annotation containing
            the start and end times, patient identifier, and class label.
            Defaults to None.
        output_path (str, optional): Directory where the extracted features
            are saved. Defaults to '/app/data/processed/Train'.

    Returns:
        None
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
        extract_features(segmented_audio, sample_rate, output_path=output_path, label=cycle[6], patient=cycle[1], index_cycle=index_cycle, type=augmentation_type)

    # Free memory after processing the audio file.
    gc.collect()

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Augment dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=202506, help='Random seed for train-test split')
parser.add_argument('--sample_rate', type=int, default=8000, help='Sampling rate for audio processing')
parser.add_argument('--duration', type=int, default=8, help='Duration of each audio segment')
parser.add_argument('--test_train_split', type=int, default=80, help='Percentage of data for training')
args = parser.parse_args()

# Create a single rng from the provided seed, shared across all augmentation steps.
rng = np.random.default_rng(args.seed)

# Select a balanced set of respiratory cycles for traditional augmentation.
cycles_selected = select_balanced_and_random_respiratory_cycles(rng=rng, 
                                                                control_file_path='/app/src/resources/aux_respiratory_cycles.npy', 
                                                                seed=args.seed, 
                                                                train_test_split=args.test_train_split)

# Apply traditional augmentation techniques to the selected cycles in parallel.
apply_traditional_augmentation_lectura_datos_parallel(input_path='/app/data/raw', 
                                                      output_path='/app/data/processed/Train', 
                                                      sample_rate=args.sample_rate, 
                                                      length=args.duration, 
                                                      cycles=cycles_selected, 
                                                      rng=rng, 
                                                      seed=args.seed)