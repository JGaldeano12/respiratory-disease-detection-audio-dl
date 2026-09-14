import librosa, numpy as np, os, gc

def standardize_audio(audio):
    """
    Standardize an audio signal to have approximately zero mean and unit variance.

    The mean of the input signal is subtracted and the result is divided by its
    standard deviation. A small constant is added to the standard deviation to
    prevent division by zero.

    Args:
        audio (np.ndarray): Input audio signal to standardize.

    Returns:
        np.ndarray: Standardized audio signal.
    """
    mean = np.mean(audio)
    std = np.std(audio) + 1e-6  # Add a small value to avoid division by zero
    standardized_audio = (audio - mean) / std
    return standardized_audio

def extract_features(audio, sample_rate, output_path, label, patient, index_cycle, type):
    """
    Extract a Mel spectrogram from an audio segment and save it as a NumPy array.

    The input audio is first standardized. A Mel spectrogram is then computed
    and converted from power values to the decibel scale. The resulting
    spectrogram is expanded with a channel dimension and saved as a `.npy`
    file using the provided metadata.

    Args:
        audio (np.ndarray): Audio segment from which the features are extracted.
        sample_rate (int): Sampling rate of the audio signal.
        output_path (str): Base directory where the extracted features are saved.
        label (str): Class label associated with the audio segment.
        patient (str): Identifier of the patient associated with the recording.
        index_cycle (str): Identifier of the respiratory cycle.
        type (str): Type of audio sample or augmentation applied.

    Returns:
        None
    """
    # Generate the Mel Spectrogram for the audio segment:
    audio = standardize_audio(audio)
    spectrogram = librosa.feature.melspectrogram(y = audio, sr=sample_rate, n_fft=2048, hop_length=256, n_mels=128, fmin=50, fmax=2500)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Concatenate the spectrogram, MFCC, and delta features along the channel dimension
    combined_features = np.expand_dims(spectrogram, axis=-1)

    # Save the features for the original audio segment
    save_features(combined_features, index_cycle, output_path, label, patient, type)

def save_features(data, index_cycle, output_path, label, patient, type):
    """
    Save extracted features as a NumPy array using a metadata-based filename.

    The input feature array is converted to `float32` before being saved as a
    `.npy` file. The output filename is constructed from the class label,
    patient identifier, respiratory cycle identifier, and sample or
    augmentation type.

    Args:
        data (np.ndarray): Feature array to save.
        index_cycle (str): Identifier of the respiratory cycle.
        output_path (str): Base directory where the feature file is saved.
        label (str): Class label used to determine the output subdirectory.
        patient (str): Identifier of the patient associated with the sample.
        type (str): Type of audio sample or augmentation used in the filename.

    Returns:
        None
    """
    # Create the spectrogram filename based on the patient ID, index, and augmentation type
    path = f"{label}/{patient}_{index_cycle}_{type}.npy"
    full_path = os.path.join(output_path, path)

    # Cast the data to float32 to save memory and ensure compatibility with most machine learning frameworks
    data = data.astype(np.float32)

    # Save the spectrogram as a .npy file
    np.save(full_path, data)

    # Print a message with the numpy array dimensions:
    print(f"Saved spectrogram for patient {patient} - Cycle {index_cycle} - Type: {type} - Shape: {data.shape}")

    # Free memory after saving the image
    gc.collect()