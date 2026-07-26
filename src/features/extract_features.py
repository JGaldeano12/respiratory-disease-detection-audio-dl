import librosa, numpy as np, os, gc, multiprocessing, cv2

def standardize_audio(audio):
    """
    Function to standardize the audio signal by removing the mean and scaling to unit variance.
    This helps in normalizing the audio data and can improve the performance of machine learning models.
    """
    mean = np.mean(audio)
    std = np.std(audio) + 1e-6  # Add a small value to avoid division by zero
    standardized_audio = (audio - mean) / std
    return standardized_audio

def extract_features(audio, output_path, label, patient, index_cycle, type):
    """
    Function to extract features (spectrograms) from the audio segments and save them as .png images in the specified directory.

    audio: the audio data.
    output_path: destination path for the features.
    label: label of the audio segment (e.g., 'Healthy', 'Unhealthy').
    patient: patient ID.
    index_cycle: index of the respiratory cycle.
    type: type of augmentation (e.g., 'Original', 'Augmented').
    """
    # Generate the Mel Spectrogram for the audio segment:
    audio = standardize_audio(audio)
    spectrogram = librosa.feature.melspectrogram(y = audio, sr=8000, n_fft=2048, hop_length=256, n_mels=128, fmin=50, fmax=2500)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # # Now, generate MGCC features for the audio segment:
    # mfcc = librosa.feature.mfcc(y=audio, sr=8000, n_mfcc=20, n_fft=2048, hop_length=512)

    # # Delta features (first derivative):
    # delta_spectrogram = librosa.feature.delta(mfcc, order=1)

    # # Delta-delta features (second derivative):
    # delta_spectrogram_2 = librosa.feature.delta(mfcc, order=2)
    
    # # Resize the MFCC and delta features to match the dimensions of the spectrogram
    # mfcc = cv2.resize(mfcc, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_LINEAR)
    # delta_spectrogram = cv2.resize(delta_spectrogram, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_LINEAR)
    # delta_spectrogram_2 = cv2.resize(delta_spectrogram_2, (spectrogram.shape[1], spectrogram.shape[0]), interpolation=cv2.INTER_LINEAR)

    # # Standardize the spectrogram to have zero mean and unit variance
    # spectrogram = standardize_audio(spectrogram)
    # mfcc = standardize_audio(mfcc)
    # delta_spectrogram = standardize_audio(delta_spectrogram)
    # delta_spectrogram_2 = standardize_audio(delta_spectrogram_2)

    # # Concatenate the spectrogram, MFCC, and delta features along the channel dimension
    # combined_features = np.stack([spectrogram, mfcc, delta_spectrogram, delta_spectrogram_2], axis=-1)
    combined_features = np.expand_dims(spectrogram, axis=-1)

    # # Save the features for the original audio segment
    # save_features(spectrogram, index_cycle, output_path, label, patient, type)
    save_features(combined_features, index_cycle, output_path, label, patient, type)

def save_features(data, index_cycle, output_path, label, patient, type):
    """
    Function to generate and save the features (spectrograms) for each audio segment. The features are saved as .npy arrays in the specified directory.
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