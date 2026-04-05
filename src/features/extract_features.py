import librosa, numpy as np, os, gc, multiprocessing, cv2

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
    spectrogram = librosa.feature.melspectrogram(y = audio, sr=4096, n_fft=256, hop_length=128, n_mels=64)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
    spectrogram *= 255

    # Reshape the spectrogram to have a single channel (grayscale)
    spectrogram = spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)

    # Save the features for the original audio segment
    save_features(spectrogram, index_cycle, output_path, label, patient, type)

def save_features(data, index_cycle, output_path, label, patient, type):
    """
    Function to generate and save the features (spectrograms) for each audio segment. The features are saved as .png images in the specified directory.
    """
    # Create the spectrogram filename based on the patient ID, index, and augmentation type
    path = f"{label}/{patient}_{index_cycle}_{type}.png"

    # Normalize the spectrogram data to the range [0, 255] and save it as a .png image
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save the spectrogram image to the specified path
    cv2.imwrite(os.path.join(output_path, path), img)

    # Free memory after saving the image
    gc.collect()