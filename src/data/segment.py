# Import funcionts from file extract_features.py and preprocess.py
from src.features.extract_features import extract_features, save_features
from src.data.preprocess import butter_bandpass_filter, check_length_and_padding

# Import other necessary libraries
import os, gc, multiprocessing, librosa, numpy as np, cv2

def divide_audio(duration, sample_rate, raw_audio, output_path, patient, cycles):
    """
    Function to divide the audio into segments based on the respiratory cycles and generate spectrograms for each segment.

    duration: target duration of the audio.
    sample_rate: sampling rate.
    raw_audio: raw audio data.
    output_path: destination path.
    patient: patient ID.
    cycles: respiratory cycles information.
    cycles_train: training respiratory cycles information.
    """
    # Define the target duration in number of samples
    target_length = int(duration * sample_rate)

    # Obtain the respiratory cycles for the specific patient
    cycles_filtered = cycles[np.isin(cycles[:, 1], patient)]

    # Auxiliar variable to keep track of the cycle number
    aux_index_cycle = 0

    # Now, for each respiratory cycle, I will extract the corresponding audio segment, apply the Butterworth filter, standardize it, and generate the spectrogram. 
    for index_cycle in cycles_filtered:
        # Define the start and end of the segment in terms of samples
        start = int(float(index_cycle[2]) * sample_rate)
        end = int(float(index_cycle[3]) * sample_rate)

        # Divide the audio segment corresponding to the respiratory cycle:
        segm = raw_audio[start:end]

        # Apply the Butterworth bandpass filter and standardize the audio segment
        segmented_audio = butter_bandpass_filter(segm, 50, 2500, sample_rate, order=5)

        # Check wether the audio segment is shorter than the target duration. If so, we will apply padding to reach the desired length.
        segmented_audio = check_length_and_padding(segmented_audio, target_length)

        # Generate the Mel Spectrogram for the audio segment:
        extract_features(segmented_audio, output_path, label = index_cycle[6], patient = patient, index_cycle = aux_index_cycle, type = "Original")

        # Increment the cycle index for the next iteration
        aux_index_cycle += 1

    # Free memory after processing the audio file
    gc.collect()