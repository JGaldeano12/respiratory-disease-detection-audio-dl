from scipy.signal import butter, lfilter

import librosa, numpy as np, os, gc, multiprocessing, cv2

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Function to design a Butterworth bandpass filter.
    """
    # Design the Butterworth bandpass filter coefficients based on the specified lowcut, highcut frequencies, sampling rate (fs), and filter order.
    nyq = 0.5 * fs

    # Normalize the lowcut and highcut frequencies by the Nyquist frequency and compute the filter coefficients using the Butterworth filter design.
    low = lowcut / nyq
    high = highcut / nyq

    # Compute the Butterworth bandpass filter coefficients using the butter function from the scipy.signal library.
    b, a = butter(order, [low, high], btype='band')

    # Return the filter coefficients (b, a) for the designed Butterworth bandpass filter.
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Function to apply a Butterworth bandpass filter to the input data.
    """
    # Apply the Butterworth bandpass filter to the input data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    # Return the filtered audio data
    return y

def standardize_audio(audio):
    """
    Function to standardize the audio data by removing the mean and scaling to unit variance.
    """
    # Standardize the audio data by removing the mean and scaling to unit variance
    standardized_audio = (audio - -1.4100063212550931e-07) / 0.03260556890932612

    # Return the standardized audio data
    return standardized_audio

def check_length_and_padding(audio, start_sample, target_length):
    """
    Function to ensure that the audio data has a specific target length by applying padding or truncation as needed.
    """
    # Check if the audio data is shorter than the target length and apply padding if necessary
    if len(audio) < target_length:
        # Randomly choose between constant padding and reflect padding to ensure the audio data reaches the target length
        if np.random.rand() > 0.5:
            padding = target_length - len(audio)
            segmented_audio = np.pad(audio, (0, padding), mode = 'constant')
        else:
            padding = target_length - len(audio)
            segmented_audio = np.pad(audio, (0, padding), mode = 'reflect')

    # Check if the audio data is longer than the target length and apply truncation if necessary
    elif len(audio) > target_length:
        segmented_audio = audio[:start_sample + target_length]

    # If the audio data is already of the target length, return it as is
    else:
        segmented_audio = audio

    # Return the audio data with the ensured target length
    return segmented_audio