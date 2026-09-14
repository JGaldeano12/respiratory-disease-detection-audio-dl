from scipy.signal import butter, lfilter
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency of the signal in Hz.
        order (int, optional): Order of the Butterworth filter. Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator (`b`) and denominator (`a`)
        coefficients of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")

    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to a signal.

    Args:
        data (np.ndarray): Input signal to filter.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency of the signal in Hz.
        order (int, optional): Order of the Butterworth filter. Defaults to 5.

    Returns:
        np.ndarray: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def check_length_and_padding(audio, target_length):
    """
    Adjust an audio signal to an exact target length.

    Signals longer than `target_length` are truncated. Shorter signals are
    extended using wrap padding, repeating samples from the beginning of the
    signal until the target length is reached.

    Args:
        audio (np.ndarray): Input audio signal.
        target_length (int): Desired number of samples.

    Returns:
        np.ndarray: Audio signal with exactly `target_length` samples.
    """
    current_length = len(audio)

    if current_length > target_length:
        audio = audio[:target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode="wrap")

    return audio