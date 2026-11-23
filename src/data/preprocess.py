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

def check_length_and_padding(audio, target_length, mode="wrap", sample_rate=8000):
    """
    Adjust an audio signal to an exact target length.

    Signals longer than `target_length` are truncated. Shorter signals are
    extended by repeating the signal until the target length is reached:

    - "wrap": the copies are joined directly, so every junction has an abrupt
      jump in the waveform.
    - "fade": a linear fade-in and fade-out of 62.5 ms (1/16 s) is applied to the
      signal before repeating it, so the copies are joined smoothly through zero
      (as in Moummad & Farrugia, 2023, and Bae et al., 2023).

    Args:
        audio (np.ndarray): Input audio signal.
        target_length (int): Desired number of samples.
        mode (str, optional): "wrap" or "fade". Defaults to "wrap".
        sample_rate (int, optional): Sampling rate in Hz, used for the fade
            length. Defaults to 8000.

    Returns:
        np.ndarray: Audio signal with exactly `target_length` samples.
    """
    current_length = len(audio)

    if current_length > target_length:
        audio = audio[:target_length]
    elif current_length < target_length:
        if mode == "fade":
            # Fade of 1/16 s, limited to half of the signal for very short cycles.
            fade_length = min(sample_rate // 16, current_length // 2)
            if fade_length > 0:
                ramp = np.linspace(0.0, 1.0, fade_length, endpoint=False, dtype=audio.dtype)
                audio = audio.copy()
                audio[:fade_length] *= ramp
                audio[-fade_length:] *= ramp[::-1]
        elif mode != "wrap":
            raise ValueError(f"Unknown padding mode: {mode} (use wrap or fade)")
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode="wrap")

    return audio