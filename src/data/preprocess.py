from scipy.signal import butter, lfilter

import librosa, numpy as np, os, gc, multiprocessing, cv2, math

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

# def check_length_and_padding(audio, target_length, sample_rate=8000, crossfade_ms=10):
#     """
#     Ensure audio has EXACT target_length using Repeat/Tile Padding.

#     Repeats the audio cyclically until reaching target_length, applying
#     a short crossfade at each repetition boundary to avoid discontinuities.

#     Args:
#         audio        : np.ndarray — input audio signal
#         target_length: int        — desired number of samples
#         sample_rate  : int        — used to compute crossfade length in samples
#         crossfade_ms : int        — crossfade duration in milliseconds (default: 10ms)
#     """
#     current_length = len(audio)

#     # TRUNCATE
#     if current_length > target_length:
#         audio = audio[:target_length]

#     # REPEAT/TILE PADDING
#     elif current_length < target_length:
#         crossfade_samples = int(sample_rate * crossfade_ms / 1000)
#         crossfade_samples = min(crossfade_samples, current_length // 2)

#         # Número de repeticiones necesarias
#         repeats = math.ceil(target_length / current_length)
#         tiled = np.tile(audio, repeats)

#         # Aplicar crossfade en cada punto de costura
#         for i in range(1, repeats):
#             join = i * current_length  # índice del punto de unión

#             if join >= len(tiled):
#                 break

#             # Ventanas de fade-out y fade-in
#             fade_out = np.linspace(1.0, 0.0, crossfade_samples)
#             fade_in  = np.linspace(0.0, 1.0, crossfade_samples)

#             # Zona antes de la costura (final del bloque anterior)
#             start_out = join - crossfade_samples
#             end_out   = join

#             # Zona después de la costura (inicio del bloque siguiente)
#             start_in  = join
#             end_in    = join + crossfade_samples

#             if end_in <= len(tiled):
#                 tiled[start_out:end_out] *= fade_out
#                 tiled[start_in:end_in]   *= fade_in
#                 # Mezcla: suma ambas zonas solapadas
#                 tiled[start_out:end_out] += tiled[start_in:end_in] * fade_out[::-1]

#         audio = tiled[:target_length]

#     return audio

def check_length_and_padding(audio, target_length, sample_rate=4096):
    """
    Ensure audio has EXACT target_length using Wrap Padding.

    Treats the audio as a circular buffer, filling the missing samples
    by cycling back to the beginning of the signal. This avoids artificial
    silence (zero padding) and phase inversion (reflect padding), while
    being simpler and more efficient than manual tiling.

    Args:
        audio         : np.ndarray — input audio signal
        target_length : int        — desired number of samples
        sample_rate   : int        — retained for API consistency
    """
    current_length = len(audio)

    # TRUNCATE
    if current_length > target_length:
        audio = audio[:target_length]

    # WRAP PADDING
    elif current_length < target_length:
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='wrap')

    return audio