import multiprocessing
import os

import librosa
import numpy as np

from src.data.segment import divide_audio

def process_file(file, input_path, output_path, sample_rate, length, cycles, highcut=2000, fmax=2500):
    """
    Load and process a single audio file.

    The function extracts the patient identifier from the filename, loads the
    audio at the given sampling rate, and delegates its segmentation to
    `divide_audio`.

    Args:
        file (str): Path or filename of the audio file to process.
        input_path (str): Directory containing the input audio files.
        output_path (str): Directory where the processed segments are saved.
        sample_rate (int): Sampling rate of the audio signal.
        length (float): Duration of each generated audio segment.
        cycles (np.ndarray): Respiratory cycle information used during audio
            segmentation.
        highcut (float, optional): Upper cutoff frequency of the bandpass
            filter in Hz. Defaults to 2000.
        fmax (float, optional): Highest frequency of the Mel filterbank in Hz.
            Defaults to 2500.

    Returns:
        None
    """
    try:
        # Get unique record ID from the filename
        base_filename = os.path.basename(file).replace(".wav", "")
    
        # Obtain the information of the recording from the filename
        info_elements = base_filename.split("_")
        id_patient = info_elements[0] + "_" + info_elements[1] + "_" + info_elements[2] + "_" + info_elements[3] + "_" + info_elements[4]

        # Load the audio file using librosa
        raw_audio, sample_rate = librosa.load(os.path.join(input_path, file), sr = sample_rate)

        # Proceed to audio segmentation
        divide_audio(duration = length, sample_rate = sample_rate, raw_audio = raw_audio, output_path = output_path, patient = id_patient, cycles = cycles, highcut = highcut, fmax = fmax)
        
    except Exception as e:
        print(f"Error while processing {input_path} - Patient {id_patient}: {e}")

def process_files_parallel(input_path, output_path, sample_rate, length, cycles, highcut=2000, fmax=2500):
    """
    Process all WAV files in a directory in parallel.

    The function identifies all `.wav` files in `input_path`, loads the
    respiratory cycle information from the specified NumPy file, and processes
    each audio file using a separate task in a multiprocessing pool.

    Args:
        input_path (str): Directory containing the input `.wav` files.
        output_path (str): Directory where the processed audio segments are
            saved.
        sample_rate (int): Sampling rate of the audio signal.
        length (float): Duration of each generated audio segment.
        cycles (str): Path to the NumPy file containing the respiratory cycle
            information.
        highcut (float, optional): Upper cutoff frequency of the bandpass
            filter in Hz. Defaults to 2000.
        fmax (float, optional): Highest frequency of the Mel filterbank in Hz.
            Defaults to 2500.

    Returns:
        str: Confirmation message after all audio files have been processed.
    """
    # List of .wav files in the directory
    wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".wav")]

    # Load respiratory cycles
    cycles = np.load(cycles)

    # Create tuples of parameters for each file
    task_args = [(wav_file, input_path, output_path, sample_rate, length, cycles, highcut, fmax) for wav_file in wav_files]

    # Use multiprocessing to parallelize the processing of files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_file, task_args)

    return "Processing completed for all files."