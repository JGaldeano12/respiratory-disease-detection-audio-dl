from src.data.segment import divide_audio
import os, gc, multiprocessing, librosa, numpy as np, cv2

def process_file(file, input_path, output_path, sample_rate, length, cycles):
    """
    Load and process a single audio file.

    The function extracts the patient identifier from the filename, loads the
    audio at a sampling rate of 8000 Hz, and delegates its segmentation to
    `divide_audio`.

    Args:
        file (str): Path or filename of the audio file to process.
        input_path (str): Directory containing the input audio files.
        output_path (str): Directory where the processed segments are saved.
        sample_rate (int): Sampling rate of the audio signal.
        length (float): Duration of each generated audio segment.
        cycles (np.ndarray): Respiratory cycle information used during audio
            segmentation.

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
        divide_audio(duration = length, sample_rate = sample_rate, raw_audio = raw_audio, output_path = output_path, patient = id_patient, cycles = cycles)
        
    except Exception as e:
        print(f"Error while processing {input_path} - Patient {id_patient}: {e}")

def lectura_datos_parallel(input_path, output_path, sample_rate, length, cycles):
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

    Returns:
        str: Confirmation message after all audio files have been processed.
    """
    # List of .wav files in the directory
    archivos_wav = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".wav")]

    # Load respiratory cycles
    cycles = np.load(cycles)

    # Create tuples of parameters for each file
    argumentos = [(archivo, input_path, output_path, sample_rate,length, cycles) for archivo in archivos_wav]

    # Use multiprocessing to parallelize the processing of files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_file, argumentos)

    return "Processing completed for all files."