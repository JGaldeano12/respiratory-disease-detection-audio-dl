# # Import funcionts from file extract_features.py and preprocess.py
# from src.features.extract_features import generar_guardar_features
# from src.data.preprocess import butter_bandpass_filter, check_length_and_padding 
# from src.data.augment import gen_augmented
from src.data.segment import divide_audio

# Import other necessary libraries
import os, gc, multiprocessing, librosa, numpy as np, cv2

def process_file(file, input_path, output_path, length, cycles):
    """
    Function to process a single audio file. Includes: loading, segmenting, generating spectrograms and saving the results.
    """
    try:
        # Get unique record ID from the filename
        base_filename = os.path.basename(file).replace(".wav", "")
    
        # Obtain the information of the recording from the filename
        info_elements = base_filename.split("_")
        id_patient = info_elements[0] + "_" + info_elements[1] + "_" + info_elements[2] + "_" + info_elements[3] + "_" + info_elements[4]

        # Load the audio file using librosa
        raw_audio, sample_rate = librosa.load(os.path.join(input_path, file), sr=8000)

        # Proceed to audio segmentation
        divide_audio(duration = length, sample_rate = sample_rate, raw_audio = raw_audio, output_path = output_path, patient = id_patient, cycles = cycles)
        
    except Exception as e:
        print(f"Error while processing {input_path} - Patient {id_patient}: {e}")

def lectura_datos_parallel(input_path, output_path, length, cycles):
    """
    Function to read the data and process .wav files in parallel.
    """  
    # List of .wav files in the directory
    archivos_wav = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".wav")]

    # Load respiratory cycles
    cycles = np.load(cycles)

    # Create tuples of parameters for each file
    argumentos = [(archivo, input_path, output_path, length, cycles) for archivo in archivos_wav]

    # Use multiprocessing to parallelize the processing of files
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_file, argumentos)

    return "Processing completed for all files."