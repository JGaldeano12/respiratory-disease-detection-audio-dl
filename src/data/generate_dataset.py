import argparse
import shutil
from src.data.load_audio import lectura_datos_parallel, process_file
from src.data.preprocess import butter_bandpass_filter, check_length_and_padding
from src.features.extract_features import extract_features, save_features
from src.data.divide import check_and_create_directories, split_patients_by_train_test, move_spectrograms_by_train_test, create_directories
from src.data.segment import divide_audio
from scipy.signal import butter, lfilter
import os, gc, multiprocessing, librosa, numpy as np, cv2

# Parsear el argumento de la semilla
parser = argparse.ArgumentParser(description='Generate dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=202506, help='Random seed for train-test split')
args = parser.parse_args()

# First, create the necessary directories for the processed data
create_directories(input_path='/app/data/processed')

# Then, process the raw audio files and create spectrograms
lectura_datos_parallel(input_path='/app/data/raw', output_path='/app/data/processed', length=7, cycles='/app/src/data/ciclos_respiratorios.npy')

# Afterwards, create the Train-Test directories and split the spectrograms accordingly
cycles_train, cycles_test = split_patients_by_train_test(aux_file='/app/src/data/ciclos_respiratorios.npy', seed=args.seed, train_test_split=80)
move_spectrograms_by_train_test(input_path='/app/data/processed', cycles_train=cycles_train, cycles_test=cycles_test)