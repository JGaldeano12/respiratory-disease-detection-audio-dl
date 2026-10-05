import argparse

from src.data.divide import create_directories, move_spectrograms_by_train_test, split_patients_by_train_test
from src.data.load_audio import process_files_parallel

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Generate dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=202506, help='Random seed for train-test split')
parser.add_argument('--sample_rate', type=int, default=8000, help='Sampling rate for audio processing')
parser.add_argument('--duration', type=int, default=8, help='Duration of each audio segment')
parser.add_argument('--test_train_split', type=int, default=80, help='Percentage of data for training')
args = parser.parse_args()

# First, create the necessary directories for the processed data
create_directories(input_path='/app/data/processed')

# Then, process the raw audio files and create spectrograms
process_files_parallel(input_path='/app/data/raw', output_path='/app/data/processed', sample_rate=args.sample_rate, length=args.duration, cycles='/app/src/resources/aux_respiratory_cycles.npy')

# Afterwards, create the Train-Test directories and split the spectrograms accordingly
cycles_train, cycles_test = split_patients_by_train_test(aux_file='/app/src/resources/aux_respiratory_cycles.npy', seed=args.seed, train_test_split=args.test_train_split)
move_spectrograms_by_train_test(input_path='/app/data/processed', cycles_train=cycles_train, cycles_test=cycles_test)