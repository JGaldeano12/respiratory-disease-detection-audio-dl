import argparse

from src.data.divide import create_directories, get_split, move_spectrograms_by_split
from src.data.load_audio import process_files_parallel

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Generate dataset with a given random seed.')
parser.add_argument('--seed', type=int, default=202506, help='Random seed for the patient-wise split')
parser.add_argument('--sample_rate', type=int, default=8000, help='Sampling rate for audio processing')
parser.add_argument('--highcut', type=float, default=2000, help='Upper cutoff frequency (Hz) of the bandpass filter; must be below sample_rate / 2')
parser.add_argument('--fmax', type=float, default=2500, help='Highest frequency (Hz) of the Mel filterbank')
parser.add_argument('--duration', type=int, default=8, help='Duration of each audio segment')
parser.add_argument('--test_train_split', type=int, default=80, help='Percentage of patients used for training + validation')
parser.add_argument('--val_split', type=int, default=15, help='Percentage of the training + validation patients used for validation')
parser.add_argument('--split_mode', type=str, default='random', choices=['random', 'official', 'respirenet'], help='Seeded patient-wise split, official ICBHI 2017 partition, or RespireNet folds')
parser.add_argument('--official_split_file', type=str, default='/app/src/resources/ICBHI_challenge_train_test.txt', help='Official ICBHI 2017 partition file')
parser.add_argument('--folds_file', type=str, default='/app/src/resources/patient_list_foldwise.txt', help='RespireNet patient-wise folds file')
parser.add_argument('--test_fold', type=int, default=4, help='RespireNet fold used as the testing set')
args = parser.parse_args()

# First, create the necessary directories for the processed data
create_directories(input_path='/app/data/processed')

# Then, process the raw audio files and create spectrograms
process_files_parallel(input_path='/app/data/raw', output_path='/app/data/processed', sample_rate=args.sample_rate, length=args.duration, cycles='/app/src/resources/aux_respiratory_cycles.npy', highcut=args.highcut, fmax=args.fmax)

# Afterwards, create the Train-Val-Test directories and split the spectrograms accordingly
cycles_train, cycles_val, cycles_test = get_split(aux_file='/app/src/resources/aux_respiratory_cycles.npy', seed=args.seed, train_test_split=args.test_train_split, val_split=args.val_split, split_mode=args.split_mode, official_file=args.official_split_file, folds_file=args.folds_file, test_fold=args.test_fold)
move_spectrograms_by_split(input_path='/app/data/processed', cycles_train=cycles_train, cycles_val=cycles_val, cycles_test=cycles_test)