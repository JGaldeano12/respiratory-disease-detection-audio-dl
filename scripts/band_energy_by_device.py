"""Mean spectrogram value per mel row and device, to see where each device's energy ends."""
import collections
import glob

import librosa
import numpy as np

FMIN, FMAX, N_MELS = 50, 2500, 128
MAX_FILES = 300

files = collections.defaultdict(list)
for path in sorted(glob.glob('/app/data/processed/*/*/*.npy')):
    name = path.rsplit('/', 1)[-1]
    if 'Original' not in name:
        continue
    device = name.split('_')[4]
    if len(files[device]) < MAX_FILES:
        files[device].append(path)

devices = sorted(files)
means = {d: np.mean([np.load(p).reshape(N_MELS, -1).mean(axis=1) for p in files[d]], axis=0) for d in devices}
centers = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=FMIN, fmax=FMAX)[1:-1]

print('row   Hz    ' + ''.join(f'{d:>10}' for d in devices))
for row in list(range(0, 88, 8)) + list(range(88, N_MELS, 2)):
    print(f'{row:3d} {centers[row]:6.0f}  ' + ''.join(f'{means[d][row]:10.2f}' for d in devices))
print('files: ' + ', '.join(f'{d}={len(files[d])}' for d in devices))
