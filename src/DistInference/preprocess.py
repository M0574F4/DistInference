import os
import glob
import numpy as np
import json
from scipy.signal import stft

##############################################################################
# Directories containing the .dat files
##############################################################################
DATA_DIRS = [
    "/project_ghent/Mostafa/ActivityRecognition/1 December 2017 Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/2 March 2017 Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/3 June 2017 Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/4 July 2018 Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/5 February 2019 UoG Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/6 February 2019 NG Homes Dataset",
    "/project_ghent/Mostafa/ActivityRecognition/7 March 2019 West Cumbria Dataset",
]

##############################################################################
# Output directory for preprocessed .npz files
##############################################################################
OUTPUT_DIR = "/project_ghent/Mostafa/ActivityRecognition/preprocessed"

##############################################################################
# STFT parameters (adjust as needed)
##############################################################################
WINDOW_SIZE = 128
OVERLAP     = 64

##############################################################################
# Normalization config:
#   - 'none': no normalization
#   - 'db': decibel domain standardization (needs global_stats)
#   - 'linear': min/max scaling (needs global_stats)
#
# If you prefer to do normalization in getitem, set this to 'none' here.
##############################################################################
NORMALIZATION = 'none'

##############################################################################
# Global stats for 'db' or 'linear' normalization if you want it in preprocessing
# Example JSON could contain keys: {"db_mean": ..., "db_std": ..., "linear_min": ..., "linear_max": ...}
##############################################################################
GLOBAL_STATS_JSON = "global_stats.json"  # e.g. "/path/to/global_stats.json" if needed

##############################################################################
# Helper Functions
##############################################################################
def parse_filename_get_label(filename):
    """
    Map the first character (e.g. '1') to an integer label (0..5).
    For example:
       '1' -> 0  # walking
       '2' -> 1  # sitting down
       ...
    """
    base = os.path.basename(filename)
    label_char = base[0]
    label_map = {
        '1': 0,  # walking
        '2': 1,  # sitting down
        '3': 2,  # stand up
        '4': 3,  # pick up an object
        '5': 4,  # drink water
        '6': 5,  # fall
    }
    return label_map.get(label_char, -1)  # -1 if not found

def parse_complex_line(line):
    """ Convert 'xx+yyi' or 'xx+yyI' to a Python complex number. """
    line = line.replace('i', 'j').replace('I', 'j')
    try:
        return complex(line)
    except ValueError:
        return None

def stft_and_normalize(slow_time_signal, prf, window_size, overlap, normalization, global_stats):
    """
    Perform STFT, then apply optional normalization (db or linear).
    Returns a float32 spectrogram array.
    """
    # Compute STFT
    f_vals, t_vals, Zxx = stft(
        slow_time_signal,
        fs=prf,
        window='hann',
        nperseg=window_size,
        noverlap=overlap,
        nfft=window_size * 2,
        return_onesided=False
    )
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
    magnitude = np.abs(Zxx_shifted)

    if normalization == 'db':
        if global_stats is None:
            raise ValueError("global_stats must be provided for 'db' normalization in preprocessing.")
        # Convert to dB
        magnitude_db = 20.0 * np.log10(magnitude + 1e-10)
        # Standardize
        mean_db = global_stats['db_mean']
        std_db  = global_stats['db_std']
        magnitude_db_normalized = (magnitude_db - mean_db) / std_db
        # Clip to [-3, 3], map to [0, 1]
        magnitude_db_normalized = np.clip(magnitude_db_normalized, -3.0, 3.0)
        magnitude_normalized = (magnitude_db_normalized + 3.0) / 6.0
        spectrogram = magnitude_normalized.astype(np.float32)
    elif normalization == 'linear':
        if global_stats is None:
            raise ValueError("global_stats must be provided for 'linear' normalization in preprocessing.")
        # Min/max scaling
        min_val = global_stats['linear_min']
        max_val = global_stats['linear_max']
        magnitude_clamped = np.clip(magnitude, min_val, max_val)
        magnitude_normalized = (magnitude_clamped - min_val) / (max_val - min_val + 1e-10)
        spectrogram = magnitude_normalized.astype(np.float32)
    else:
        # 'none'
        spectrogram = magnitude.astype(np.float32)
    return spectrogram

def preprocess_one_file(dat_path, output_dir, window_size, overlap, normalization, global_stats):
    """
    Read a .dat file, parse metadata, compute slow_time_signal, do STFT, optional normalization,
    then save the resulting spectrogram + label to an .npz file in output_dir with the same base name.
    """
    label = parse_filename_get_label(dat_path)

    with open(dat_path, 'r') as f:
        content = f.read().strip().split('\n')

    if len(content) < 5:
        print(f"[WARNING] Skipping {dat_path}, not enough lines.")
        return None

    # Parse metadata
    try:
        carrier_freq_ghz = float(content[0])
        chirp_duration_ms = float(content[1])
        samples_per_beat_note = int(float(content[2]))
        bandwidth_mhz = float(content[3])
    except ValueError as e:
        print(f"[WARNING] Skipping {dat_path}, can't parse metadata: {e}")
        return None

    data_lines = content[4:]
    complex_samples = [parse_complex_line(line) for line in data_lines if parse_complex_line(line) is not None]
    complex_samples = np.array(complex_samples, dtype=np.complex64)
    total_samples = len(complex_samples)

    if total_samples < samples_per_beat_note or samples_per_beat_note <= 0:
        print(f"[WARNING] Skipping {dat_path}, invalid samples.")
        return None

    # Reshape
    total_beat_notes = total_samples // samples_per_beat_note
    complex_samples = complex_samples[: total_beat_notes * samples_per_beat_note]
    complex_samples = complex_samples.reshape((total_beat_notes, samples_per_beat_note))

    # Example: compute slow_time_signal
    slow_time_signal = np.mean(complex_samples, axis=1)
    slow_time_signal -= np.mean(slow_time_signal)

    # Compute PRF
    prf = 1.0 / (chirp_duration_ms * 1e-3)

    # STFT and optional normalization
    spectrogram = stft_and_normalize(
        slow_time_signal,
        prf,
        window_size,
        overlap,
        normalization,
        global_stats
    )

    # Save .npz
    base = os.path.basename(dat_path)          # e.g. "1_something.dat"
    base_no_ext, _ = os.path.splitext(base)    # e.g. "1_something"
    out_path = os.path.join(output_dir, base_no_ext + ".npz")
    np.savez_compressed(out_path, spectrogram=spectrogram, label=label)
    return out_path

def main():
    # Optionally load global_stats if needed for 'db' or 'linear' normalization
    gs = None
    if GLOBAL_STATS_JSON and os.path.exists(GLOBAL_STATS_JSON):
        with open(GLOBAL_STATS_JSON, 'r') as f:
            gs = json.load(f)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Gather all .dat files from the 7 directories
    all_dat_files = []
    for data_dir in DATA_DIRS:
        pattern = os.path.join(data_dir, "**", "*.dat")
        matched = glob.glob(pattern, recursive=True)
        all_dat_files.extend(matched)

    print(f"Found {len(all_dat_files)} .dat files in the specified directories.")

    # Preprocess each
    for dat_file in all_dat_files:
        out_file = preprocess_one_file(
            dat_file,
            OUTPUT_DIR,
            WINDOW_SIZE,
            OVERLAP,
            NORMALIZATION,
            gs
        )
        if out_file:
            print(f"Preprocessed: {dat_file} -> {out_file}")

if __name__ == "__main__":
    main()
