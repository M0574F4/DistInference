# %matplotlib inline
from PIL import Image
import torch
from DistInference.general_utils import analyze_input, grid_image_show, histogram_plotter, print_module_summary, set_module_grad, save_dict_pickle
from DistInference.tcm import TCM
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def tcm_loader(checkpoint, net):
    checkpoint = torch.load(checkpoint, map_location=device)
    
    dictory = {}
    for k, v in checkpoint["state_dict"].items():
        dictory[k.replace("module.", "")] = v
    net.load_state_dict(dictory)
    return net


N = 64
lambda_value = "0_05" # 0_0025 0_05


device = 'cuda'

# TCM
checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/mse_lambda_{lambda_value}_N{N}.pth"
# checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/pths/20240926_063733.pth.tar"
# checkpoint = f"/project_ghent/Mostafa/image/ImageTransmission/src/gen_comm/lic_tcm/LIC_TCM/checkpointscheckpoint_latest.pth.tar"
tcm = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=N, M=320)
# tcm.num_slices=192
tcm = tcm_loader(checkpoint, tcm)
tcm.cuda()
tcm.update()

img = torch.rand(1, 3, 256, 40).cuda()

z = tcm.g_a(img)
img_hat = tcm.g_s(z).clamp_(0, 1)

# psnr = compute_metrics(img, img_hat)['psnr']
# print(psnr)
print(z.shape)
# grid_image_show([img, img_hat], scale=.5)

file_path = '/project_ghent/Mostafa/ActivityRecognition/1 December 2017 Dataset/3P50A03R02.dat'

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Split the content into lines
lines = content.strip().split('\n')

# Extract metadata (first four lines)
metadata = {
    'carrier_frequency_GHz': float(lines[0]),
    'chirp_duration_ms': float(lines[1]),
    'samples_per_beat_note': int(float(lines[2])),
    'bandwidth_MHz': float(lines[3])
}

# Extract data (remaining lines)
data_lines = lines[4:]

# Function to convert a line to a complex number
def parse_complex(line):
    line = line.replace('i', 'j').replace('I', 'j')  # Replace 'i' with 'j' for Python
    try:
        return complex(line)
    except ValueError:
        return None  # or handle differently if needed

# Parse all data lines into complex numbers, filtering out invalid entries
complex_samples = [parse_complex(line) for line in data_lines]
complex_samples = [sample for sample in complex_samples if sample is not None]


# Convert to a NumPy array
complex_samples = np.array(complex_samples)

# Optionally, reshape into beat-note signals if samples_per_beat_note > 0
samples_per_beat = metadata['samples_per_beat_note']

total_samples = len(complex_samples)
total_beat_notes = total_samples // samples_per_beat
if total_beat_notes == 0:
    raise ValueError("Number of samples per beat-note is larger than total samples.")
# Trim excess samples
complex_samples = complex_samples[:total_beat_notes * samples_per_beat]
# Reshape
beat_notes = complex_samples.reshape((total_beat_notes, samples_per_beat))


####################
from scipy.signal import stft

# --- Step 1: Extract Metadata ---
samples_per_chirp = metadata['samples_per_beat_note']//1
chirp_duration_ms = metadata['chirp_duration_ms']  # in milliseconds
prf = 1 / (chirp_duration_ms * 1e-3)  # Pulse Repetition Frequency in Hz
bandwidth_hz = metadata['bandwidth_MHz'] * 1e6  # Convert MHz to Hz
carrier_freq_hz = metadata['carrier_frequency_GHz'] * 1e9  # Convert GHz to Hz

num_chirps = total_samples // samples_per_chirp

# Reshape to 2D array: (num_chirps x samples_per_chirp)
complex_data = complex_samples.reshape((num_chirps, samples_per_chirp))

slow_time_signal = np.mean(complex_data, axis=1)

# Remove DC component to avoid spectral leakage
slow_time_signal -= np.mean(slow_time_signal)



window_size, overlap = 128, 64


# --- Step 6: Perform STFT on Slow-Time Signal ---
# The sampling frequency for slow-time is PRF
f, t_stft, Zxx = stft(
    slow_time_signal,
    fs=prf,
    window='hann',
    nperseg=window_size,
    noverlap=overlap,
    nfft=window_size*2,  # Increase nfft for better frequency resolution
    return_onesided=False  # To obtain both negative and positive frequencies
)



# --- Step 7: Shift STFT Output to Center Zero Frequency ---
Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
f_shifted = np.fft.fftshift(f)

real = np.real(Zxx_shifted)
imag = np.imag(Zxx_shifted)
magnitude = np.abs(Zxx_shifted)

max_val = max(np.abs(real).max(), np.abs(imag).max())

real_normalized = (real / max_val + 1) / 2  # Maps from [-1, 1] to [0, 1]
imag_normalized = (imag / max_val + 1) / 2  # Maps from [-1, 1] to [0, 1]

magnitude_normalized = magnitude / np.max(magnitude)
magnitude_normalized = np.clip(magnitude_normalized, 0, 1)  # Ensure within [0, 1]


stft_3d = np.stack([real_normalized, imag_normalized, magnitude_normalized], axis=-1)


# Expand dimensions if necessary (e.g., batch size of 1)
stft_3d = np.expand_dims(stft_3d, axis=0)  # Shape: (1, height, width, 3)

# Convert to torch tensor and move to GPU
stft_tensor = torch.tensor(stft_3d).permute(0, 3, 1, 2).cuda().float()  # Shape: (1, 3, height, width)



print(f"Real Channel: min={stft_tensor[0,0].min().item()}, max={stft_tensor[0,0].max().item()}")
print(f"Imaginary Channel: min={stft_tensor[0,1].min().item()}, max={stft_tensor[0,1].max().item()}")
print(f"Magnitude Channel: min={stft_tensor[0,2].min().item()}, max={stft_tensor[0,2].max().item()}")



z_magnitude_db = tcm.g_a(stft_tensor)
img_hat_magnitude_db = tcm.g_s(z_magnitude_db)




# # --- Step 8: Compute Magnitude in dB ---
# magnitude = np.abs(Zxx_shifted)
# magnitude_db = 20 * np.log10(magnitude + 1e-6)  # Convert to dB
# magnitude_db -= np.max(magnitude_db)  # Normalize for better contrast

# ######################
# import matplotlib.pyplot as plt


# plt.figure(figsize=(8, 4))
# # --- Subplot 1: Magnitude ---
# # plt.subplot(3, 1, 1)
# plt.pcolormesh(t_stft, f_shifted, magnitude_db, shading='gouraud', cmap='jet')
# plt.title('Micro-Doppler Spectrogram (Magnitude)')
# plt.ylabel('Doppler Frequency (Hz)')
# plt.xlabel('Time (s)')
# cbar1 = plt.colorbar(label='Normalized Log Magnitude (dB)')
# plt.ylim(-prf/2, prf/2)  # Limit frequency axis to -PRF/2 to PRF/2
# plt.show()


# ######
# print(magnitude_db.shape)
# img_magnitude_db = magnitude_db/-100
# img_magnitude_db = np.clip(img_magnitude_db, 0, 1)
# img_magnitude_db = np.expand_dims(np.expand_dims(img_magnitude_db, axis=0), axis=0)
# img_magnitude_db = torch.tensor(img_magnitude_db).cuda().float()
# print(img_magnitude_db.shape, img.shape)

# print(torch.min(img), torch.max(img))
# print(torch.min(img_magnitude_db), torch.max(img_magnitude_db))


# z_magnitude_db = tcm.g_a(img_magnitude_db)
# img_hat_magnitude_db = tcm.g_s(z_magnitude_db)

# plt.figure(figsize=(8, 4))
# # --- Subplot 1: Magnitude ---
# # plt.subplot(3, 1, 1)
# plt.pcolormesh(t_stft, f_shifted, img_hat_magnitude_db, shading='gouraud', cmap='jet')
# plt.title('Micro-Doppler Spectrogram (img_hat_magnitude_db)')
# plt.ylabel('Doppler Frequency (Hz)')
# plt.xlabel('Time (s)')
# cbar1 = plt.colorbar(label='Normalized Log Magnitude (dB)')
# plt.ylim(-prf/2, prf/2)  # Limit frequency axis to -PRF/2 to PRF/2
# plt.show()
