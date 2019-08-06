import numpy as np
from scipy import io
import soundfile as sf
import time
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.linalg import solve_toeplitz
from scipy.linalg import toeplitz
from scipy.signal import stft
from data_mat_find_ir import get_impulse_responses

sig_duration = int(100e-3)
zero_duration = 2.5
DFT_LEN = 8192
num_channels = 8
num_sweeps_sources = 1

source_fname = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_100ms/audible_chirps/audible_chirp_100ms_96khz.wav'
recording_fname = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_100ms/audible_chirps/1_rec_audible_chirp_100ms_96khz.wav'

src, Fs_src = sf.read(source_fname)
rec, Fs_rec = sf.read(recording_fname)

# to_compute = int(96e3 * 500e-3)

rec_matrix, src_matrix = create_data_matrix(src, rec, Fs_rec)
to_compute = rec.shape[0]
h_matrix = np.empty((to_compute, num_channels, num_sweeps_sources))

a = get_impulse_responses(src, rec, Fs_src)
plt.figure(1)
plt.plot(a[:, :, 1])
plt.show()




A = np.fft.fft(a, axis=0)
#
# First, we find the R matrix via the use of sum(source_power * H * H')
#
offset = int(Fs_src*2.5)
R = np.zeros((a.shape[1], a.shape[1]), dtype=complex)
start = time.time()
for src_idx in range(8):
    A[:, :, src_idx] = np.transpose(A[:, :, src_idx], [1, 2, 0])
    A[:, :, src_idx] /= np.sqrt(np.mean(np.sum(np.mean(np.abs(A[:, :, src_idx])**2, axis=0), axis=1), axis=2))
    R += np.dot(A[:, :, src_idx].conj().T, A[:, :, src_idx])
print("R matrix time", time.time()-start)

# Second, we find the R matrix by find the mean over time of X(t, f) * X(t, f)'


# First, we find the spectogram of the recorded audio signal to create
# [Frequency bins x Time bins x number of microphones/channels]

# For now doing it from the first speaker to the first microphone

window_size = 100
overlap_size = 50

f, t, stft_x = stft(rec_matrix[:, 0, 0], fs=Fs_src, nfft=DFT_LEN, nperseg=window_size, noverlap=overlap_size)

stft_x = stft_x[:DFT_LEN, :]
stft_xfm = stft_x[:, :, np.newaxis]

for src_idx in np.arange(1, src.shape[1]):
    f, t, stft_x = stft(rec_matrix[:, 0, src_idx], fs=Fs_src, nfft=DFT_LEN, nperseg=window_size, noverlap=overlap_size)
    stft_x = stft_x[:DFT_LEN, :]
    stft_xfm = np.concatenate((stft_xfm, stft_x[:, :, np.newaxis]), axis=2)

R_f = np.zeros((stft_xfm.shape[2], stft_xfm.shape[2], stft_xfm.shape[0]), dtype=np.complex128)
for f in np.arange(stft_xfm.shape[0]):
    print(f)
    R_f[:, :, f] = np.dot(stft_xfm[f, :, :].conj().T, stft_xfm[f, :, :])

for f in np.arange(stft_xfm.shape[0]):
    R_f[:, :, f] = R_f[:, :, f]/stft_xfm.shape[1]
print("R matrix time", time.time()-start)

# mvdr_weights = np.dot(np.linalg.inv(R_f), A)/np.dot(np.dot(A, np.linalg.inv(R_f).conj().T), A)
#
# # R_inter = np.zeros((a.shape[0], a.shape[0]))
# # for mic in range(8):
# #     R_inter += (np.sqrt(np.mean(src[offset:(Fs_src*11 + offset), src_idx]**2)))\
# #                                   *np.matmul(A[:, mic, src_idx][:, np.newaxis], \
# #                                   A[:, mic, src_idx][:, np.newaxis].H)
# # offset += int(11*Fs_src + 2.5*Fs_src)
# # R[:, :, src_idx] = R_inter