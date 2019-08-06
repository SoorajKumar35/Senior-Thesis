import numpy as np
from scipy import io
import soundfile as sf
import scipy.signal

def get_impulse_responses(sources, recordings, Fs):
    """
    This function takes in the source file and the correspoding recoding file and returns the impulse response matrix
    per channel/microphone and per source.
    :return: 
    """

    offset = int(Fs*2.5)
    a = np.zeros((1056000, 8, 8))
    for src in range(8):
        S = np.fft.fft(sources[offset:(11*Fs + offset), src])
        X = np.fft.fft(recordings[offset:(11*Fs + offset), :])
        A = X/S[:, np.newaxis]
        a[:, :, src] = np.fft.ifft(A)
        offset += int(11*Fs + 2.5*Fs)
    return a

source_fname = '/Users/soorajkumar/Desktop/ECE496/ul_chirp_w_space.wav'
recording_fname = '/Users/soorajkumar/Desktop/ECE496/1_rec_ultrasonic_chirp_w_spaces.wav'

src, Fs_src = sf.read(source_fname)
rec, Fs_rec = sf.read(recording_fname)

a = get_impulse_responses(src, rec, Fs_src)
A = np.fft.fft(a, axis=0)

# First, we find the R matrix via the use of sum(source_power * H * H')

# offset = int(Fs_src*2.5)
# R = np.zeros((a.shape[0], a.shape[0], a.shape[2]))
# for src_idx in range(8):
#     R_inter = np.zeros((a.shape[0], a.shape[0]))
#     for mic in range(8):
#         R_inter += (np.sqrt(np.mean(src[offset:(Fs_src*11 + offset), src_idx]**2)))*np.matmul(A[:, mic, src_idx][:, np.newaxis], \
#                                                                                               A[:, mic, src_idx][:, np.newaxis].H)
#     offset += int(11*Fs_src + 2.5*Fs_src)
#     R[:, :, src_idx] = R_inter

# Second, we find the R matrix by find the mean over time of X(t, f) * X(t, f)'

f, t, stft_x = scipy.signal.stft(src[:, 0], fs=Fs_src)
stft_xfm = stft_x[:, :, np.newaxis]
for src_idx in np.arange(1, src.shape[1]):
    f, t, stft_x = scipy.signal.stft(src[:, src_idx], fs=Fs_src)
    stft_xfm = np.concatenate((stft_xfm, stft_x[:, :, np.newaxis]), axis=2)
pass

R_f = np.zeros((stft_xfm.shape[2], stft_xfm.shape[2], stft_xfm.shape[1]), dtype=np.complex128)
for t in np.arange(stft_xfm.shape[0]):
    print("t: ", t)
    for f in np.arange(stft_xfm.shape[1]):
        R_f[:, :, f] += np.dot(stft_xfm[t, f, :], stft_xfm[t, f, :].T)

for f in np.arange(stft_xfm.shape[1]):
    R_f = R_f[:, :, f]/stft_xfm.shape[0]


mvdr_weights = np.dot(np.linalg.inv(R_f), A)/np.dot(np.dot(A, np.linalg.inv(R_f).T), A)

