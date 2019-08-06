import numpy as np
import soundfile as sf
from data_mat_find_ir import get_impulse_responses
import matplotlib.pyplot as plt
import os
from scipy.linalg import solve_toeplitz
from sklearn.metrics import mean_squared_error

dft_len = 8192
Fs = 96e3
num_sources = 1
num_channels = 8

# Plot the impulse response of the 1min long audible range chirp

aud_chirp_1min_path = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_1min/audible_chirp_1min_96khz.wav'
rec_aud_chirp_1min_path = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_1min/4_lowerloud_recording_chirp_1min.wav'

aud_chirp_1min, _ = sf.read(aud_chirp_1min_path)
rec_aud_chirp_1min, _ = sf.read(rec_aud_chirp_1min_path)
rec_aud_chirp_1min = rec_aud_chirp_1min[:aud_chirp_1min.shape[0], :]

_, _, ir_aud_chirp_1min = get_impulse_responses(sources=aud_chirp_1min,
                                                recordings=rec_aud_chirp_1min,
                                                sig_duration=60,
                                                ultrasonic=0)

# Code below to plot the impulse response at every microphone for the 1min chirp

first_10ms = int(Fs*50e-3)
#
# plt.figure(60)
# for src_idx in range(num_sources):
#     for chan_idx in range(num_channels):
#         # plt.subplot(4, 2, chan_idx+1)
#         plt.plot(ir_aud_chirp_1min[:first_10ms, chan_idx, src_idx])
#         plt.xlabel('n')
#         plt.ylabel('a[n]')
#         plt.title('Microphone: ' + str(chan_idx))
#         break
#     break
# plt.tight_layout()
# plt.show(60)

# Find an plot the impulse responses of various lengths audible chirps and plot them. Then, find the mean squared error
# ultrasonic_24_48_chirps
aud_chirp_path = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_1s/ultrasonic_24_48Khz_chirps_wNoise/ultrasonic_24_48_3s_wNoise_96khz.wav'
rec_aud_chirp_path = '/Users/soorajkumar/Box/MicArrayProjects/Sooraj Data/Sources and Recordings/chirps_1s/ultrasonic_24_48Khz_chirps_wNoise/1_rec_ultrasonic_24_48_3s_wNoise_96khz.wav'
aud_chirp, _ = sf.read(aud_chirp_path)
rec_aud_chirp, _ = sf.read(rec_aud_chirp_path)
rec_aud_chirp = rec_aud_chirp[:aud_chirp.shape[0], :]

rec_data, src_data, ir_aud_chirp = get_impulse_responses(sources=aud_chirp,
                                                         recordings=rec_aud_chirp,
                                                         sig_duration=1,
                                                         ultrasonic=1)

to_compute = rec_data.shape[0]
h_matrix = np.empty((to_compute, num_channels, num_sources))
#
# for src_idx in range(num_sources):
#     for chan_idx in range(num_channels):
#
#         print("src_idx:", src_idx)
#         print("chan_idx:", chan_idx)
#
#         # Test to check if it can succesfully invert a matrix that has more rows than cols
#
#         toeplitz_row = np.zeros(to_compute)
#         toeplitz_row[0] = src_data[0, src_idx]
#         toeplitz_col = src_data[:to_compute, src_idx]
#         b = rec_data[:to_compute, chan_idx, src_idx]
#         h_matrix[:, chan_idx, src_idx] = solve_toeplitz((toeplitz_col[:, np.newaxis], toeplitz_row.T), b)
#
#         plt.figure(1)
#         plt.plot(h_matrix[:, chan_idx, src_idx])
#         plt.show()

plt.figure(1)
for src_idx in range(num_sources):
    for chan_idx in range(num_channels):
        # plt.subplot(4, 2, chan_idx+1)
        plt.plot(ir_aud_chirp[:first_10ms, chan_idx, src_idx])
        # plt.plot(ir_aud_chirp_1min[:first_10ms, 0, 0])
        # diff = ir_aud_chirp[:first_10ms, 0, 0] - ir_aud_chirp_1min[:first_10ms, 0, 0]
        # diff_fft = np.fft.fft(diff, axis=0)
        # plt.plot(diff)
        plt.xlabel('n')
        plt.ylabel('a[n]')
        # plt.legend(['100ms', '1min'])
        plt.title('Microphone: ' + str(chan_idx))
        break
    break
plt.tight_layout()
plt.show(1)

# Print out the error

mse_sum = 0
for src_idx in range(num_sources):
    for chan_idx in range(num_channels):
        mse_sum += (mean_squared_error(ir_aud_chirp[:first_10ms, chan_idx, src_idx]/np.max(ir_aud_chirp[:first_10ms, chan_idx, src_idx]),
                                       ir_aud_chirp_1min[:first_10ms, chan_idx, src_idx]/np.max(ir_aud_chirp_1min[:first_10ms, chan_idx, src_idx])))

print(mse_sum/8)



# Find and plot impulse responses for ultrasonic signals w and wo noise in the recordings.