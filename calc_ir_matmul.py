import numpy as np
import soundfile as sf
import time

# source_fname = '/Users/soorajkumar/Desktop/ECE496/ul_chirp_w_space.wav'
# recording_fname = '/Users/soorajkumar/Desktop/ECE496/1_rec_ultrasonic_chirp_w_spaces.wav'

source_fnmae = '7th_src_chrip.wav'
recording_fname = '5_rec_7th_src_chrip.wav'

sources, Fs_src = sf.read(source_fnmae)
recordings, Fs_rec = sf.read(recording_fname)

recordings = recordings[:sources.shape[0], :]

num_samples, num_srcs, num_recs = Fs_rec*11, 5, recordings.shape[1]
sig_duration, zero_duration = 11, 2.5


offset = int(Fs_rec*zero_duration)
x = np.zeros((num_samples, num_recs, num_srcs))
src_data_mat = np.zeros((num_samples, num_recs, num_srcs))

for src in range(num_srcs):
    x[:, :, src] = recordings[offset:offset + int(Fs_src*sig_duration) - 1, :]
    offset += int(Fs_src*(zero_duration + sig_duration))

src_conv_mat = np.zeros((num_samples, 2*num_samples, num_srcs))
a = np.zeros((num_samples, num_recs, num_srcs))

for s in range(num_srcs):
    offset = int(Fs_rec*2.5)
    for row in range(num_samples):
        start = time.time()
        src_conv_mat[row, row:(row + num_samples), s] = sources[offset:offset+(Fs_rec * 11) - 1, 6]
        print("Time for one loop: ", time.time() - start)
    offset += int(Fs_rec(11 + 2.5))

    start = time.time()
    a_store = np.linalg.lstsq(src_conv_mat[:, :, src], x[:, :, src])
    print("Time to get one lstsq sol: ", int(time.time() - start))
    a[:, :, src] = a_store


