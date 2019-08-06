import math        #import needed modules
import pyaudio     #sudo apt-get install python-pyaudio
import numpy as np
from scipy.signal import chirp
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Generate ultrasonic signal
# source_file_name = input("Please specify a file name")
# source_file_name = '22_sec_chirp.wav'
source_file_name = 'ultrasonic_fullrange_1min_chirp_96khz.wav'
zero_duration = 2.5
duration = 60
Fs = 96000
num_channels = 1
num_sweeps = 1
t = np.arange(0, duration, 1/Fs, dtype=float)
num_samples = t.shape[0]
scale = 0.5
curr = int(Fs*zero_duration)

multi_channel_sweep = np.zeros((int((Fs*duration)*num_sweeps + Fs*2.5*num_sweeps), num_channels))

for chan in np.arange(num_sweeps):
    multi_channel_sweep[curr:curr+int(Fs*duration), chan] = chirp(t=t, f0=0, f1=48e3, t1=duration) * scale
    # multi_channel_sweep[curr:curr + int(Fs * duration), 1] = np.random.normal(0, 1, int(Fs * duration)) * scale
    curr += int(Fs*duration + int(Fs*zero_duration))
write(source_file_name, rate=Fs, data=multi_channel_sweep)

# g_noise = np.zeros((int(t.shape[0]*num_sweeps + Fs*2.5*num_sweeps), num_channels))
# curr = int(Fs*2.5)
# for chan in np.arange(num_sweeps):
#     g_noise[curr:curr+num_samples, chan] = np.random.normal(0, 0.5, num_samples)
#     curr += (num_samples + int(Fs*2.5))
# write('g_noise_8ch.wav', rate=96000, data=g_noise)
