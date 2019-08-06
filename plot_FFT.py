import wave
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fftpack import fft
import soundfile as sf

def plot_FFT(audio_data, Fs, channels=8):
    """
    Plots the frequeny response of an input audio file
    :param audio_file: The file path to the input audio file
    :return: Plots the FFT and returns
    """
    # X-axis
    T = audio_data.shape[0]/Fs
    freq = np.arange(0, audio_data.shape[0])/T

    # Find the frequency response
    A = np.fft.fft(audio_data, axis=0)
    db_A = 20*np.log10(abs(A))

    # Plot the FFT
    for chan in np.arange(channels):
        plt.subplot(4, 2, chan)
        plt.figure(1)
        plt.plot(freq[:int(db_A.shape[0]/2)], db_A[:int(db_A.shape[0]/2), chan])
        plt.xlabel("Frequency")
        plt.ylabel("Frequency Response")
        plt.show(1)


def find_and_plot_IR(x, y, buffer=0):
    """
    This function takes in the source signals (x) and the recorded signals (y) and finds the impulse response.
    :param x: The source signal
    :param y: The recorded signal per microphone. Thus, this would be a multi-dim array
    :param buffer: If the signal has zeros between each chirp, then we include the space per buffer here
    :return: Impulse response in time-domain
    """

    # Find the FFT of the source and recorded signals
    X = np.fft.fft(x, axis=0)
    Y = np.fft.fft(y, axis=0)
    A = np.zeros(Y.shape)

    for src in np.arange(X.shape[1]):
        for mic in np.arange(Y.shape[1]):
            pass


def process_rec_data(y, Fs, buffer, num_sources):
    """
    This function takes in recorded data from the microphones and creates a data matrix that can be used to find IRs.
    It outputs a (duration, num channels, num sources) data array.
    :param y: The recorded signals of size (total time of recording * Fs, num mics recording)
    :param Fs: The sampling rate the recording was taken in
    :param buffer: The length of time between consecutive source signal activations
    :param num_sources: The total number of sources of signals in the experiment
    :return: The record data data matrix
    """

    buffer_samples = buffer * Fs



    y_data_mat = np.zeros(Fs*,y.shape[1], num_sources)

    for src in np.arange(num_sources):







filename = '/Users/soorajkumar/Desktop/ECE496/SoundFiles/better_ul_recording.wav'
audio_data, Fs = sf.read(filename)
plot_FFT(audio_data, Fs)

filename2 = '/Users/soorajkumar/Desktop/ECE496/SoundFiles/nice_speaker_recording.wav'
audio_data2, Fs2 = sf.read(filename2)
plot_FFT(audio_data2, Fs2)



# plt.figure(1)
# plt.specgram(x=audio_data[:, 0], Fs=Fs)
# plt.title("Spectogram vs time")
# plt.show(1)
pass

# Fs, audio_data = read(filename)

# non_zero_first_channel = audio_data[np.nonzero(audio_data[:, 0]), 0].T
# T = np.ceil(non_zero_first_channel.shape[0]/Fs)
# t = np.arange(0, T, T/non_zero_first_channel.shape[0])
# plt.figure(1)
# plt.plot(t, non_zero_first_channel)
# plt.show(1)
# pass
#
# fft_audio_data = fft(audio_data[:,1], axis=0)
# fft_mag = np.abs(fft_audio_data)
# fft_phase = np.angle(fft_audio_data)
#
# freq = range(int(audio_data.shape[0]/2))/T
# plt.figure(1)
# plt.plot(freq, fft_audio_data[:int(fft_audio_data.shape[0]/2), 0], 'r')
# plt.show(1)

# audio_file = wave.open(filename, 'r')
# audio_frames = []
# for af in range(audio_file.getnframes()):
#     audio_frames += [int(audio_file.readframes(1).hex(), 16)]
# audio_frames_array = np.array(audio_frames)
# n = len(audio_frames_array)
# audio_frames_fft = np.fft.fft(audio_frames)/n
# audio_frames_magnitude = np.square(np.abs(audio_frames_fft))
#
# Fs = audio_file.getframerate()
# k = np.arange(n)
# T = math.ceil(n/Fs)
# frq = k/T
#
# plt.figure(1)
# plt.ylabel("Amplitude/Magnitude")
# plt.xlabel("Frequency")
# plt.plot(frq, audio_frames_magnitude, 'r')
# plt.show(1)
