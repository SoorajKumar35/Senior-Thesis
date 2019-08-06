import numpy as np
import scipy
from scipy.io.wavfile import write
from scipy import signal
import matplotlib.pyplot as plt
from math import *
import wave

class get_angle_of_arrival:

    def __init__(self):

        # First, we either record a signal or we generate a custom signal and then find
        # the room impulse response
        chirp = self.generate_custom_signal()
        # rir = self.get_RIR(chirp, chirp)
        # wave_read_obj = wave.open('SoundFiles/chirp_8ch_48.wav')
        # param_tuple = wave_read_obj.getparams()
        # nframes = param_tuple[3]
        # chirp = wave_read_obj.readframes(nframes)
        # chirp = np.array(chirp)
        pass


    def get_input_mat_and_IRs(self, source_signals, received_signals):
        """
        :param source_signal: The signal that is played at the sources (source length vs num sources)
        :param received_signal: The signal that is received at the microphones (for the test it would the sombrero)
                                (received length, num microphones)
        :return: input_data_matrix (duration, num channels, num sources)
                impulse responses (duration, num sources)
        """

        num_sources, num_channels = source_signals.shape[1], received_signals.shape[1]
        start_times = [None] * num_sources

        for src in range(num_sources):
            start_times[src] = (source_signals[:, src] != 0).argmax(axis=0)

        test_duration = np.diff(np.sort(start_times)).min()


    def generate_custom_signal(self):

        # Here we generate a sine sweep as seen from the sound source
        # We are assuming that the RIR is 1 for when there is no sound signal

        N = 48000
        fs = 48000.0
        sine_list_x = []
        K = (10000.0 - 1000.0) / (48000.0)
        for x in range(N):
            sine_list_x.append(sin(2 * pi * (1000.0 * (x / 48000.0) + (K / 2.0) * (x ** 2) / (48000.0))))

        xf = np.linspace(0.0, fs / 2.0, N/2)
        chirp_FFT = np.fft.fft(sine_list_x)
        # Now, we plot the FFT to check if we created the signal correctly
        # plt.figure(1)
        # plt.plot(xf, abs(chirp_FFT[:int(N/2)]))
        # plt.xlabel("Frequency")
        # plt.ylabel("FFT")
        # plt.show(1)

        chirp_signal = np.array(sine_list_x)
        write(filename='linear_chirp_sample.wav', rate=int(fs), data=chirp_signal)
        return sine_list_x

    def get_RIR(self, original_signal, received_signal):

        # To get the RIR from the received signal, we use the DFT rep of both to derive the DFT of the RIR
        # We then use an inverse DFT to retreive the time domain RIR

        o_s_fft = np.fft.fft(original_signal)
        r_s_fft = np.fft.fft(received_signal)

        rir_fft = r_s_fft/o_s_fft

        rir = np.fft.ifft(rir_fft)

        return rir

    def compute_steering_vector(self, mic_coords):
        """
        :param mic_coords: (N x (2d coords)
        :return: steering vector h
        """

        return np.exp(np.complex())


    def find_angle_of_arrival(self, input_spec, steering_vec):
        """
        :param input_spec: The input data spectogram
        :param steering_vec: Steering vectors to search over
        :return: argmax of steered response power to get max angle
        """

        F, T, M = input_spec.shape
        num_angles = steering_vec.shape[2]

        X = np.transpose(input_spec, (2, 0, 1))
        p = np.zeros(T, num_angles)

        for t in np.arange(T):
            X_phat = input_spec[:, :, t]
            X_phat = X_phat/np.abs(X_phat)
            for w in np.arange(num_angles):
                X_steered = X_phat * np.squeeze(steering_vec[:, :, w]).T
                Yw = np.sum(X_steered, 0)
                p[t, w] = Yw*Yw.T
        return np.argmax(X_steered, 0)


AoA = get_angle_of_arrival()