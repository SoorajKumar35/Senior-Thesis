import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_FFT(frequency):

    # Generate the sine wave
    frequency = np.arange(0,200)
    ang_freq = 2 * np.pi * frequency
    sine_wave_samples = np.sin(ang_freq*np.arange(0, 2*np.pi, 2*np.pi/200))
    sine_wave_FFT = np.fft.fft(sine_wave_samples)

    # Plot the sine wave
    plt.figure(1)
    plt.plot(np.arange(0, sine_wave_FFT.shape[0]), np.sqrt(np.square(np.abs(sine_wave_FFT))))
    plt.xticks(np.arange(0,100))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("FFT of generic sine wave")
    plt.show(1)

freq = 4
generate_and_plot_FFT(freq)