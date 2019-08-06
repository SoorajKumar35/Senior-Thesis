import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

noisy_rec, _ = sf.read('/Users/soorajkumar/Desktop/ECE496/1rec_DemoSourceWNoise_Source.wav')
clean_rec, _ = sf.read('/Users/soorajkumar/Desktop/ECE496/bfapplied_1rec_DemoSourceWNoise_Source.wav')

scale = 10
sf.write('Demo_before.wav', samplerate=96000, data=noisy_rec[:, 0]*scale)
sf.write('Demo_after.wav', samplerate=96000, data=clean_rec*scale*2)


plt.figure(1)
plt.subplot(2,1,1)
Pxx1, freqs1, bins1, im1 = plt.specgram(x=noisy_rec[:, 0],
             Fs=96000,
             mode='magnitude',
             scale='dB',
             NFFT=100,
             noverlap=50
             )
freqs1 /= 1e3
plt.title("Before beamforming")
plt.xlabel("Time(s)")
plt.ylabel("Frequency (Hz)")
plt.subplot(2,1,2)
Pxx2, freqs2, bins2, im2 = plt.specgram(x=clean_rec,
                                     Fs=96000,
                                     NFFT=100,
                                     scale='dB',
                                     mode='magnitude',
                                     noverlap=50
                                     )
freqs2 /= 1e3
plt.title("After beamforming")
plt.xlabel("Time(s)")
plt.ylabel("Frequency (Hz)")
plt.show(1)



