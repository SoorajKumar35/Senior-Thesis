import numpy as np
import soundfile as sf
from scipy.signal import chirp
from scipy.io.wavfile import write
from scipy.signal import resample

# Needed constants
Fs = 96e3
total_duration = 13.5
sig_duration = 11
zero_duration = 2.5
zero_duration_samples = zero_duration * Fs
scale = 0.5
t = np.arange(0, 100e-3, 1/Fs, dtype=float)

# Fill each channel with specified acoustic signal
demo_audio = np.zeros((int(total_duration * Fs), 5))

# Speakers #2 and #4 will emit scaled-gaussian white noise
demo_audio[int(zero_duration_samples):, 2] = np.random.normal(0, 1, int(Fs * sig_duration)) * scale
demo_audio[int(zero_duration_samples):, 3] = np.random.normal(0, 1, int(Fs * sig_duration)) * scale

# Read in Keanu Reeves by Logic
desired_acoustic_signal_path = '/Users/soorajkumar/Desktop/ECE496/Keanu Reeves Logic.wav'
desired_acoustic_signal, _ = sf.read(desired_acoustic_signal_path)
desired_acoustic_signal_chan1 = desired_acoustic_signal[:, 0]

upsample_desired_signal = resample(desired_acoustic_signal_chan1[int(60 * Fs/2):int(60 * Fs/2) + int(11 * Fs/2)]*scale, int(Fs*11)) * scale
demo_audio[int(zero_duration_samples):, 0] = upsample_desired_signal
demo_audio[int(zero_duration_samples):int(zero_duration_samples) + int(100e-3*Fs), 0] = chirp(t=t, f0=0, f1=24e3, t1=100e-3) * scale

# The first speaker and 5th speaker is occupied with VCTK
human_speaker_1_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_1.wav'
human_speaker_2_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_2.wav'
human_speaker_1, _ = sf.read(human_speaker_1_path)
human_speaker_2, _ = sf.read(human_speaker_2_path)
demo_audio[int(zero_duration_samples):, 1] = resample(human_speaker_1[:int(Fs/2 * 11)] * scale, int(Fs*11))
demo_audio[int(zero_duration_samples):, 4] = resample(human_speaker_2[:int(Fs/2 * 11)] * scale, int(Fs*11))

demo_file_name = 'DemoSourceWNoise_Source.wav'
sf.write(demo_file_name, samplerate=int(Fs), data=demo_audio)

#
# import numpy as np
# import soundfile as sf
#
# Fs = 96e3
# total_duration = 13.5
# sig_duration = 11
# zero_duration = 2.5
# zero_duration_samples = zero_duration * Fs
# scale = 0.5
# t = np.arange(0, 100e-3, 1/Fs, dtype=float)
#
# from scipy.signal import resample
# from scipy.signal import chirp
#
# demo_audio = np.zeros((int(total_duration * Fs), 5))
# human_speaker_1_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_1.wav'
# human_speaker_2_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_2.wav'
# human_speaker_3_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_3.wav'
# human_speaker_4_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_4.wav'
# human_speaker_5_path = '/Users/soorajkumar/Desktop/ECE496/human_speaker_5.wav'
#
# human_speaker_1, _ = sf.read(human_speaker_1_path)
# human_speaker_2, _ = sf.read(human_speaker_2_path)
# human_speaker_3, _ = sf.read(human_speaker_3_path)
# human_speaker_4, _ = sf.read(human_speaker_4_path)
# human_speaker_5, _ = sf.read(human_speaker_5_path)
#
#
# demo_audio[int(zero_duration_samples):, 0] = resample(human_speaker_1[:int(Fs/2 * 11)] * scale, int(Fs*11))
# demo_audio[int(zero_duration_samples):, 1] = resample(human_speaker_2[:int(Fs/2 * 11)] * scale, int(Fs*11))
# demo_audio[int(zero_duration_samples):, 2] = resample(human_speaker_3[:int(Fs/2 * 11)] * scale, int(Fs*11))
# demo_audio[int(zero_duration_samples):, 3] = resample(human_speaker_4[:int(Fs/2 * 11)] * scale, int(Fs*11))
# demo_audio[int(zero_duration_samples):, 4] = resample(human_speaker_5[:int(Fs/2 * 11)] * scale, int(Fs*11))
# demo_audio[int(zero_duration_samples):int(zero_duration_samples) + int(100e-3*Fs), 0] = chirp(t=t, f0=0, f1=24e3, t1=100e-3) * scale
#
# cocktail_demo_file_name = 'cocktail_demo_audible_chirp.wav'
# sf.write(cocktail_demo_file_name, samplerate=int(Fs), data=demo_audio)
#
#
