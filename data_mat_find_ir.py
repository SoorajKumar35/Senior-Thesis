import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from scipy.fftpack import ifft

def get_impulse_responses(sources, recordings, Fs=96000, DFT_LEN=8192, sig_duration=100e-3, zero_duration=2.5, ultrasonic=0):
    """
    This function takes in the source file and the correspoding recoding file and returns the impulse response matrix
    per channel/microphone and per source.
    :return:
    """

    # Find the number of microphones and the number of speakers if multiple speakers or number of sweeps if multiple
    # sweeps instead

    num_channels = recordings.shape[1]
    num_sweeps_sources = int(sources.shape[0]//((sig_duration + zero_duration)*Fs))

    # Offset by the number of zeros before a chirp
    offset = int(Fs * zero_duration)
    x = np.empty((int(sig_duration * Fs), num_channels, num_sweeps_sources)) # [(sampling rate x duration of chirp) x num_microphones x num_speakers/sweeps]
    s = np.empty((int(sig_duration * Fs), num_sweeps_sources)) # [((sampling rate x duration of chirp) x num_speakers/sweeps)]

    for src in range(num_sweeps_sources):
        x[:, :, src] = recordings[offset:offset + int(Fs * sig_duration), :]
        s[:, src] = sources[offset:int(sig_duration*Fs + offset), src]
        offset += int((sig_duration + zero_duration)*Fs)

    offset = int(Fs*zero_duration)

    if not ultrasonic:
        a = np.zeros((int(Fs*sig_duration/2), num_channels, num_sweeps_sources))
    else:
        a = np.zeros((int(Fs*sig_duration), num_channels, num_sweeps_sources))

    for src in range(num_sweeps_sources):
        curr_s = sources[offset:(int(sig_duration*Fs) + offset), src]
        curr_rec = recordings[offset:(int(sig_duration*Fs) + offset), :]

        if not ultrasonic:
            curr_s_fil = decimate(x=curr_s, q=2, ftype='fir', axis=0)
            curr_rec_fil = decimate(x=curr_rec, q=2, ftype='fir', axis=0)
        else:
            curr_s_fil = curr_s
            curr_rec_fil = curr_rec

        S = np.fft.fft(curr_s_fil, axis=0)
        X = np.fft.fft(curr_rec_fil, axis=0)
        A = X/S[:, np.newaxis]
        a[:, :, src] = ifft(A, axis=0)
        offset += int((sig_duration + zero_duration)*Fs)

    return x, s, a
