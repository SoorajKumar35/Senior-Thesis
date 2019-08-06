import numpy as np
import wave
import pyaudio

def srp_phat(X):
    """
    :param X: (N x num mics) - Input signal recorded with a variable num of mics
    :return: The angle of arrival (the angle where power is maximum)
    """

    angles = range(0, 360)*(np.pi/180)
    C = 343
    MIC_POS = [[np.cos((x*np.pi)/180) for x in range(0, 300, 60)], [np.sin((x*np.pi)/180) for x in range(0, 300, 60)], [0 for x in range(1, 7)]]
    MIC_POS = np.concatenate((np.zeros(3), np.array(MIC_POS)), axis=1)
    omega = (range(0, X.shape[0])/X.shape[0]) * 2 * np.pi
    delays = -1 * np.matmul(MIC_POS.T, np.array([[np.cos(x) for x in angles], [np.sin(x) for x in angles], 0]))

    X = X[:, :7]

    # Phase transform
    X_fft = np.fft.fft(X)
    X_fft = X_fft/np.abs(X_fft)

    # Frequency domain beamforming
    P = np.zeros((angles.shape[0]))
    for angle in np.arange(angles.shape[0]):
        W = np.exp(1j * delays[:, angle]*omega).T
        Y = np.sum(W*X_fft, axis=1)
        P[angle] = np.mean(np.square(np.abs(Y)))

def record_respeaker(rate, channels, chunk_size):

    pyaudio_in = pyaudio.PyAudio()

    def _callback(self, in_data, frame_count, time_info, status):
        return None, pyaudio.paContinue

    # Get the index of the respeaker array
    device_index = None
    for i in range(pyaudio_in.get_device_count()):
        dev = pyaudio_in.get_device_info_by_index(i)
        name = dev['name'].encode('utf-8')
        print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
        if dev['maxInputChannels'] == channels:
            print('Use {}'.format(name))
            device_index = i
            break

    if device_index == None:
        raise Exception("Can't find the respeaker")

    # Create a Pyaudio stream and start recording
    stream = pyaudio_in.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        frames_per_buffer=chunk_size,
        input=True,
        input_device_index=device_index,
        start=False,
        stream_callback=_callback,
        
    )

    record_seconds = 5
    out_data = []
    print("Recording....")

    for c in range(int((rate/chunk_size) * record_seconds)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        out_data += data

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    pyaudio_in.terminate()
    
    waveFile = wave.open('SoundFiles/sample_respeaker_recording.wav', 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(pyaudio_in.get_sample_size(pyaudio_in.paInt16))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(out_data))
    waveFile.close()
    
# wave_obj = wave.open('/Users/soorajkumar/Desktop/ECE496/SoundFiles/sombrero_20180327.wav')
# param_tuple = wave_obj.getparams()
# nframes = param_tuple[3]
# sombrero_rec = wave_obj.readframes(nframes)
# srp_phat(sombrero_rec)
# record_respeaker(16000, 8, int(16000/4))




