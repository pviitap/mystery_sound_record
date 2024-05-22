import pyaudio
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import struct
import wave
import time
import os

MIN_HZ=4500
MAX_HZ=5500
THRESHOLD = 10
TIMEOUT_LENGTH = 2

SHORT_NORMALIZE = (1.0/32768.0)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SWIDTH = 2


f_name_directory = r'./output'

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / SWIDTH
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def autocorrelation(self, x):
        """Calculate the autocorrelation of the signal"""
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]


    def get_dominant_frequency(self, signal, rate):
        # Perform FFT
        N = len(signal)
        T = 1.0 / RATE
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]

        # Find the peak in the FFT magnitude spectrum
        idx = np.argmax(np.abs(yf[:N//2]))
        frequency = xf[idx]
        return frequency


    def get_pitch(self, signal, rate):
        """Estimate the pitch of the signal"""
        # Calculate the autocorrelation of the signal
        corr = self.autocorrelation(signal)
        d = np.diff(corr)
        
        # Find the first peak
        start = np.nonzero(d > 0)[0][0]
        
        # Find the peaks in the autocorrelation
        peaks, _ = find_peaks(corr[start:])
        if len(peaks) == 0:
            return 0
        
        # The first peak corresponds to the fundamental frequency
        peak = peaks[0]
        
        # Convert the peak index to a frequency
        pitch = rate / (start + peak)
        return pitch

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=CHUNK)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:
            data = self.stream.read(CHUNK)
            if self.rms(data) >= THRESHOLD: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)


        audio_data = np.frombuffer(b''.join(rec), dtype=np.int16)
        #pitch = self.get_pitch(audio_data,RATE)
        #print(f"The estimated pitch is {pitch:.2f} Hz")
        frequency = self.get_dominant_frequency(audio_data,RATE)
        print(f"The dominant frequency is {frequency:.2f} Hz")
        if frequency > MIN_HZ and frequency < MAX_HZ:
            print("mystery sound possibly detected")
            self.write(b''.join(rec))
        else:
            print("it was something else")

    def write(self, recording):
        n_files = len(os.listdir(f_name_directory))

        filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')



    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(CHUNK)
            rms_val = self.rms(input)
            if rms_val > THRESHOLD:
                self.record()

a = Recorder()

a.listen()
