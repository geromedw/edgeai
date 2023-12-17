import pyaudio
import pathlib
""" import numpy as np
import matplotlib.pyplot as plt """

import tensorflow as tf
from keras import models

import wave

import pyaudio
import math
import struct
import wave
import time
import os


#FUNCTIES
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]

  return spectrogram

#MODEL
imported = models.load_model("saved1.keras")
#imported.summary()

def checkModel(data=None):
    # waveform = wave.open(f"test/test.wav", "w")
    # waveform.setnchannels(1)
    # waveform.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    # waveform.setframerate(16000)
    # waveform.writeframes(b''.join(data))
    # waveform.close()

    x = pathlib.Path("test/")/'test.wav'
    x = tf.io.read_file(str(x))

    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)

    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    prediction = imported(x)
    x_labels = ['boom', 'hond', 'kat', 'uitlaat', 'vrede', 'water']
    prediction = tf.nn.softmax(prediction[0])

    #print(prediction.numpy())
    for i in range(len(x_labels)):
        if prediction[i].numpy() == max(prediction.numpy()):
            print(f"{x_labels[i]}: {max(prediction.numpy())}%")

#AUDIO

Threshold = 20

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 2

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        self.write(b''.join(rec))

    def write(self, recording):
        n_files = len(os.listdir("test"))

        filename = os.path.join("test", 'test.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')
        checkModel()

    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()

a = Recorder()

a.listen()