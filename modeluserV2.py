import numpy as np
import tensorflow as tf

import pyaudio
import wave
import pathlib

import pyaudio
import math
import struct
import wave
import time
import os

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def checkModel():



    x = pathlib.Path("test/")/'test.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100)
    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = x
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

    x_labels = ['boom', 'hond', 'kat', 'uitlaat', 'vrede', 'water']
    prediction = tf.nn.softmax(output_data[0])

    print(max(prediction.numpy()))
    for i in range(len(x_labels)):
        if prediction[i].numpy() == max(prediction.numpy()):
            print("gevonden")
            print(f"{x_labels[i]}: {max(prediction.numpy())}%")


""" CHUNK = 1024
RATE = 44100
LEN = 2 """


""" stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK,input_device_index=0)
player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK) """


""" try:
    while True:
        arrayFrames=[]
        for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
            test = stream.read(CHUNK)
            data = np.fromstring(test, dtype=np.int16)
            arrayFrames.append(test)
            #player.write(data, CHUNK)
        checkModel(arrayFrames)
        arrayFrames.clear()

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate() """




Threshold = 20

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
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
                                  output=False,
                                  start=True,
                                  frames_per_buffer=chunk,input_device_index=0)

    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk, exception_on_overflow=False)
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
            input = self.stream.read(chunk, exception_on_overflow = False)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()

a = Recorder()

a.listen()
