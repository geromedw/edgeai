import pyaudio

import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models

import wave
import platform


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


#FUNCTIES
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

#tlfite model importeren
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)



def checkModel(data):
    waveform = wave.open(f"test.wav", "w")
    waveform.setnchannels(1)
    waveform.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    waveform.setframerate(16000)
    waveform.writeframes(b''.join(data))
    waveform.close()


    x = pathlib.Path("test/")/'test.wav'
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]
    # Get input and output tensors.

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0][x]
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.tensor(output_details[0]['index'])
    print(output_data)

    """ x = pathlib.Path("test/")/'test.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    prediction = interpreter(x)
    x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
    prediction = tf.nn.softmax(prediction[0])

    print(max(prediction.numpy()))
    for i in range(len(x_labels)):
        if prediction[i].numpy() == max(prediction.numpy()):
            print("gevonden")
            print(f"{x_labels[i]}: {max(prediction.numpy())}%")
 """

CHUNK = 2 ** 5
RATE = 16000
LEN = 2

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK
)
player = p.open(
    format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK
)

try:
    while True:
        arrayFrames=[]
        for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
            test = stream.read(CHUNK)
            data = np.fromstring(test, dtype=np.int16)
            arrayFrames.append(test)
            player.write(data, CHUNK)
        checkModel(arrayFrames)
        arrayFrames.clear()

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()