import numpy as np
import tensorflow as tf

import pyaudio
import wave
import pathlib


def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def checkModel(data):
    waveform = wave.open(f"test.wav", "w")
    waveform.setnchannels(1)
    waveform.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    waveform.setframerate(16000)
    waveform.writeframes(b''.join(data))
    waveform.close()

    x = 'test.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
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


CHUNK = 2 ** 5
RATE = 16000
LEN = 2

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK,input_device_index=0
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