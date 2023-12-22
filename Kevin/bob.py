#mdt pull /EdgeAi/edgeai/Kevin/test/test.wav C:

import pyaudio
import pathlib
import numpy as np
import tensorflow as tf

from keras import models

import wave

#FUNCTIES
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

#MODEL

interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()
#imported.summary()

def checkModel():
    x = pathlib.Path("test/")/'test.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=44100)
    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    print(output_data[0])
    
    x_labels = ['boom', 'deur', 'hond', 'tafel', 'vrede', 'water']

    index = np.argmax(output_data[0])
    confidense = int(tf.nn.softmax(output_data[0])[index].numpy() * 100)
    print(f"{x_labels[index]}: {confidense}")

#AUDIO
CHUNK = 1024
RATE = 44100
LEN = 1

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)
#player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

try:
    while True:
        arrayFrames=[]
        for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
            audio = stream.read(CHUNK, exception_on_overflow=False)
            #data = np.fromstring(audio, dtype=np.int16)
            arrayFrames.append(audio)
            #player.write(data, CHUNK)
        wf = wave.open("test/test.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt44))
        wf.setframerate(44100)
        wf.writeframes(b"".join(arrayFrames))
        wf.close()

        checkModel()

        arrayFrames.clear()

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
