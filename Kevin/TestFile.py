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
imported = models.load_model("EDGEAI/saved.keras")
#imported.summary()

def checkModel(data):
    waveform = wave.open(f"test/test.wav", "w")
    waveform.setnchannels(1)
    waveform.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    waveform.setframerate(16000)
    waveform.writeframes(b''.join(data))
    waveform.close()

    x = pathlib.Path("test/")/'test.wav'
    x = tf.io.read_file(str(x))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
    x = tf.squeeze(x, axis=-1)

    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    prediction = imported(x)
    x_labels = ['boom', 'deur', 'hond', 'tafel', 'vrede', 'water']
    prediction = tf.nn.softmax(prediction[0])

    #print(prediction.numpy())
    for i in range(len(x_labels)):
        if prediction[i].numpy() == max(prediction.numpy()):
            print(f"{x_labels[i]}: {max(prediction.numpy())}%")

#AUDIO
CHUNK = 1024
RATE = 16000
LEN = 1

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
#player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

try:
    while True:
        arrayFrames=[]
        for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
            audio = stream.read(CHUNK)
            #data = np.fromstring(audio, dtype=np.int16)
            arrayFrames.append(audio)
            #player.write(data, CHUNK)
        checkModel(arrayFrames)
        arrayFrames.clear()

except KeyboardInterrupt:
    stream.stop_stream()
    stream.close()
    p.terminate()
