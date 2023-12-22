import pyaudio


import tensorflow as tf

#AUDIO
CHUNK = 1024
RATE = 16000
LEN = 1

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)
#player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

def get_audio():
    arrayFrames=[]
    for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
        audio = stream.read(CHUNK)
        #player.write(audio, CHUNK)

        arrayFrames.append(np.frombuffer(audio, dtype=np.int16))

    return np.concatenate(arrayFrames)

#Audio conversion
import numpy as np

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]

  return spectrogram

def preprocess_audiobuffer(waveform, expected_shape):
    """
    waveform: ndarray of size (16000, )
    
    output: Spectrogram Tensor of size: (1, `height`, `width`, `channels`)
    """
    # Normalize from [-32768, 32767] to [-1, 1]
    waveform = waveform / 32768.0

    # Get the spectrogram.
    spectrogram = get_spectrogram(waveform)

    # Resize or pad the spectrogram to match the expected input shape
    spectrogram = np.resize(spectrogram, expected_shape)

    # Add one dimension to match the expected input shape.
    spectrogram = np.expand_dims(spectrogram, 0)

    return spectrogram
