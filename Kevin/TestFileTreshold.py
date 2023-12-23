import pyaudio
import numpy as np
import wave

import GPIO

import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

# PyAudio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 2500

p = pyaudio.PyAudio()
interpreter = Interpreter("model.tflite")
interpreter.allocate_tensors()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def wait_for_audio():
    print("Listening...")
    while True:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        energy = np.sum(np.abs(audio_data) ** 2) / len(audio_data)
        if energy > THRESHOLD:
            print("Voice activated! Recording...")
            break

def record_audio():
    print("Recording")
    audioFrames=[]
    for i in range(int(1 * RATE / CHUNK)):  # go for a LEN seconds
        audio = stream.read(CHUNK, exception_on_overflow=False)
        audioFrames.append(np.frombuffer(audio, dtype=np.int16))

    return np.concatenate(audioFrames)

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]

  return spectrogram

def process_audiobuffer(waveform):
    waveform = waveform / 32768.0
    spectrogram = get_spectrogram(waveform)
    spectrogram = np.resize(spectrogram, (343,129,1))
    spectrogram = np.expand_dims(spectrogram, 0)

    return spectrogram

def checkModel(audio):
    x = process_audiobuffer(audio)
    x = x.astype(np.float32)

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])

    print(output_data[0])
    
    x_labels = ['boom', 'deur', 'hond', 'tafel', 'vrede', 'water']

    index = np.argmax(output_data[0])
    confidense = int(tf.nn.softmax(output_data[0])[index].numpy() * 100)
    print(f"{x_labels[index]}: {confidense}")

    if confidense > 90:
        GPIO.changeLed(x_labels[index])
        pass

while True:
    try:
        wait_for_audio()    #Waits until threshold
        audio = record_audio()  #record to test.wav
        checkModel(audio)    #Verify input

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
