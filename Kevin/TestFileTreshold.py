import pyaudio
import numpy as np
import wave

import GPIO

import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import pathlib

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
        audioFrames.append(audio)
        #audioFrames.append(np.frombuffer(audio, dtype=np.int16))

    waveform = wave.open(f"test/test.wav", "w")
    waveform.setnchannels(1)
    waveform.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    waveform.setframerate(44100)
    waveform.writeframes(b''.join(audioFrames))
    waveform.close()

    #return np.concatenate(audioFrames)

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

    if confidense > 90:
        GPIO.changeLed(x_labels[index])

while True:
    try:
        wait_for_audio()    #Waits until threshold
        record_audio()  #record to test.wav
        checkModel()    #Verify input

    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
