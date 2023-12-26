#EdgeAI
#Team AI
#Moens Ralph, De Wilde Gerome, Lahey Kevin

#Imports
import pyaudio
import numpy as np
import tensorflow as tf #Used for the spectrogram
from tflite_runtime.interpreter import Interpreter
import GPIO #Used for the leds


#Variables
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 2500

p = pyaudio.PyAudio()

interpreter = Interpreter("model.tflite")
interpreter.allocate_tensors()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

#Wait for threshold to be exeeded
def wait_for_audio():
    print("Luisteren...")
    while True:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        energy = np.sum(np.abs(audio_data) ** 2) / len(audio_data)
        if energy > THRESHOLD:
            print("Geluid!, opnemen...")
            break

#Record the audio and store in array
def record_audio():
    audioFrames=[]
    for _ in range(int(1 * RATE / CHUNK)):  # go for a LEN seconds
        audio = stream.read(CHUNK, exception_on_overflow=False)
        audioFrames.append(np.frombuffer(audio, dtype=np.int16))

    return np.concatenate(audioFrames)

#Convert waveform to spectrogram using tf
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]

  return spectrogram

#Make the audio array ready to convert and return sepctrogram
def process_audiobuffer(waveform):
    waveform = waveform / 32768.0   #convert to [-1,1]
    spectrogram = get_spectrogram(waveform)
    spectrogram = np.resize(spectrogram, (343,129,1))   #resize to shape requerid by nn
    spectrogram = np.expand_dims(spectrogram, 0)    #Add dimesion to match input of NN

    return spectrogram

#Softmax for the output
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#Verifies the audio with the model
def checkModel(audio):
    spectrogram = process_audiobuffer(audio)
    spectrogram = spectrogram.astype(np.float32)

    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], spectrogram)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

    predictions = softmax(output_data)
    print(predictions)
    
    x_labels = ['boom', 'deur', 'hond', 'tafel', 'vrede', 'water']

    index = np.argmax(predictions)  #Get index of max value
    confidense = int(predictions[index]*100)    #Convert to %
    print(f"{x_labels[index]}: {confidense}%")  #Print result

    if confidense > 90:
        GPIO.changeLed(x_labels[index])

while True:
    try:
        wait_for_audio()    #Waits until threshold
        audio = record_audio()  #record
        checkModel(audio)    #Verify input

    except KeyboardInterrupt:
        GPIO.turnOff()  #Turn leds off
        stream.stop_stream()
        stream.close()
        p.terminate()