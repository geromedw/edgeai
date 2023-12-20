import pyaudio
import pathlib
import numpy as np
import tensorflow as tf

from keras import models

import wave

from functions import *

#import tflite_runtime.interpreter as tflite


#imported = models.load_model("saved1.keras")
interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()

def checkModel(data):

    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.tensor(input_tensor_index)()[0] = data

    interpreter.invoke()
    
    output_tensor_index = interpreter.get_output_details()[0]['index']
    output_data = interpreter.tensor(output_tensor_index)()[0]
    print(output_data)
    
    #prediction = interpreter(data)
    x_labels = ['boom', 'hond', 'kat', 'uitlaat', 'vrede', 'water']

    print(x_labels[np.argmax(output_data)])
    # prediction = tf.nn.softmax(prediction[0])

    # #print(prediction.numpy())
    # for i in range(len(x_labels)):
    #     if prediction[i].numpy() == max(prediction.numpy()):
    #         print(f"{x_labels[i]}: {max(prediction.numpy())}%")

#
while True:
    audio = get_audio()
    print(audio.size)
    spectrogram = preprocess_audiobuffer(audio, (124,129,1))
    checkModel(spectrogram)
