import pyaudio
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models

import wave

from functions import *


imported = models.load_model("saved1.keras")


def checkModel(data):

    prediction = imported(data)
    x_labels = ['boom', 'hond', 'kat', 'uitlaat', 'vrede', 'water']
    prediction = tf.nn.softmax(prediction[0])

    #print(prediction.numpy())
    for i in range(len(x_labels)):
        if prediction[i].numpy() == max(prediction.numpy()):
            print(f"{x_labels[i]}: {max(prediction.numpy())}%")

#
while True:
    audio = get_audio()
    print(audio.size)
    spectrogram = preprocess_audiobuffer(audio, (124,129,1))
    checkModel(spectrogram)