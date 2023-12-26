#EdgeAI
#Team AI
#Moens Ralph, De Wilde Gerome, Lahey Kevin

#Imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras import layers
from keras import models

#Variables
SHOW_PLOT = False
DATA_DIR = "data/"
SAMPLE_RATE = 44100
EPOCHS = 10

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Deviding train data and test data
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=DATA_DIR,
    batch_size=32,
    validation_split=0.2,
    seed=0,
    output_sequence_length=SAMPLE_RATE,
    subset='both')

#Get the names of folders and print them
label_names = np.array(train_ds.class_names)
print("---------------------------------------")
print("label names:", label_names)

#Remove the extra audio channel
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

#Keep the test and val ds seperate
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

#Print the shape of the audio and lables
for example_audio, example_labels in train_ds.take(1):
  print(f"Audio shape:{example_audio.shape}")
  print(f"Label shape: {example_labels.shape}")

#Plot 9 audio waveforms
if SHOW_PLOT:
  plt.figure(figsize=(16, 10))
  rows = 3
  cols = 3
  n = rows * cols
  for i in range(n):
    plt.subplot(rows, cols, i+1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])

#Convert waveform to specogram with tf
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

#Print the label, waveform shape, spectrogram shape, for 3 items
for i in range(3):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print(f"Label: {label}")
  print(f"Waveform shape: {waveform.shape}")
  print(f"Spectrogram shape: {spectrogram.shape}")

#Plot spectogram
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

#Plot waveform and spectrogram
if SHOW_PLOT:
  fig, axes = plt.subplots(2, figsize=(12, 8))
  timescale = np.arange(waveform.shape[0])
  axes[0].plot(timescale, waveform.numpy())
  axes[0].set_title('Waveform')
  axes[0].set_xlim([0, SAMPLE_RATE])

  plot_spectrogram(spectrogram.numpy(), axes[1])
  axes[1].set_title('Spectrogram')
  plt.suptitle(label.title())

#Create spectrogram dataset from the audio datasets
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

#Plot 9 spectrograms
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

if SHOW_PLOT:
  rows = 3
  cols = 3
  n = rows*cols
  fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

  for i in range(n):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      plot_spectrogram(example_spectrograms[i].numpy(), ax)
      ax.set_title(label_names[example_spect_labels[i].numpy()])

#To reduce read latency while training the model
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

#Get input shape for the model
input_shape = example_spectrograms.shape[1:]
print(f"Input shape: {input_shape}")
num_labels = len(label_names)

#Normalize each pixel in the image based on its mean and standard deviation
norm_layer = layers.Normalization()
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

#Build the model
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

#Print the summary of the model
model.summary()

#Compile the model with adam optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

#Train the model for x EPOCHS
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

#Plot the training and validation curves
if SHOW_PLOT:
  metrics = history.history
  plt.figure(figsize=(16,6))
  plt.subplot(1,2,1)
  plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
  plt.legend(['loss', 'val_loss'])
  plt.ylim([0, max(plt.ylim())])
  plt.xlabel('Epoch')
  plt.ylabel('Loss [CrossEntropy]')

  plt.subplot(1,2,2)
  plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
  plt.legend(['accuracy', 'val_accuracy'])
  plt.ylim([0, 100])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy [%]')

#Test the models performance
model.evaluate(test_spectrogram_ds, return_dict=True)

#Show confusion matrix
if SHOW_PLOT:
  y_pred = model.predict(test_spectrogram_ds)
  y_pred = tf.argmax(y_pred, axis=1)
  y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
  confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_mtx,
              xticklabels=label_names,
              yticklabels=label_names,
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()

#Show how well the model did on an example file
if SHOW_PLOT:
  x = f"{DATA_DIR}deur/35.wav"
  x = tf.io.read_file(str(x))
  x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=SAMPLE_RATE)
  x = tf.squeeze(x, axis=-1)
  waveform = x
  x = get_spectrogram(x)
  x = x[tf.newaxis,...]

  prediction = model(x)
  plt.bar(label_names, tf.nn.softmax(prediction[0]))
  plt.title('deur')
  plt.show()

#If Y then save the model
save = input("Model opslaan? y/n: ")
if save =="y":
  #save model with folder name "SavedModel"
  model.save("SavedModel")

  #Convert to tflite
  converter = tf.lite.TFLiteConverter.from_saved_model("SavedModel")
  tflite_model=converter.convert()

  #Save the file
  with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
  print("Model saved as 'model.tflite'")
