import tensorflow as tf
modeldir = 'C:/Users/gerom/OneDrive/Bureaublad/ap/edge AI/audiotest/saved'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(modeldir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)