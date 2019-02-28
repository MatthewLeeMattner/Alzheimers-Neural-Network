import tensorflow as tf
import numpy as np
from autoencoder import Autoencoder
from model_helper import ModelHelper

autoencoder = Autoencoder(load=True, name="1551066199.059682-autoencoder")
print(autoencoder.model.summary())

model_helper = ModelHelper(autoencoder.model)
encoded_kernel = model_helper.get_layer_kernel("encoded")
print(encoded_kernel)
print(encoded_kernel.shape)

def fully_connected_model(encoded_kernel):
   model = tf.keras.Sequential()
   conv_layer = tf.keras.layers.Conv3D(filters=150, kernel_size=(5, 5, 5), input_shape=(68, 95, 79, 1))
   model.add(conv_layer)
   model.add(tf.keras.layers.MaxPool3D(pool_size=(5, 5, 5)))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(units=800))
   model.add(tf.keras.layers.Dense(units=3))
   return model

model = fully_connected_model(None)
output = model.predict(np.random.randn(1, 68, 95, 79, 1))
print(output.shape)
print(model.summary())


