import tensorflow as tf
import numpy as np
from autoencoder import Autoencoder
from model_helper import ModelHelper

autoencoder = Autoencoder(load=True, name="1551066199.059682-autoencoder")
print(autoencoder.model.summary())

e_weights, e_bias = autoencoder.get_weights("encoded")
print(e_weights.shape)
print(e_bias.shape)

def fully_connected_model():
   model = tf.keras.Sequential()
   conv_layer = tf.keras.layers.Conv3D(filters=150, kernel_size=(5, 5, 5), input_shape=(68, 95, 79, 1))
   model.add(conv_layer)
   model.add(tf.keras.layers.MaxPool3D(pool_size=(5, 5, 5)))
   model.add(tf.keras.layers.Flatten())
   model.add(tf.keras.layers.Dense(units=800))
   model.add(tf.keras.layers.Dense(units=3))
   return model, conv_layer

model, conv_layer = fully_connected_model()
model.compile(optimizer=tf.train.AdamOptimizer(), loss="mse")
#output = model.predict(np.random.randn(1, 68, 95, 79, 1))
print(model.summary())
print(len(conv_layer.get_weights()))
print(conv_layer.get_weights()[0].shape)
print(conv_layer.get_weights()[1].shape)


