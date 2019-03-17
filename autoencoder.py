import tensorflow as tf
import numpy as np
from time import time
from data_helper import get_dummy_patches
from model_helper import ModelHelper
# tf.enable_eager_execution()

train, test, val = get_dummy_patches()


class Autoencoder(ModelHelper):
    '''
    Class to contain functions required to setup, train and run the autoencoder part
    of the neural network structure. Extends model helper for helper functions
    '''
    def __init__(self, name=None, location="models/"):
        '''
        Sets up the model and compiles it. If name is set to None then a new instance
        of the autoencder model is created. If name is provided, a previous version of the model
        is loaded.

        @param name: The name of the model file to be loaded

        @param location: The location of the model directory
        '''
        self.model = self.model_define()
        self.set_model(self.model)
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.02), loss="mse")
        self.model.predict(np.random.randn(1, 5, 5, 5))
        if name is not None:
            if location is not None:
                self.load_model(name, location)
            else:
                self.load_model(name)

    def custom_sparse_activation_regularization(self, activations):
        '''
        Custom implementation of the sparse activation function used in the paper
        '''
        return tf.multiply(tf.divide(1, tf.size(activations)), tf.reduce_sum(activations))

    def model_define(self):
        '''
        Defines the model structure
        returns: model object
        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(125,)),
            tf.keras.layers.Dense(units=150, activation="sigmoid", name="encoded",
                                  activity_regularizer=self.custom_sparse_activation_regularization),  # Encoded representation
            tf.keras.layers.Dense(units=125),
            tf.keras.layers.Reshape(target_shape=(5, 5, 5), name="decoded")
        ])
        return model




if __name__ == "__main__":
    autoencoder = Autoencoder()
    model = autoencoder.model
    model.compile(optimizer=tf.train.MomentumOptimizer(0.02, 0.1), loss="mse")
    model.fit(train, train, epochs=10)
    autoencoder.save_model(name="autoencoder")

    import matplotlib.pyplot as plt
    from display import display_comparison_batch

    input_imgs = test[:5]
    output = model.predict(input_imgs)

    display_comparison_batch(input_imgs[:, :, :, 3], output[:, :, :, 3])
    plt.show()

'''
print(model.summary())



from model_helper import ModelHelper
helper = ModelHelper(model)
layer_kernel = helper.get_layer_kernel("encoded")
print(layer_kernel.shape)
print(layer_kernel)


'''
