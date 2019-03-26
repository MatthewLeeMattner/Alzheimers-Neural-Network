import tensorflow as tf
import numpy as np
from model_helper import ModelHelper
import utils
# tf.enable_eager_execution()


class Autoencoder(ModelHelper):
    '''
    Class to contain functions required to setup, train and run the autoencoder
    part of the neural network structure.
    Extends model helper for helper functions
    '''
    def __init__(self, name=None, location="models/"):
        '''
        Sets up the model and compiles it. If name is set to None
        then a new instance of the autoencder model is created.
        If name is provided, a previous version of the model
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

    def sparse_activation_regularization(self, activations):
        '''
        Custom implementation of the sparse activation
        function used in the paper
        '''
        return tf.multiply(
            tf.divide(1, tf.size(activations)),
            tf.reduce_sum(activations)
        )

    def model_define(self):
        '''
        Defines the model structure
        returns: model object
        '''
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(125,)),
            tf.keras.layers.Dense(
                units=150,
                activation="sigmoid",
                name="encoded",
                activity_regularizer=self.sparse_activation_regularization
            ),
            tf.keras.layers.Dense(units=125),
            tf.keras.layers.Reshape(target_shape=(5, 5, 5), name="decoded")
        ])
        return model


if __name__ == "__main__":
    patch_dir = "/home/matthew-lee/Data/ADNI/clean/patches/"
    train_gen = utils.patch_generator(patch_dir, "train")
    test_gen = utils.patch_generator(patch_dir, "test")
    val_gen = utils.patch_generator(patch_dir, "val")

    autoencoder = Autoencoder()
    model = autoencoder.model
    model.compile(optimizer=tf.train.MomentumOptimizer(0.02, 0.1), loss="mse")
    model.fit(train_gen, epochs=20, steps_per_epoch=2000)
    autoencoder.save_model(name="autoencoder")

    import matplotlib.pyplot as plt
    from display import display_comparison_batch

    input_imgs = test_gen.__next__()[0]
    output = model.predict(input_imgs)
    print(input_imgs.shape)
    print(output.shape)

    display_comparison_batch(input_imgs[:4, :, :, 3], output[:4, :, :, 3])
    plt.show()
