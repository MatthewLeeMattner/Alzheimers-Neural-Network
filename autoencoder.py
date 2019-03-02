import tensorflow as tf
import numpy as np
from time import time
from data_helper import get_dummy_patches
from model_helper import ModelHelper
# tf.enable_eager_execution()

train, test, val = get_dummy_patches()


class Autoencoder(ModelHelper):

    def __init__(self, load=False, name=None, location=None):
        self.model = self.model_define()
        self.set_model(self.model)
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.02), loss="mse")
        self.model.predict(np.random.randn(1, 5, 5, 5))
        if load is True:
            if location is not None:
                self.load_model(name, location)
            else:
                self.load_model(name)

    def custom_sparse_activation_regularization(self, activations):
        return tf.multiply(tf.divide(1, tf.size(activations)), tf.reduce_sum(activations))

    def model_define(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(125,)),
            tf.keras.layers.Dense(units=150, activation="sigmoid", name="encoded",
                                  activity_regularizer=self.custom_sparse_activation_regularization),  # Encoded representation
            tf.keras.layers.Dense(units=125),
            tf.keras.layers.Reshape(target_shape=(5, 5, 5), name="decoded")
        ])
        return model

    def save_model(self, location="models"):
        curr_time = time()
        print("Saving model {}/{}-autoencoder".format(location, curr_time))
        tf.keras.models.save_model(
            self.model,
            "{}/{}-autoencoder".format(location, curr_time),
            overwrite=False
        )

    def load_model(self, name, location="models", compile=True):
        if location is None:
            raise ValueError("Location of model cannot be None")
        if name is None:
            raise ValueError("Name of model cannot be None")
        self.model.load_weights("{}/{}".format(location, name))
        return self.model


if __name__ == "__main__":
    autoencoder = Autoencoder()
    model = autoencoder.model
    model.compile(optimizer=tf.train.MomentumOptimizer(0.02, 0.1), loss="mse")
    model.fit(train, train, epochs=10)
    autoencoder.save_model()

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
