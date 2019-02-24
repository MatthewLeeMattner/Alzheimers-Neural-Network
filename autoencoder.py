import tensorflow as tf
import numpy as np
from time import time
from data_helper import get_dummy_patches
# tf.enable_eager_execution()

train, test, val = get_dummy_patches()


class Autoencoder:

    def __init__(self):
        self.model = self.model_define()

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
        tf.keras.models.save_model(
            self.model,
            "{}/{}-autoencoder".format(location, time()),
            overwrite=False
        )

    def load_model(self, name, location="models", compile=True):
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
