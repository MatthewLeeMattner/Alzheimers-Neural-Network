import tensorflow as tf
import numpy as np
from autoencoder import Autoencoder
from model_helper import ModelHelper
import utils


class NeuralNetwork(ModelHelper):
    '''
    Class to contain functions required to setup,
    train and run the fully connected part of the neural network structure.
    Extends model helper for helper functions
    '''

    def __init__(self, encoded_weights=None, neural_network_name=None):
        '''
        Sets up the model and compiles it. Either the encoded weights are
        required for new setup of the model or the neural network name to load
        a previously trained model.

        If both are supplied a value error is raised as the pre-trained model
        will already have had the encoded weights applied

        @param encoded_weights: A two-tuple containing the
            kernel and bias weights from the autoencoder
        @param neural_network_name: The name of the model to load
        '''
        if encoded_weights is None and neural_network_name is None:
            raise ValueError("Either encoded weights or neural "
                             "network name must have a value")
        self.model, self.conv_layer = self.model_define()
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.02), loss="mse")
        self.model.predict(np.random.randn(1, 81, 97, 81, 1))
        if neural_network_name is not None:
            self.load_model(neural_network_name)
        else:
            self.assign_encoded(encoded_weights)

    def model_define(self):
        '''
        Defines the model structure
        returns: model object and convolutional layer
        (to apply encoded weights to later)
        '''
        model = tf.keras.Sequential()
        conv_layer = tf.keras.layers.Conv3D(
            filters=150,
            kernel_size=(5, 5, 5),
            input_shape=(81, 97, 81, 1),
            name="embedded_layer",
            trainable=False
        )
        model.add(conv_layer)
        model.add(tf.keras.layers.MaxPool3D(pool_size=(5, 5, 5)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=800))
        model.add(tf.keras.layers.Dense(units=3))
        return model, conv_layer

    def assign_encoded(self, encoded_weights):
        '''
        Applies the encoded weights from the autoencoder to the
        convolutional layer

        @param encoded_weights: A two-tuple containing the kernel
            and bias weights from the autoencoder
        return convolution layer that weights are applied to
        '''
        weights, bias = encoded_weights
        weights_reshape = np.reshape(weights, (5, 5, 5, 1, 150))
        self.conv_layer.set_weights([weights_reshape, bias])
        return self.conv_layer


if __name__ == "__main__":
    auto_model = Autoencoder(name="1553563429.1773698-autoencoder",
                             location="./models")
    encoded_values = auto_model.get_weights("encoded")
    print(encoded_values[0].shape)
    print(encoded_values[1].shape)

    batch_dir = "/home/matthew-lee/Data/ADNI/clean/batches/"
    train_gen = utils.batch_generator(batch_dir, "train")
    test_gen = utils.batch_generator(batch_dir, "test")

    neural_network = NeuralNetwork(encoded_values)
    model = neural_network.model
    model.compile(
        tf.train.AdamOptimizer(),
        loss="mse",
        metrics=["accuracy"]
    )
    model.fit(train_gen, epochs=20, steps_per_epoch=124)
