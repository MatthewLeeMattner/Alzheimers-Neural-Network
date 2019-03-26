from time import time
import tensorflow as tf


class ModelHelper():

    def set_model(self, model):
        self.model = model

    def get_layer(self, name):
        for layer in self.model.layers:
            if layer._name == name:
                return layer

    def get_weights(self, name):
        layer = self.get_layer(name)
        return layer.get_weights()

    def load_model(self, name, location="models", compile=True):
        if location is None:
            raise ValueError("Location of model cannot be None")
        if name is None:
            raise ValueError("Name of model cannot be None")
        self.model.load_weights("{}/{}".format(location, name))
        return self.model

    def save_model(self, name, location="models"):
        if name is None:
            raise ValueError("No name provided to save model")
        curr_time = time()
        print("Saving model {}/{}-{}".format(location, curr_time, name))
        tf.keras.models.save_model(
            self.model,
            "{}/{}-{}".format(location, curr_time, name),
            overwrite=False
        )


if __name__ == "__main__":
    from autoencoder import Autoencoder
    model = Autoencoder(load=True, name="1551066199.059682-autoencoder")
    print(model.get_weights("encoded")[1].shape)
