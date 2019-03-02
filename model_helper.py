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

if __name__ == "__main__":
    from autoencoder import Autoencoder
    model = Autoencoder(load=True, name="1551066199.059682-autoencoder")
    print(model.get_weights("encoded")[1].shape)
