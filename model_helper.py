class ModelHelper():

    def __init__(self, model):
        self.model = model

    def get_layer(self, name):
        for layer in self.model.layers:
            if layer._name == name:
                return layer

    def get_layer_kernel(self, name):
        layer = self.get_layer(name)
        return layer.kernel
