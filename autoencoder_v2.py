import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
from display import display_comparison_batch
from model_helper import ModelHelper
import utils


def input_shape(shape, trainable):
    inp = k.layers.Conv3D(
        filters=32,
        kernel_size=(5, 5, 5),
        input_shape=shape,
        name="input",
        trainable=trainable,
        kernel_regularizer=k.regularizers.l2(),
        activation="elu"
    )
    return inp


def create_model(shape, trainable=True):
    model = k.Sequential()
    model.add(input_shape(shape, trainable))
    layers = [
        k.layers.Dropout(0.5),
        k.layers.AveragePooling3D((2, 2, 2), 2),
        k.layers.Conv3D(
            filters=64,
            kernel_size=(3, 3, 3),
            trainable=trainable,
            kernel_regularizer=k.regularizers.l2(),
            activation="elu"
        ),
        k.layers.Dropout(0.5),
        k.layers.AveragePooling3D((2, 2, 2), 2, name="encoding"),
        k.layers.Conv3DTranspose(
            filters=32,
            kernel_size=(6, 6, 6),
            trainable=trainable
        ),
        k.layers.Conv3DTranspose(
            filters=1,
            kernel_size=(7, 7, 7),
            trainable=trainable
        )
    ]
    for layer in layers:
        model.add(layer)
    return model


if __name__ == "__main__":
    PATCH_DIR = "/home/matthew-lee/Data/ADNI/clean/patches/"
    TRAIN_GEN = utils.patch_generator(PATCH_DIR, "train", 32)
    TEST_GEN = utils.patch_generator(PATCH_DIR, "test")


    model = create_model((12, 12, 12, 1))

    model.compile(
        optimizer=tf.train.AdamOptimizer(0.0002),
        loss="mse"
    )
    model.predict(np.random.randn(1, 12, 12, 12, 1))
    model.summary()

    model.fit(
        TRAIN_GEN,
        epochs=5,
        steps_per_epoch=5000,
        workers=6,
        use_multiprocessing=True
    )

    input_imgs = TEST_GEN.__next__()[0]
    output = model.predict(input_imgs)
    input_imgs = np.squeeze(input_imgs, axis=4)
    output = np.squeeze(output, axis=4)

    display_comparison_batch(input_imgs[4:9, :, :, 3], output[4:9, :, :, 3])
    plt.show()

    helper = ModelHelper()
    helper.set_model(model)
    helper.save_model("autoencoder")
    display_comparison_batch(input_imgs[4:9, :, :, 3], output[4:9, :, :, 3])
    plt.show()
