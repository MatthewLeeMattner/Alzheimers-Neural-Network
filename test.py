import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras as k
from display import display_comparison_batch
from autoencoder_v2 import create_model
from model_helper import ModelHelper
import utils


def create_full_model(autoencoder, shape):
    model = create_model(shape, trainable=False)
    helper = ModelHelper()
    helper.set_model(model)
    helper.load_model(autoencoder)
    helper.pop()
    helper.pop()
    helper.add(
        k.layers.Conv3D(
            filters=64,
            kernel_size=(5, 5, 5),
            kernel_regularizer=k.regularizers.l2(),
            activation="relu"
        )
    )
    helper.add(k.layers.Dropout(0.5))
    helper.add(
        k.layers.AveragePooling3D(
            (2, 2, 2),
            2
        )
    )
    helper.add(
        k.layers.Conv3D(
            filters=32,
            kernel_size=(5, 5, 5),
            kernel_regularizer=k.regularizers.l2(),
            activation="relu"
        )
    )
    helper.add(k.layers.Dropout(0.5))
    helper.add(
        k.layers.AveragePooling3D(
            (2, 2, 2),
            2
        )
    )
    helper.add(k.layers.Flatten())
    helper.add(
        k.layers.Dense(
            units=500,
            activation="tanh"
        )
    )
    helper.add(k.layers.Dropout(0.5))
    helper.add(k.layers.Dense(units=3))

    model = helper.model
    return model


if __name__ == "__main__":
    autoencoder_loc = "1554700196.69537-autoencoder"
    shape = np.expand_dims(
        np.load(
            "/home/matthew-lee/Data/ADNI/clean/batches/test_0_x.npy"
        ),
        axis=4
    ).shape[1:]

    model = create_full_model(autoencoder_loc, shape)
    model.summary()


    #class_weights = np.load(
    #    "/home/matthew-lee/Data/ADNI/clean/train_weightings.npy"
    #)



    batch_loc = "/home/matthew-lee/Data/ADNI/clean/batches/"
    train_gen = utils.batch_generator(batch_loc, "train")
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.00002),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    class_weights = {
        0: 0.075,
        1: 0.025,
        2: 0.9
    }

    model.fit(
        train_gen,
        epochs=20,
        steps_per_epoch=300,
        workers=1,
        use_multiprocessing=False,
        class_weight=class_weights
    )
