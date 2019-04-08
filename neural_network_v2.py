import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import utils

tf.reset_default_graph()

BATCH_DIR = "/home/matthew-lee/Data/ADNI/clean/batches/"
TRAIN_GEN = utils.batch_generator(BATCH_DIR, "train")
TEST_GEN = utils.batch_generator(BATCH_DIR, "test")
VAL_GEN = utils.batch_generator(BATCH_DIR, "val")


model = k.Sequential([
    k.layers.Conv3D(
        filters=150,
        kernel_size=(5, 5, 5),
        input_shape=(142, 179, 152, 1)
    ),
    k.layers.MaxPool3D(pool_size=(3, 3, 3)),
    k.layers.Dropout(0.5),
    k.layers.Conv3D(
        filters=64,
        kernel_size=(5, 5, 5)
    ),
    k.layers.MaxPool3D(pool_size=(3, 3, 3)),
    k.layers.Dropout(0.5),
    k.layers.Conv3D(
        filters=8,
        kernel_size=(3, 3, 3)
    ),
    k.layers.MaxPool3D(pool_size=(2, 2, 2)),
    k.layers.Flatten(),
    k.layers.Dense(
        units=800,
        activation=tf.nn.relu,
        kernel_regularizer=k.regularizers.l2(0.001)
    ),
    k.layers.Dense(units=3, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(0.002),
    loss="mse",
    metrics=['accuracy']
)
model.predict(np.random.randn(1, 142, 179, 152, 1))
model.summary()

model.fit(TRAIN_GEN, epochs=10, steps_per_epoch=100)
