from __future__ import absolute_import, division, \
    print_function, unicode_literals

from models import dbn, rbm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def flatten_image(x, label=False):
    if label:
        return (tf.divide(
            tf.dtypes.cast(
                tf.reshape(x["image"], (1, 28*28)), tf.float32), 256.0),
                x["label"])
    else:
        return (
            tf.divide(tf.dtypes.cast(
                tf.reshape(x["image"], (1, 28*28)), tf.float32), 256.0))


def run_dbn():
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()

    mnist_train = mnist_builder.as_dataset(split="train")

    rbm_params = [
        {"number_hidden_units": 500, "number_visible_units": 784},
        {"number_hidden_units": 500, "number_visible_units": 500},
        {"number_hidden_units": 2000, "number_visible_units": 500},
        {"number_hidden_units": 10, "number_visible_units": 2000}
    ]

    deep_belief_network = dbn.DBN(rbm_params, tolerance=1)

    # pre-training and wake-sleep

    deep_belief_network.train_dbn(mnist_train.map(
            lambda x: flatten_image(x, label=False)))

    # backprop

    deep_belief_network.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    deep_belief_network.fit(
        x=mnist_train.map(
            lambda x: flatten_image(x, label=True)).batch(1000), )


if __name__ == "__main__":
    run_dbn()
