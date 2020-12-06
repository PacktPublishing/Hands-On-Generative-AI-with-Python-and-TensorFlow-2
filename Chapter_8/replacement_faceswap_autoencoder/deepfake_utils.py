from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense,
                                     Flatten, Reshape, LeakyReLU, Conv2D)
from tensorflow.keras.layers import UpSampling2D


def conv(x, filters):
    x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    return x


def upscale(x, filters):
    x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D()(x)
    return x


def Encoder(input_shape, encoder_dim):
    input_ = Input(shape=input_shape)
    x = input_
    x = conv(x, 128)
    x = conv(x, 256)
    x = conv(x, 512)
    x = conv(x, 1024)
    x = Dense(encoder_dim)(Flatten()(x))
    x = Dense(4 * 4 * 1024)(x)
    # Passed flattened X input into 2 dense layers, 1024 and 1024*4*4
    x = Reshape((4, 4, 1024))(x)
    # Reshapes X into 4,4,1024
    x = upscale(x, 128)
    return Model(input_, x)


def Decoder(input_shape=(8, 8, 512)):
    input_ = Input(shape=input_shape)
    x = input_
    x = upscale(x, 256)
    x = upscale(x, 128)
    x = upscale(x, 64)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)
