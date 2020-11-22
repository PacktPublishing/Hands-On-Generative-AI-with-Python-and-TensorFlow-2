from tensorflow.keras.layers import Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow_addons.layers import InstanceNormalization


def downsample_block(incoming_layer,
                     num_filters,
                     kernel_size=4):
    """
    Downsampling block of layers used by U-Net generator. Block consists of:
    {Conv2D -> LeakyRelu-> InstanceNormalization}
    Parameters:
        incoming_layer:         type:tf.keras.Layer. Layer which will
                                pass its output to this block
        num_filters:            type:int. Number of filters for
                                the 2d Conv layer
        kernel_size:            type:int. Size of the kernel for
                                the 2d Conv layer
    Returns:
        type:tf.keras.Layer. A downsampling block of layers.
    """
    downsample_layer = Conv2D(num_filters,
                              kernel_size=kernel_size,
                              strides=2, padding='same')(incoming_layer)
    downsample_layer = LeakyReLU(alpha=0.2)(downsample_layer)
    downsample_layer = InstanceNormalization()(downsample_layer)
    return downsample_layer


def upsample_block(incoming_layer,
                   skip_input_layer,
                   num_filters,
                   kernel_size=4,
                   dropout_rate=0):
    """
    Upsampling block of layers used by U-Net generator. Block consists of
    {UpSampling2D -> Conv2D -> [Dropout] -> InstanceNorm}+Skip_Connection
    Parameters:
        incoming_layer:         type:tf.keras.Layer. Layer which will
                                pass its output to this block
        skip_input_layer:       type:tf.keras.Layer. Layer which will
                                be the skip connection to this block
        num_filters:            type:int. Number of filters for
                                the 2d Conv layer
        kernel_size:            type:int. Size of the kernel for
                                the 2d Conv layer
        dropout_rate:           type:float. Adds a dropout layer with given
                                rate to the block. No dropout layer is added
                                if set to 0
    Returns:
        type:tf.keras.Layer. A upsampling block of layers.
    """
    upsample_layer = UpSampling2D(size=2)(incoming_layer)
    upsample_layer = Conv2D(num_filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            activation='relu')(upsample_layer)
    if dropout_rate:
        upsample_layer = Dropout(dropout_rate)(upsample_layer)
    upsample_layer = InstanceNormalization()(upsample_layer)
    upsample_layer = Concatenate()([upsample_layer, skip_input_layer])
    return upsample_layer


def discriminator_block(incoming_layer,
                        num_filters,
                        kernel_size=4,
                        instance_normalization=True):
    """
    Block of layers used by patch-GAN discriminator. Block consists of:
    {Conv2D -> LeakyRelu-> [InstanceNorm]}
    Parameters:
        incoming_layer:         type:tf.keras.Layer. Layer which will
                                pass its output to this block
        num_filters:            type:int. Number of filters for
                                the 2d Conv layer
        kernel_size:            type:int. Size of the kernel for
                                the 2d Conv layer
        instance_normalization: type:bool. if set to true, adds instance
                                normalization layer to the block
    Returns:
        type:tf.keras.Layer. A block of layers for discriminator.
    """
    disc_layer = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(incoming_layer)
    disc_layer = LeakyReLU(alpha=0.2)(disc_layer)
    if instance_normalization:
        disc_layer = InstanceNormalization()(disc_layer)
    return disc_layer
