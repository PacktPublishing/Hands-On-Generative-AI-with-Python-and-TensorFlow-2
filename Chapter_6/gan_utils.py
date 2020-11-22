import numpy as np
from matplotlib import pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense,  Input, Conv2D, Activation, multiply
from tensorflow.keras.layers import UpSampling2D, ZeroPadding2D, Dropout, Embedding
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Reshape


def wasserstein_loss(y_true, y_pred):
    """
    Custom loss function for W-GAN
    Parameters:
        y_true: type:np.array. Ground truth
        y_pred: type:np.array. Predicted score
    """
    return K.mean(y_true * y_pred)


def build_discriminator(input_shape=(28, 28,), verbose=True):
    """
    Utility method to build a MLP discriminator
    Parameters:
        input_shape:    type:tuple. Shape of input image for classification.
                        Default shape is (28,28)->MNIST
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    if verbose:
        model.summary()

    return model


def build_dc_discriminator(input_shape=(28, 28, 1), verbose=True):
    """
    Utility method to build a DCGAN discriminator
    Parameters:
        input_shape:    type:tuple. Shape of input image for classification.
                        Default shape is (28,28,1)->MNIST
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, strides=2,
                     input_shape=input_shape,
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    if verbose:
        model.summary()
    return model


def build_conditional_discriminator(input_shape=(28, 28, 1),
                                    num_classes=10, verbose=True):
    """
    Utility method to build a conditional MLP discriminator
    Parameters:
        input_shape:    type:tuple. Shape of input image for classification.
                        Default shape is (28,28)->MNIST
        num_classes:    type:int. Number of unique class labels.
                        Default is 10->MNIST digits
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """

    img = Input(shape=input_shape)
    flat_img = Flatten()(img)

    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes,
                                          np.prod(input_shape))(label))

    model_input = multiply([flat_img, label_embedding])

    mlp = Dense(512, input_dim=np.prod(input_shape))(model_input)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dropout(0.4)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dropout(0.4)(mlp)
    mlp = Dense(1, activation='sigmoid')(mlp)

    model = Model([img, label], mlp)

    if verbose:
        model.summary()

    return model


def build_critic(input_shape=(28, 28, 1), verbose=True):
    """
    Utility method to build a MLP critic for W-GAN
    Parameters:
        input_shape:    type:tuple. Shape of input image for classification.
                        Default shape is (28,28, 1)->MNIST
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    if verbose:
        model.summary()
    return model


def build_generator(z_dim=100, output_shape=(28, 28), verbose=True):
    """
    Utility method to build a MLP generator
    Parameters:
        z_dim:          type:int(positive). Size of input noise vector to be
                        used as model input.
                        Default value is 100
        output_shape:   type:tuple. Shape of output image .
                        Default shape is (28,28)->MNIST
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=(z_dim,)))
    model.add(Dense(256, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))

    if verbose:
        model.summary()
    return model


def build_dc_generator(z_dim=100, verbose=True):
    """
    Utility method to build a DCGAN generator
    Parameters:
        z_dim:          type:int(positive). Size of input noise vector to be
                        used as model input.
                        Default value is 100
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    model = Sequential()
    model.add(Input(shape=(z_dim,)))
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=z_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    if verbose:
        model.summary()

    return model


def build_conditional_generator(z_dim=100, output_shape=(28, 28, 1),
                                num_classes=10, verbose=True):
    """
    Utility method to build a MLP generator
    Parameters:
        z_dim:          type:int(positive). Size of input noise vector to be
                        used as model input.
                        Default value is 100
        output_shape:   type:tuple. Shape of output image .
                        Default shape is (28,28)->MNIST
        num_classes:    type:int. Number of unique class labels.
                        Default is 10->MNIST digits
        verbose:        type:boolean. Print model summary if set to true.
                        Default is True
    Returns:
        tensorflow.keras.model object
    """
    noise = Input(shape=(z_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, z_dim)(label))
    model_input = multiply([noise, label_embedding])

    mlp = Dense(256, input_dim=z_dim)(model_input)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = BatchNormalization(momentum=0.8)(mlp)
    mlp = Dense(512)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = Dense(1024)(mlp)
    mlp = LeakyReLU(alpha=0.2)(mlp)
    mlp = BatchNormalization(momentum=0.8)(mlp)
    mlp = Dense(np.prod(output_shape), activation='tanh')(mlp)
    mlp = Reshape(output_shape)(mlp)

    model = Model([noise, label], mlp)

    if verbose:
        model.summary()
    return model


def sample_images(epoch, generator, z_dim=100,
                  save_output=True,
                  output_dir="images"):
    """
    Utility method to sample and plot random 25 generator samples
    in a 5x5 grid and save
    Parameters:
        epoch:      type:int. Epoch number
        generator:  type:tensorflow.keras.model. Generator model object
        z_dim:      int(positive). Size of input noise vector.
                    Default is 100
        save_output:type:boolean. Saves plot to disk if true.
                    Default is True
        output_dir: type:str. Directory path to save generated samples.
                    used only if save_output=True.
                    Default value is "images"
    Returns:
        None
    """

    # get label if conditional generator is active
    model_type = 'cgan' if isinstance(generator.input_shape, list) else 'others'

    if model_type == 'cgan':
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, z_dim))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        gen_imgs = generator.predict([noise, sampled_labels])
    else:
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, z_dim))
        gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # get output shape
    output_shape = len(generator.output_shape)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if output_shape == 3:
                axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
            else:
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            if model_type == 'cgan':
                axs[i, j].set_title("Label: %d" % sampled_labels[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()
    if save_output:
        fig.savefig("{}/{}.png".format(output_dir, epoch))
    plt.close()
