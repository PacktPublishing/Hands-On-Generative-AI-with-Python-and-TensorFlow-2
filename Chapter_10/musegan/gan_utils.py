from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Lambda
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Concatenate, Conv3D
from tensorflow.keras.layers import Layer, Activation

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from functools import partial

import os
import numpy as np

from music21 import note, stream, duration, tempo


def set_trainable(m, val):
    m.trainable = val
    for l in m.layers:
        l.trainable = val


def keras_gradient(y, x):
    deriv = Lambda(lambda z: K.gradients(z[0], z[1]), output_shape=[1])([y, x])
    return deriv


class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'batch_size': self.batch_size
        })
        return config


def gradient_penalty_loss(y_true, y_pred, interpolated_samples):
    """
    calculate gradient penalty loss
    """
    gradients = keras_gradient(y_pred, interpolated_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)

    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def wasserstein(y_true, y_pred):
    return -K.mean(y_true * y_pred)


def conv_3d(incoming_layer,
            num_filters,
            kernel_size,
            stride, padding,
            weight_init):
    incoming_layer = Conv3D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=stride,
        kernel_initializer=weight_init
    )(incoming_layer)
    incoming_layer = LeakyReLU()(incoming_layer)
    return incoming_layer


def build_critic(input_dim, weight_init, n_bars):

    critic_input = Input(shape=input_dim, name='critic_input')

    x = critic_input

    x = conv_3d(x,
                num_filters=128,
                kernel_size=(2, 1, 1),
                stride=(1, 1, 1),
                padding='valid',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=64,
                kernel_size=(n_bars - 1, 1, 1),
                stride=(1, 1, 1),
                padding='valid',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=64,
                kernel_size=(1, 1, 12),
                stride=(1, 1, 12),
                padding='same',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=64,
                kernel_size=(1, 1, 7),
                stride=(1, 1, 7),
                padding='same',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=64,
                kernel_size=(1, 2, 1),
                stride=(1, 2, 1),
                padding='same',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=64,
                kernel_size=(1, 2, 1),
                stride=(1, 2, 1),
                padding='same',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=128,
                kernel_size=(1, 4, 1),
                stride=(1, 2, 1),
                padding='same',
                weight_init=weight_init)

    x = conv_3d(x,
                num_filters=256,
                kernel_size=(1, 3, 1),
                stride=(1, 2, 1),
                padding='same',
                weight_init=weight_init)

    x = Flatten()(x)

    x = Dense(512, kernel_initializer=weight_init)(x)
    x = LeakyReLU()(x)

    critic_output = Dense(1,
                          activation=None,
                          kernel_initializer=weight_init)(x)

    critic = Model(critic_input, critic_output)
    return critic


def build_temporal_network(z_dim, n_bars, weight_init):

    input_layer = Input(shape=(z_dim,), name='temporal_input')
    x = Reshape([1, 1, z_dim])(input_layer)

    x = Conv2DTranspose(
        filters=512,
        kernel_size=(2, 1),
        padding='valid',
        strides=(1, 1),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(
        filters=z_dim,
        kernel_size=(n_bars - 1, 1),
        padding='valid',
        strides=(1, 1),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    output_layer = Reshape([n_bars, z_dim])(x)

    return Model(input_layer, output_layer)


def build_bar_generator(z_dim, n_steps_per_bar, n_pitches, weight_init):

    input_layer = Input(shape=(z_dim * 4,), name='bar_generator_input')

    x = Dense(1024)(input_layer)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Reshape([2, 1, 512])(x)

    x = Conv2DTranspose(
        filters=512,
        kernel_size=(2, 1),
        padding='same',
        strides=(2, 1),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(
        filters=256,
        kernel_size=(2, 1),
        padding='same',
        strides=(2, 1),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(
        filters=256,
        kernel_size=(2, 1),
        padding='same',
        strides=(2, 1),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(
        filters=256,
        kernel_size=(1, 7),
        padding='same',
        strides=(1, 7),
        kernel_initializer=weight_init
    )(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(
        filters=1,
        kernel_size=(1, 12),
        padding='same',
        strides=(1, 12),
        kernel_initializer=weight_init
    )(x)
    x = Activation('tanh')(x)

    output_layer = Reshape([1, n_steps_per_bar, n_pitches, 1])(x)

    return Model(input_layer, output_layer)


def build_generator(z_dim,
                    n_tracks,
                    n_bars,
                    n_steps_per_bar,
                    n_pitches,
                    weight_init):

    track1_input = Input(shape=(z_dim,), name='track1_input')
    track2_input = Input(shape=(z_dim,), name='track2_input')
    track3_input = Input(shape=(n_tracks, z_dim), name='track3_input')
    track4_input = Input(shape=(n_tracks, z_dim), name='track4_input')

    # track1 temporal network
    track1_temp_network = build_temporal_network(z_dim, n_bars, weight_init)
    track1_over_time = track1_temp_network(track1_input)

    # track3 temporal network
    track3_over_time = [None] * n_tracks
    track3_temp_network = [None] * n_tracks
    for track in range(n_tracks):
        track3_temp_network[track] = build_temporal_network(z_dim,
                                                            n_bars,
                                                            weight_init)
        melody_track = Lambda(lambda x: x[:, track, :])(track3_input)
        track3_over_time[track] = track3_temp_network[track](melody_track)

    # bar generator for each track
    bar_gen = [None] * n_tracks
    for track in range(n_tracks):
        bar_gen[track] = build_bar_generator(z_dim,
                                             n_steps_per_bar,
                                             n_pitches,
                                             weight_init)

    # output for each track-bar
    bars_output = [None] * n_bars
    for bar in range(n_bars):
        track_output = [None] * n_tracks

        t1 = Lambda(lambda x: x[:, bar, ],
                    name='track1_input_bar_' + str(bar))(track1_over_time)
        t2 = track2_input

        for track in range(n_tracks):

            t3 = Lambda(lambda x: x[:, bar, :])(track3_over_time[track])
            t4 = Lambda(lambda x: x[:, track, :])(track4_input)

            z_input = Concatenate(axis=1,
                                  name='total_input_bar_{}_track_{}'.
                                  format(bar, track))([t1, t2, t3, t4])

            track_output[track] = bar_gen[track](z_input)

        bars_output[bar] = Concatenate(axis=-1)(track_output)

    generator_output = Concatenate(axis=1, name='concat_bars')(bars_output)

    generator = Model([track1_input, track2_input, track3_input, track4_input],
                      generator_output)
    return generator


def build_gan(generator,
              critic,
              input_dim,
              z_dim,
              n_tracks,
              batch_size,
              grad_weight):

    # Freeze generator layers for critic training
    set_trainable(generator, False)

    # Image input (real sample)
    real_img = Input(shape=input_dim)

    # Fake image
    track1_input = Input(shape=(z_dim,), name='track1_input')
    track2_input = Input(shape=(z_dim,), name='track2_input')
    track3_input = Input(shape=(n_tracks, z_dim), name='track3_input')
    track4_input = Input(shape=(n_tracks, z_dim), name='track4_input')

    fake_img = generator([track1_input,
                          track2_input,
                          track3_input,
                          track4_input])

    # critic determines validity of the real and fake images
    fake = critic(fake_img)
    valid = critic(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage(batch_size)([real_img, fake_img])
    validity_interpolated = critic(interpolated_img)

    partial_gp_loss = partial(gradient_penalty_loss,
                              interpolated_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model(inputs=[real_img,
                                 track1_input,
                                 track2_input,
                                 track3_input,
                                 track4_input],
                         outputs=[valid,
                                  fake,
                                  validity_interpolated])

    critic_model.compile(
        loss=[wasserstein,
              wasserstein,
              partial_gp_loss],
        optimizer=Adam(lr=0.001, beta_1=0.5, beta_2=0.9),
        loss_weights=[1, 1, grad_weight]
    )

    # Freeze critic layers for generator training
    set_trainable(critic, False)
    set_trainable(generator, True)

    # Sampled noise for input to generator
    track1_input = Input(shape=(z_dim,), name='track1_input')
    track2_input = Input(shape=(z_dim,), name='track2_input')
    track3_input = Input(shape=(n_tracks, z_dim), name='track3_input')
    track4_input = Input(shape=(n_tracks, z_dim), name='track4_input')

    # Generate images based of noise
    img = generator([track1_input, track2_input, track3_input, track4_input])
    model_output = critic(img)

    # define gan
    gan = Model([track1_input,
                 track2_input,
                 track3_input,
                 track4_input], model_output)

    gan.compile(optimizer=Adam(lr=0.001, beta_1=0.5, beta_2=0.9),
                loss=wasserstein)

    # reset critic layers
    set_trainable(critic, True)

    return critic_model, gan


def train_critic(x_train,
                 critic_model,
                 z_dim,
                 batch_size,
                 n_tracks,
                 use_gen):

    valid = np.ones((batch_size, 1), dtype=np.float32)
    fake = -np.ones((batch_size, 1), dtype=np.float32)
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    if use_gen:
        true_imgs = next(x_train)[0]
        if true_imgs.shape[0] != batch_size:
            true_imgs = next(x_train)[0]
    else:
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]

    track1_noise = np.random.normal(0, 1, (batch_size, z_dim))
    track2_noise = np.random.normal(0, 1, (batch_size, z_dim))
    track3_noise = np.random.normal(0, 1, (batch_size, n_tracks, z_dim))
    track4_noise = np.random.normal(0, 1, (batch_size, n_tracks, z_dim))

    d_loss = critic_model.train_on_batch([true_imgs,
                                          track1_noise,
                                          track2_noise,
                                          track3_noise,
                                          track4_noise], [valid, fake, dummy])
    return d_loss


def train_generator(gan_model, z_dim, n_tracks, batch_size):
    valid = np.ones((batch_size, 1), dtype=np.float32)

    track1_noise = np.random.normal(0, 1, (batch_size, z_dim))
    track2_noise = np.random.normal(0, 1, (batch_size, z_dim))
    track3_noise = np.random.normal(0, 1, (batch_size, n_tracks, z_dim))
    track4_noise = np.random.normal(0, 1, (batch_size, n_tracks, z_dim))

    return gan_model.train_on_batch([track1_noise,
                                     track2_noise,
                                     track3_noise,
                                     track4_noise], valid)


def argmax_output(output):
    max_pitches = np.argmax(output, axis=3)
    return max_pitches


def notes_to_midi(n_bars,
                  n_steps_per_bar,
                  n_tracks,
                  epoch,
                  output_folder,
                  output):

    for score_num in range(len(output)):

        max_pitches = argmax_output(output)

        midi_note_score = max_pitches[score_num].reshape([n_bars *
                                                          n_steps_per_bar,
                                                          n_tracks])
        parts = stream.Score()
        parts.append(tempo.MetronomeMark(number=66))

        for i in range(n_tracks):
            last_x = int(midi_note_score[:, i][0])
            s = stream.Part()
            dur = 0

            for idx, x in enumerate(midi_note_score[:, i]):
                x = int(x)

                if (x != last_x or idx % 4 == 0) and idx > 0:
                    n = note.Note(last_x)
                    n.duration = duration.Duration(dur)
                    s.append(n)
                    dur = 0

                last_x = x
                dur = dur + 0.25

            n = note.Note(last_x)
            n.duration = duration.Duration(dur)
            s.append(n)

            parts.append(s)

        parts.write('midi', fp=os.path.join(output_folder,
                                            "sample_{}_{}.midi".
                                            format(epoch, score_num)))


def sample_predictions(output_folder,
                       z_dim,
                       n_tracks,
                       n_bars,
                       n_steps_per_bar,
                       epoch,
                       generator):
    r = 5
    track1_noise = np.random.normal(0, 1, (r, z_dim))
    track2_noise = np.random.normal(0, 1, (r, z_dim))
    track3_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))
    track4_noise = np.random.normal(0, 1, (r, n_tracks, z_dim))

    gen_scores = generator.predict([track1_noise,
                                    track2_noise,
                                    track3_noise,
                                    track4_noise])

    notes_to_midi(n_bars,
                  n_steps_per_bar,
                  n_tracks,
                  epoch,
                  output_folder,
                  gen_scores)


# def save(folder,
#          input_dim,
#          critic_learning_rate,
#          generator_learning_rate,
#          optimiser,
#          grad_weight,
#          z_dim,
#          batch_size,
#          n_tracks,
#          n_bars,
#          n_steps_per_bar,
#          n_pitches):
#
#     with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
#         pickle.dump([
#             input_dim,
#             critic_learning_rate,
#             generator_learning_rate,
#             optimiser,
#             grad_weight,
#             z_dim,
#             batch_size,
#             n_tracks,
#             n_bars,
#             n_steps_per_bar,
#             n_pitches
#         ], f)


def save_models(gan_model, critic_model, generator_model, output_folder):
    gan_model.save_weights(os.path.join(output_folder, 'model.h5'))
    critic_model.save_weights(os.path.join(output_folder, 'critic.h5'))
    generator_model.save_weights(os.path.join(output_folder, 'generator.h5'))


def train_musegan(x_train,
                  critic_model,
                  gen_model,
                  gan_model,
                  z_dim,
                  n_tracks,
                  n_bars,
                  n_steps_per_bar,
                  batch_size,
                  epochs,
                  output_folder,
                  print_every_n_batches=10,
                  n_critic=5,
                  use_gen=False):

    d_losses = []
    g_losses = []
    for epoch in range(epochs):
        for _ in range(n_critic):
            d_loss = train_critic(x_train,
                                  critic_model,
                                  z_dim,
                                  batch_size,
                                  n_tracks,
                                  use_gen)

        g_loss = train_generator(gan_model, z_dim, n_tracks, batch_size)

        print("Epoch=%d [D loss: (%.1f)(Real=%.1f,Fake=%.1f, Grad.Penalty=%.1f)] [Gen loss: %.1f]" % (epoch,
                                                                                                      d_loss[0],
                                                                                                      d_loss[1],
                                                                                                      d_loss[2],
                                                                                                      d_loss[3],
                                                                                                      g_loss))

        d_losses.append(d_loss)
        g_losses.append(g_loss)

        # save progress
        if epoch % print_every_n_batches == 0:
            sample_predictions(output_folder,
                               z_dim,
                               n_tracks,
                               n_bars,
                               n_steps_per_bar,
                               epoch,
                               gen_model)

            # save_models(gan_model, critic_model, gen_model, output_folder)
    return d_losses, g_losses


def load_music(data_path, n_bars, n_steps_per_bar):
    filename = data_path

    with np.load(filename, encoding='bytes', allow_pickle=True) as nf:
        data = nf['train']

    data_ints = []
    for x in data:
        counter = 0
        cont = True
        while cont:
            if not np.any(np.isnan(x[counter:(counter+4)])):
                cont = False
            else:
                counter += 4

        if n_bars * n_steps_per_bar < x.shape[0]:
            data_ints.append(x[counter:(counter + (n_bars *
                                                   n_steps_per_bar)), :])

    data_ints = np.array(data_ints)

    n_songs = data_ints.shape[0]
    n_tracks = data_ints.shape[2]

    data_ints = data_ints.reshape([n_songs, n_bars, n_steps_per_bar, n_tracks])

    max_note = 83

    where_are_NaNs = np.isnan(data_ints)
    data_ints[where_are_NaNs] = max_note + 1
    max_note = max_note + 1

    data_ints = data_ints.astype(int)

    num_classes = max_note + 1

    encoded_scores = np.eye(num_classes)[data_ints]
    encoded_scores[encoded_scores == 0] = -1
    encoded_scores = np.delete(encoded_scores, max_note, -1)

    encoded_scores = encoded_scores.transpose([0, 1, 2, 4, 3])

    return encoded_scores
