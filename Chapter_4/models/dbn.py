import tensorflow as tf

from . import rbm as rbm


class DBN(tf.keras.Model):

    def __init__(self, rbm_params=None, name='deep_belief_network',
                 num_epochs=100, tolerance=1e-3, batch_size=32,
                 shuffle_buffer=1024, **kwargs):
        super().__init__(name=name, **kwargs)
        self._rbm_params = rbm_params
        self._rbm_layers = list()
        self._dense_layers = list()
        for num, rbm_param in enumerate(rbm_params):
            self._rbm_layers.append(rbm.RBM(**rbm_param))
            self._rbm_layers[-1].build([rbm_param["number_visible_units"]])
            if num < len(rbm_params)-1:
                self._dense_layers.append(
                    tf.keras.layers.Dense(
                                          rbm_param["number_hidden_units"],
                                          activation=tf.nn.sigmoid)
                )
            else:
                self._dense_layers.append(
                    tf.keras.layers.Dense(
                                          rbm_param["number_hidden_units"],
                                          activation=tf.nn.softmax)
                )
            self._dense_layers[-1].build([rbm_param["number_visible_units"]])
        self._num_epochs = num_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._shuffle_buffer = shuffle_buffer

    def call(self, x, training):
        for dense_layer in self._dense_layers:
            x = dense_layer(x)
        return x

    def train_rbm(self, rbm, inputs):

        last_cost = None

        for epoch in range(self._num_epochs):
            cost = 0.0
            count = 0.0
            for datapoints in inputs.shuffle(self._shuffle_buffer).batch(
                self._batch_size):
                cost += rbm.cd_update(datapoints)
                count += 1.0
            cost /= count
            print("epoch: {}, cost: {}".format(epoch, cost))
            if last_cost and abs(last_cost-cost) <= self._tolerance:
                break
            last_cost = cost

        return rbm

    def train_dbn(self, inputs):

        # pretraining:
        
        inputs_layers = []
        for num in range(len(self._rbm_layers)):

            if num == 0:
                inputs_layers.append(inputs)
                self._rbm_layers[num] = \
                    self.train_rbm(self._rbm_layers[num],
                                   inputs)
            else:  # pass all data through previous layer
                inputs_layers.append(inputs_layers[num-1].map(
                    self._rbm_layers[num-1].forward))
                self._rbm_layers[num] = \
                    self.train_rbm(self._rbm_layers[num],
                                   inputs_layers[num])

        # wake-sleep:

        for epoch in range(self._num_epochs):

            # wake pass
            inputs_layers = []
            for num in range(len(self._rbm_layers)):

                if num == 0:
                    inputs_layers.append(inputs)
                else:
                    inputs_layers.append(inputs_layers[num-1].map(
                        self._rbm_layers[num-1].forward))

            for num in range(len(self._rbm_layers)-1):
                cost = 0.0
                count = 0.0
                for datapoints in \
                    inputs_layers[num].shuffle(
                        self._shuffle_buffer).batch(self._batch_size):
                    cost += self._rbm_layers[num].wake_update(datapoints)
                    count += 1.0
                cost /= count
                print("epoch: {}, wake_cost: {}".format(epoch, cost))

            # top-level associative:
            self._rbm_layers[-1] = \
                self.train_rbm(self._rbm_layers[-1],
                               inputs_layers[-2].map(
                                   self._rbm_layers[-2].forward),
                               num_epochs=self._num_epochs,
                               tolerance=self._tolerance,
                               batch_size=self._batch_size,
                               shuffle_buffer=self._shuffle_buffer)

            reverse_inputs = inputs_layers[-1].map(
                self._rbm_layers[-1].forward)

            # sleep pass

            reverse_inputs_layers = []
            for num in range(len(self._rbm_layers)):

                if num == 0:
                    reverse_inputs_layers.append(reverse_inputs)
                else:
                    reverse_inputs_layers.append(
                        reverse_inputs_layers[num-1].map(
                            self._rbm_layers[
                                len(self._rbm_layers)-num].reverse))

            for num in range(len(self._rbm_layers)):
                if num > 0:
                    cost = 0.0
                    count = 0.0
                    for datapoints in \
                        reverse_inputs_layers[num].shuffle(
                            self._shuffle_buffer).batch(self._batch_size):
                        cost += self._rbm_layers[
                            len(self._rbm_layers)-1-num]\
                                .sleep_update(datapoints)
                        count += 1.0
                    cost /= count
                    print("epoch: {}, sleep_cost: {}".format(epoch, cost))

        for dense_layer, rbm_layer in zip(self._dense_layers,
                                          self._rbm_layers):
            dense_layer.set_weights([rbm_layer.w_rec, rbm_layer.hb])
