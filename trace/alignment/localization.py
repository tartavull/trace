import tensorflow as tf


def get_weight_var_and_hist(name, shape, dtype=tf.float32, trainable=True):
    w = tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                        trainable=trainable)
    hist = tf.summary.histogram(name, w)
    return w, hist


def get_bias_var_and_hist(name, shape, dtype=tf.float32, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    b = tf.get_variable(name, initializer=initial, dtype=dtype, trainable=trainable)
    hist = tf.summary.histogram(name, b)
    return b, hist


class LocalizationNetwork(object):
    def __init__(self, in_dim, params_dim, trainable=True):
        self.params_dim = params_dim
        self.trainable = trainable
        self.in_dim = in_dim

    def __call__(self, ref_slice, off_slice):
        raise NotImplementedError('Abstract class, the inheritor must implement')


# TODO(beisner): add conv3d version
class Conv2DLocalizationNetwork(LocalizationNetwork):
    def __init__(self, in_dim, params_dim, l1_n, l2_n, l3_n, l4_n, fc1_n, trainable=True):
        super(Conv2DLocalizationNetwork, self).__init__(in_dim, params_dim, trainable)

        with tf.variable_scope('conv_localization'):
            with tf.variable_scope('layer1'):
                self.__w11, h_w11 = get_weight_var_and_hist('w11', [1, 3, 3, 1, l1_n], trainable=trainable)
                self.__w12, h_w12 = get_weight_var_and_hist('w12', [1, 3, 3, l1_n, l1_n], trainable=trainable)
                self.__b1, h_b1 = get_bias_var_and_hist('b1', [l1_n], dtype=tf.float32, trainable=trainable)

                self.__histogram1 = [h_w11, h_w12, h_b1]

                new_dim = self.in_dim / 2 + self.in_dim % 2
                # print(new_dim)

            with tf.variable_scope('layer2'):
                self.__w21, h_w21 = get_weight_var_and_hist('w21', [1, 3, 3, l1_n, l2_n], trainable=trainable)
                self.__w22, h_w22 = get_weight_var_and_hist('w22', [1, 3, 3, l2_n, l2_n], trainable=trainable)
                self.__w23, h_w23 = get_weight_var_and_hist('w23', [1, 2, 2, l2_n, l2_n], trainable=trainable)
                self.__b2, h_b2 = get_bias_var_and_hist('b2', [l2_n], trainable=trainable)

                self.__histogram2 = [h_w21, h_w22, h_w23, h_b2]

                new_dim = (new_dim - 2 + 1) / 2 + (new_dim - 2 + 1) % 2
                # print(new_dim)

            with tf.variable_scope('layer3'):
                self.__w31, h_w31 = get_weight_var_and_hist('w31', [1, 3, 3, l2_n, l3_n], trainable=trainable)
                self.__w32, h_w32 = get_weight_var_and_hist('w32', [1, 3, 3, l3_n, l3_n], trainable=trainable)
                self.__b3, h_b3 = get_bias_var_and_hist('b3', [l3_n], trainable=trainable)

                self.__histogram3 = [h_w31, h_w32, h_b3]

                new_dim = new_dim / 2 + new_dim % 2
                # print(new_dim)

            with tf.variable_scope('layer4'):
                self.__w41, h_w41 = get_weight_var_and_hist('w41', [1, 3, 3, l3_n, l4_n], trainable=trainable)
                self.__w42, h_w42 = get_weight_var_and_hist('w42', [1, 3, 3, l4_n, l4_n], trainable=trainable)
                self.__w43, h_w43 = get_weight_var_and_hist('w43', [1, 2, 2, l4_n, l4_n], trainable=trainable)
                self.__b4, h_b4 = get_bias_var_and_hist('b4', [l4_n], trainable=trainable)

                self.__histogram4 = [h_w41, h_w42, h_w43, h_b4]

                new_dim = new_dim / 2 + new_dim % 2
                # print(new_dim)

            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                flat_dim = new_dim ** 2
                self.__wfc1, h_wfc1 = get_weight_var_and_hist('wfc1', [2 * flat_dim * l4_n, fc1_n], trainable=trainable)
                self.__bfc1, h_bfc1 = get_bias_var_and_hist('bfc1', [fc1_n], trainable=trainable)

                self.__histogram5 = [h_wfc1, h_bfc1]

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2
                self.__wfc2, h_wfc2 = get_weight_var_and_hist('wfc2', [fc1_n, params_dim], trainable=trainable)
                self.__bfc2, h_bfc2 = get_bias_var_and_hist('bfc2', [params_dim], trainable=trainable)

                self.__histogram6 = [h_wfc2, h_bfc2]

        self.saver = tf.train.Saver()

    def __call__(self, ref_slice, off_slice):
        # Stack so that they have the shape [z, y, x], then expand so they have [batch, z, y, x, chan]
        assert (len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # ref_slice = tf.Print(ref_slice, [tf.reduce_max(ref_slice)], message='Max of ref')
        # off_slice = tf.Print(off_slice, [tf.reduce_max(off_slice)], message='Max of offset')

        # Stack on the z dim, then expand to batch=1
        stacked = tf.expand_dims(tf.concat([ref_slice, off_slice], axis=0), axis=0)

        # Layer 1
        l11 = tf.nn.convolution(stacked, self.__w11, padding='SAME')
        l12 = tf.nn.convolution(l11, self.__w12, padding='SAME')
        l1 = tf.nn.relu(l12 + self.__b1)

        self.__histogram1.append(tf.summary.histogram('l1', l1))
        histogram1 = tf.summary.merge(self.__histogram1)

        p1 = tf.nn.pool(l1, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        # print(p1.get_shape())

        # Layer 2
        l21 = tf.nn.convolution(p1, self.__w21, padding='SAME')
        l22 = tf.nn.convolution(l21, self.__w22, padding='SAME')
        l23 = tf.nn.convolution(l22, self.__w23, padding='VALID')
        l2 = tf.nn.relu(l23 + self.__b2)

        self.__histogram2.append(tf.summary.histogram('l2', l2))
        histogram2 = tf.summary.merge(self.__histogram2)

        p2 = tf.nn.pool(l2, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        # print(p2.get_shape())

        # Layer 3
        l31 = tf.nn.convolution(p2, self.__w31, padding='SAME')
        l32 = tf.nn.convolution(l31, self.__w32, padding='SAME')
        l3 = tf.nn.relu(l32 + self.__b3)

        self.__histogram3.append(tf.summary.histogram('l3', l3))
        histogram3 = tf.summary.merge(self.__histogram3)

        p3 = tf.nn.pool(l3, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        # print(p3.get_shape())

        # Layer 4
        l41 = tf.nn.convolution(p3, self.__w41, padding='SAME')
        l42 = tf.nn.convolution(l41, self.__w42, padding='SAME')
        l43 = tf.nn.convolution(l42, self.__w43, padding='SAME')
        l4 = tf.nn.relu(l43 + self.__b4)

        self.__histogram4.append(tf.summary.histogram('l4', l4))
        histogram4 = tf.summary.merge(self.__histogram4)

        p4 = tf.nn.pool(l4, window_shape=[1, 2, 2], strides=[1, 2, 2], pooling_type='MAX', padding='SAME')

        # print(p4.get_shape())

        l4_flat = tf.reshape(p4, [1, -1])

        # Fully connected 1
        fc1 = tf.nn.tanh(tf.matmul(l4_flat, self.__wfc1) + self.__bfc1)

        self.__histogram5.append(tf.summary.histogram('l5', fc1))
        histogram5 = tf.summary.merge(self.__histogram5)

        # Fully connected 2
        theta = tf.nn.tanh(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

        self.__histogram6.append(tf.summary.histogram('theta', theta))
        histogram6 = tf.summary.merge(self.__histogram6)

        histograms = tf.summary.merge([histogram1, histogram2, histogram3, histogram4, histogram5, histogram6])

        # Squish it down to 2d, so that our transformer layer can process it
        return tf.reshape(theta, [-1]), histograms


class FCLocalizationNetwork(LocalizationNetwork):
    def __init__(self, in_dim, params_dim, fc1_units, trainable=True):
        super(FCLocalizationNetwork, self).__init__(in_dim=in_dim, params_dim=params_dim, trainable=trainable)

        with tf.variable_scope('fc_localization'):
            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                self.__wfc1, h_wfc1 = get_weight_var_and_hist('wfc1', [2 * in_dim * in_dim, fc1_units],
                                                              dtype=tf.float32, trainable=trainable)
                self.__bfc1, b_wfc1 = get_bias_var_and_hist('bfc1', [fc1_units], dtype=tf.float32, trainable=trainable)

                self.__histogram1 = [h_wfc1, b_wfc1]

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2. with bias set to 0
                self.__wfc2, h_wfc2 = get_weight_var_and_hist('wfc2', [fc1_units, params_dim], dtype=tf.float32,
                                                              trainable=trainable)
                self.__bfc2, b_wfc2 = get_bias_var_and_hist('bfc2', shape=[params_dim], dtype=tf.float32,
                                                            trainable=trainable)

                self.__histogram2 = [h_wfc2, b_wfc2]

    def __call__(self, ref_slice, off_slice):
        # make sure they're in the shape [z, y, x, chan]
        assert (len(ref_slice.get_shape()) == 4)
        assert (len(off_slice.get_shape()) == 4)

        # Stack on the z dim, then expand to batch=1
        stacked = tf.concat([ref_slice, off_slice], axis=0)

        # Flatten
        flat = tf.reshape(stacked, [1, -1])

        with tf.variable_scope('fc_localization'):
            with tf.variable_scope('layer_fc1'):
                # Fully connected 1
                fc1 = tf.nn.tanh(tf.matmul(flat, self.__wfc1) + self.__bfc1)

                self.__histogram1.append(tf.summary.histogram('fc1', fc1))
                histogram1 = tf.summary.merge(self.__histogram1)

            with tf.variable_scope('layer_fc2'):
                # Fully connected 2, w/ tanh
                theta = tf.nn.tanh(tf.matmul(fc1, self.__wfc2) + self.__bfc2)

                self.__histogram2.append(tf.summary.histogram('theta', theta))
                histogram2 = tf.summary.merge(self.__histogram2)

        histograms = tf.summary.merge([histogram1, histogram2])

        # Make sure theta is 1D
        return tf.reshape(theta, [-1]), histograms
