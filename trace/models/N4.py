import tensorflow as tf

from trace.models.common import conv2d, bias_variable, weight_variable, max_pool


class N4:
    def __init__(self, fov, out, learning_rate=0.001):

        self.fov = fov
        self.out = out
        self.input = self.fov + 2 * (self.out//2)

        # layer 0
        self.image = tf.placeholder(tf.float32, shape=[1, inpt, inpt, 1])
        self.target = tf.placeholder(tf.float32, shape=[1, out, out, 2])

        # layer 1 - original stride 1
        W_conv1 = weight_variable([4, 4, 1, 48])
        b_conv1 = bias_variable([48])
        h_conv1 = tf.nn.relu(conv2d(self.image, W_conv1, dilation=1) + b_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, 48, 48])
        b_conv2 = bias_variable([48])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2) + b_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, 48, 48])
        b_conv3 = bias_variable([48])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4) + b_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=4)

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, 48, 48])
        b_conv4 = bias_variable([48])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8) + b_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1, 1], dilation=8)

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, 48, 200])
        b_fc1 = bias_variable([200])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=16) + b_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, 200, 2])
        b_fc2 = bias_variable([2])
        self.prediction = conv2d(h_fc1, W_fc2, dilation=16) + b_fc2

        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.prediction, self.target))
        self.loss_summary = tf.scalar_summary('cross_entropy', self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
