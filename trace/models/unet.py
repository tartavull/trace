import tensorflow as tf


class UNet:
    def __init__(self, params, is_training=False):

        print(params['model_name'])

        learning_rate = params['learning_rate']

        # Define the inputs
        # self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        # self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        self.image = tf.placeholder(tf.float32, shape=[1, 572, 572, 1])
        self.target = tf.placeholder(tf.float32, shape=[1, 388, 388, 2])

