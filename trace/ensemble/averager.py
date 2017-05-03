import tensorflow as tf
import numpy as np


class ModelAverager:
    name = 'model_averaging'

    def __init__(self, ensembler_folder):
        self.ensembler_folder = ensembler_folder

    # No training needed
    def train(self, classifiers):
        pass

    def predict(self, predictions):
        with tf.Session() as sess:
            # Initialize the variable
            sess.run(tf.global_variables_initializer())

            # num models, num slices, dim, dim, channels
            outputs = tf.placeholder(tf.float32, shape=[None, None, None, None, None])

            # Take the average of the outputs along the first dimension
            averaged = tf.reduce_mean(outputs, axis=0)

            # Run the prediction
            result = sess.run(averaged, feed_dict={outputs: np.asarray(predictions)})

        return result

