# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import tensorflow as tf

try:
    from trace.thirdparty.segascorus import io_utils
    from trace.thirdparty.segascorus import utils
    from trace.thirdparty.segascorus.metrics import *
except ImportError:
    print("Segascorus is not installed. Please install by going to trace/trace/thirdparty/segascorus and run 'make'."
          " If this fails, segascorus is likely not compatible with your computer (i.e. Macs).")


class Learner:
    def __init__(self, model, ckpt_folder):
        self.model = model
        self.sess = tf.Session()
        self.ckpt_folder = ckpt_folder

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def train(self, training_params, dset_sampler, hooks, continue_training=False):
        sess = self.sess
        model = self.model

        # We will write our summaries here
        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        # Define an optimizer
        optimize_step = training_params.optimizer(training_params.learning_rate).minimize(model.cross_entropy,
                                                                                          global_step=model.global_step)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        if continue_training:
            self.restore()
            print('Starting training at step: ' + str(sess.run(model.global_step)))
        dset_sampler.initialize_session_variables(sess)

        # Create enqueue op and a QueueRunner to handle queueing of training examples
        enqueue_op = model.queue.enqueue(dset_sampler.get_sample_op())
        qr = tf.train.QueueRunner(model.queue, [enqueue_op] * 4)

        '''
        sess.run(enqueue_op)
        sess.run(enqueue_op)
        sess.run(optimize_step)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(optimize_step, options=run_options, run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        print('done')
        '''

        # Create a Coordinator, launch the Queuerunner threads
        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, daemon=True, start=True)

        '''
        def train_function():
            step = 0
            while True:
                print(step)
                sess.run(optimize_step)
                step += 1

        train_threads = []
        for _ in range(8):
            th = threading.Thread(target=train_function)
            th.daemon = True
            train_threads.append(th)
        for th in train_threads:
            th.start()
        for th in train_threads:
            th.join()
        '''

        # Iterate through the dataset
        begin_step = sess.run(model.global_step)
        for step in range(begin_step, training_params.n_iterations):
            if coord.should_stop():
                break
            if step % 10 == 0:
                print(step)

            # Run the optimizer
            sess.run(optimize_step)

            for hook in hooks:
                hook.eval(step, model, sess, summary_writer)

        # with tf.device('/cpu:0'):
        coord.request_stop()
        coord.join(enqueue_threads)

    def restore(self):
        self.model.saver.restore(self.sess, self.ckpt_folder + 'model.ckpt')
        print("Model restored.")

    def predict(self, inputs, inference_params, mirror_inputs=False):
        # Make sure that the inputs are 5-dimensional, in the form [batch_size, z_dim, y_dim, x_dim, n_chan]

        assert (len(inputs.shape) == 5)

        return self.model.predict(self.sess, inputs, inference_params, mirror_inputs)
