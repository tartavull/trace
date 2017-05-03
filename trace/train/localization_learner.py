import tensorflow as tf
import tensorflow.contrib.image as tfim
import numpy as np

from trace.sampling.augmentation import gkern
from trace.evaluation import tf_pixel_error_scalar, tf_l2_loss_scalar, tf_cross_entropy_scalar, \
    tf_cross_correlation_scalar


class LocalizationTrainer(object):
    def __init__(self, transformer, localization_sampler, validation_set, ckpt_folder):
        self.transformer = transformer
        self.ckpt_folder = ckpt_folder
        self.validation_set = validation_set

        ref_op, sec_op, true_realign_op, self.true_theta_op = localization_sampler.get_sample_funcs()

        self.ref_placeholder = tf.placeholder_with_default(ref_op, [1, None, None, 1], name='ref_placeholder')
        self.sec_placeholder = tf.placeholder_with_default(sec_op, [1, None, None, 1], name='sec_placeholder')

        # Assumption: images are on the interval [0, 255], so we reduce by
        self.ref_scaled = self.ref_placeholder / 255.0
        self.sec_scaled = self.sec_placeholder / 255.0
        self.tru_scaled = true_realign_op / 255.0

        pred_realign_2d_op, pred_theta_op, self.valid_size = self.transformer(self.ref_scaled, self.sec_scaled)

        # Transformer produces [None, None]
        self.pred_realign_op = tf.expand_dims(tf.expand_dims(pred_realign_2d_op, axis=0), axis=3)

        # Expand theta a bit, and scale the dx and dy
        # TODO(beisner): investigate whether or not we need to artificially scale the transformation (i don't think)
        shape = tf.cast(tf.shape(ref_op)[1], tf.float32)
        self.exp_pred_theta = tf.stack([pred_theta_op[0], pred_theta_op[1], pred_theta_op[2] * shape, pred_theta_op[3],
                                        pred_theta_op[4], pred_theta_op[5] * shape, 0.0, 0.0], axis=0)

        # Compute the image externally
        self.externally_computed = tfim.transform(self.sec_scaled, self.exp_pred_theta)

        # Pixel error
        self.pixel_error, p_err_sum = tf_pixel_error_scalar(self.pred_realign_op, self.tru_scaled, inc_summary=True)

        # Cross correlation
        self.x_corr, x_corr_sum = tf_cross_correlation_scalar(self.pred_realign_op, self.tru_scaled, inc_summary=True)

        # Externally computed
        self.ext_x_corr, ext_x_corr_sum = tf_cross_correlation_scalar(self.externally_computed, self.tru_scaled,
                                                                      name='external_x_corr', inc_summary=True)

        # Externally computed
        self.ext_p_err, ext_p_err_sum = tf_pixel_error_scalar(self.externally_computed, self.tru_scaled,
                                                              name='external_pixel_error', inc_summary=True)
        # L2 loss for theta
        self.l2_loss, l2_loss_sum = tf_l2_loss_scalar(self.exp_pred_theta, self.true_theta_op, inc_summary=True)

        # Since there is potentially some black space induced by the transformation, we must do our loss based on
        # a patch where no overlap is guaranteed
        size = tf.shape(self.pred_realign_op)[1]
        bound = (size - self.valid_size) / 2

        true_patch = self.tru_scaled[:, bound:(bound + self.valid_size), bound:(bound + self.valid_size), :]
        pred_patch = self.pred_realign_op[:, bound:(bound + self.valid_size), bound:(bound + self.valid_size), :]

        pat_as_filter = tf.expand_dims(tf.squeeze(pred_patch, axis=0), axis=3)

        # Smooth pat_as_filter so that the gradient is better... maybe
        # Idea is that xcorrelation is really not smoothly differentiable, so we smooth it out with a gaussian

        gaussian_kernel = np.asarray(np.expand_dims(np.expand_dims(gkern(10, 0.5), axis=2), axis=3), dtype=np.float32)
        normed_kernel = gaussian_kernel / np.sum(gaussian_kernel)

        smoothed_true = tf.nn.convolution(true_patch, normed_kernel, padding='SAME')

        # Calculate the cross-correlation
        self.pat_x_corr, p_x_his = tf_cross_correlation_scalar(pred_patch, true_patch, name='patch_x_corr',
                                                               inc_summary=True)
        self.pat_smooth_x_corr, p_sm_x_his = tf_cross_correlation_scalar(pred_patch, smoothed_true,
                                                                         name='patch_smooth_x_corr',
                                                                         inc_summary=True)

        self.training_summaries = tf.summary.merge(
            [p_err_sum, x_corr_sum, l2_loss_sum, ext_p_err_sum, ext_x_corr_sum, p_x_his, p_sm_x_his])

        ref_summary = tf.summary.image('img_reference', self.ref_scaled)
        sec_summary = tf.summary.image('img_misaligned', self.sec_scaled)
        true_realign_summary = tf.summary.image('realigned_true', self.tru_scaled)
        pred_realign_summary = tf.summary.image('realigned_pred', self.pred_realign_op)
        ext_realign_summary = tf.summary.image('realigned_ext', self.externally_computed)
        true_patch_sum = tf.summary.image('patch_true', true_patch)
        pred_patch_sum = tf.summary.image('patch_pred', pred_patch)

        self.image_summaries = tf.summary.merge([ref_summary, sec_summary, true_realign_summary, pred_realign_summary,
                                                 ext_realign_summary, true_patch_sum, pred_patch_sum])

        self.validation_pix_error = tf.placeholder(tf.float32, shape=(), name='validation_value')
        self.val_pix_summary = tf.summary.scalar('validation_pixel_error', self.validation_pix_error)

    # def train_on_transformations(self, sess, n_iter, sampler, lr, validation_set):
        # ref_op, sec_op, true_realign_op, true_theta_op = sampler.get_sample_funcs()

        # ref_placeholder = tf.placeholder_with_default(ref_op, [1, None, None, 1], name='ref_placeholder')
        # sec_placeholder = tf.placeholder_with_default(sec_op, [1, None, None, 1], name='sec_placeholder')

        # Assumption: images are on the interval [0, 255], so we reduce by

        # small_pred_realign_op, pred_theta_op, _ = self.transformer(ref_placeholder, sec_placeholder)

        # Transformer produces [None, None]
        # pred_realign_op = tf.expand_dims(tf.expand_dims(small_pred_realign_op, axis=0), axis=3)

        # Scale theta for tfim
        # shape = tf.cast(tf.shape(ref_op)[1], tf.float32)
        # exp_pred_theta = tf.stack([pred_theta_op[0], pred_theta_op[1], pred_theta_op[2] * shape, pred_theta_op[3],
        #                            pred_theta_op[4], pred_theta_op[5] * shape, 0.0, 0.0], axis=0)

        # Compute the image externally
        # externally_computed = tfim.transform(sec_op, exp_pred_theta)
        #
        # # Pixel error
        # pixel_error, p_err_sum = tf_pixel_error_scalar(pred_realign_op, true_realign_op, inc_summary=True)
        #
        # # Cross correlation
        # x_corr, x_corr_sum = tf_cross_correlation_scalar(pred_realign_op, true_realign_op, inc_summary=True)
        #
        # ext_x_corr, ext_x_corr_sum = tf_cross_correlation_scalar(externally_computed, true_realign_op, name='external_x_corr', inc_summary=True)
        #
        # ext_p_err, ext_p_err_sum = tf_pixel_error_scalar(externally_computed, true_realign_op, name='external_pixel_error', inc_summary=True)

        # L2 loss for theta
        # l2_loss, l2_loss_sum = tf_l2_loss_scalar(exp_pred_theta, true_theta_op, inc_summary=True)

        # Minimize L2 Loss
        # optimizer = tf.train.AdamOptimizer(lr).minimize(l2_loss)
        #
        # summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        # training_summaries = tf.summary.merge([p_err_sum, x_corr_sum, l2_loss_sum, ext_p_err_sum, ext_x_corr_sum])

        # ref_summary = tf.summary.image('img_reference', ref_op)
        # sec_summary = tf.summary.image('img_misaligned', sec_op)
        # true_realign_summary = tf.summary.image('realigned_true', true_realign_op)
        # pred_realign_summary = tf.summary.image('realigned_pred', pred_realign_op)
        # ext_realign_summary = tf.summary.image('realigned_ext', externally_computed)
        #
        # image_summaries = tf.summary.merge([ref_summary, sec_summary, true_realign_summary, pred_realign_summary,
        #                                     ext_realign_summary])
        #
        # validation_pix_error = tf.placeholder(tf.float32, shape=(), name='validation_value')
        # val_pix_summary = tf.summary.scalar('validation_pixel_error', validation_pix_error)

        # sess.run(tf.initialize_all_variables())

        # for i in range(n_iter):
        #     # Run the optimizer
        #     sess.run(optimizer)
        #
        #     # Output to tensorboard
        #     if i % 10 == 0:
        #         print(i)
        #         train_summary_rep = sess.run(self.training_summaries)
        #         summary_writer.add_summary(train_summary_rep, i)
        #
        #     # Image summary
        #     if i % 100 == 0:
        #         image_summary_rep, pred_theta_vals, t_theta_vals = sess.run([self.image_summaries, self.exp_pred_theta,
        #                                                                      self.true_theta_op])
        #         print('theta_pred: ' + ', '.join(["%.6f" % j for j in pred_theta_vals]))
        #         print('theta_true: ' + ', '.join(["%.6f" % j for j in t_theta_vals]))
        #         summary_writer.add_summary(image_summary_rep, i)
        #         print('Computed images')
        #
        #     # Validation summary
        #     if i % 500 == 0:
        #         preds = []
        #         truth = []
        #         for example in validation_set:
        #             pred = sess.run([self.pred_realign_op], feed_dict={
        #                 self.ref_placeholder: example.ref_slice,
        #                 self.sec_placeholder: example.off_slice,
        #             })
        #             preds.append(pred)
        #             truth.append(example.corr_off_slice)
        #
        #         pix_err = np.mean(np.abs(np.asarray(preds) - np.asarray(truth)))
        #
        #         print('Validation pixel error %.6f' % pix_err)
        #
        #         val_res = sess.run(self.val_pix_summary, feed_dict={
        #             self.validation_pix_error: pix_err,
        #         })
        #
        #         summary_writer.add_summary(val_res, i)
        #
        #     # Model saver
        #     if i % 1000 == 0:
        #         hist_rep = sess.run(self.transformer.histograms)
        #         summary_writer.add_summary(hist_rep, i)
        #         print('Computed histograms')
        #
        #         self.transformer.saver.save(sess, self.ckpt_folder + 'model.ckpt')
        #         print('Model saved in %smodel.ckpt' % self.ckpt_folder)

    # def train_on_images(self, sess, n_iter, sampler, lr):
    #
    #     # ref_op, sec_op, true_realign_op, true_theta = sampler.get_sample_funcs()
    #
    #     # pred_realign, theta, valid_size = self.transformer(ref_op, sec_op)
    #
    #     # pred_realign = tf.expand_dims(tf.expand_dims(pred_realign, axis=0), axis=3)
    #
    #     # externally_computed = tfim.transform(sec_op, [theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], 0, 0])
    #
    #     # pixel_error = tf.reduce_mean(tf.abs(pred_realign - true_realign_op))
    #
    #
    #
    #     # conv_x_corr = tf.squeeze(tf.nn.convolution(true_patch, pat_as_filter, padding='VALID')) / float(
    #     #     self.valid_size * self.valid_size)
    #
    #     optimizer = tf.train.AdamOptimizer(lr).minimize(-self.pat_smooth_x_corr)
    #
    #
    #
    #     # ref_summary = tf.summary.image('img_reference', ref_op)
    #     # sec_summary = tf.summary.image('img_misaligned', sec_op)
    #     # true_realign_summary = tf.summary.image('realigned_true', true_realign_op)
    #     # pred_realign_summary = tf.summary.image('realigned_pred', pred_realign)
    #     # diff_summary = tf.summary.image('diff', tf.abs(pred_realign - true_realign_op))
    #     # ext_summary = tf.summary.image('externally_computed', externally_computed)
    #     #
    #     # true_patch_sum = tf.summary.image('patch_true', true_patch)
    #     # pred_patch_sum = tf.summary.image('patch_pred', pred_patch)
    #     #
    #     # image_summaries = tf.summary.merge(
    #     #     [ref_summary, sec_summary, true_realign_summary, pred_realign_summary, true_patch_sum, pred_patch_sum,
    #     #      diff_summary, ext_summary])
    #
    #
    #
    #     for i in range(n_iter):
    #         # Run the optimizer
    #         sess.run(optimizer)
    #
    #         print(i)
    #
    #         # Output to tensorboard
    #         if i % 10 == 0:
    #             train_summary_rep, theta_vals = sess.run([training_summaries, theta])
    #             summary_writer.add_summary(train_summary_rep, i)
    #             print('theta: ' + ', '.join(["%.6f" % j for j in theta_vals.tolist()]))
    #
    #         # Image summary
    #         if i % 100 == 0:
    #             loss_summary_rep, hist_summaries = sess.run([image_summaries, self.transformer.histograms])
    #             summary_writer.add_summary(hist_summaries, i)
    #             summary_writer.add_summary(loss_summary_rep, i)
    #             print('Computed images and histogram')
    #
    #         # Model saver
    #         if i % 1000 == 0:
    #             self.transformer.saver.save(sess, self.ckpt_folder + 'model.ckpt')
    #             print('Model saved in %smodel.ckpt' % self.ckpt_folder)

    def train(self, sess, optimizer, n_iter, lr, train_on_parameters=False):
        if train_on_parameters:
            # Minimize L2 Loss
            optimizer = tf.train.AdamOptimizer(lr).minimize(self.l2_loss)
        else:
            optimizer = tf.train.AdamOptimizer(lr).minimize(-self.pat_smooth_x_corr)

        sess.run(tf.initialize_all_variables())

        summary_writer = tf.summary.FileWriter(self.ckpt_folder + '/events', graph=sess.graph)

        for i in range(n_iter):
            # Run the optimizer
            sess.run(optimizer)

            # Output to tensorboard
            if i % 10 == 0:
                print(i)
                train_summary_rep = sess.run(self.training_summaries)
                summary_writer.add_summary(train_summary_rep, i)

            # Image summary
            if i % 100 == 0:
                image_summary_rep, pred_theta_vals, t_theta_vals = sess.run([self.image_summaries, self.exp_pred_theta,
                                                                             self.true_theta_op])
                print('theta_pred: ' + ', '.join(["%.6f" % j for j in pred_theta_vals]))
                print('theta_true: ' + ', '.join(["%.6f" % j for j in t_theta_vals]))
                summary_writer.add_summary(image_summary_rep, i)
                print('Computed images')

            # Validation summary
            if i % 500 == 0:
                preds = []
                truth = []
                for example in self.validation_set:
                    pred = sess.run([self.pred_realign_op], feed_dict={
                        self.ref_placeholder: example.ref_slice,
                        self.sec_placeholder: example.off_slice,
                    })
                    preds.append(pred)
                    truth.append(example.corr_off_slice)

                pix_err = np.mean(np.abs(np.asarray(preds) - np.asarray(truth)))

                print('Validation pixel error %.6f' % pix_err)

                val_res = sess.run(self.val_pix_summary, feed_dict={
                    self.validation_pix_error: pix_err,
                })

                summary_writer.add_summary(val_res, i)

            # Model saver
            if i % 1000 == 0:
                hist_rep = sess.run(self.transformer.histograms)
                summary_writer.add_summary(hist_rep, i)
                print('Computed histograms')

                self.transformer.saver.save(sess, self.ckpt_folder + 'model.ckpt')
                print('Model saved in %smodel.ckpt' % self.ckpt_folder)
