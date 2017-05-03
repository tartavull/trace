import tensorflow as tf


def __cond_scalar_summary(scalar, inc_summary, name):
    if inc_summary:
        summary = tf.summary.scalar(name, scalar)
        return scalar, summary
    else:
        return scalar


def tf_pixel_error_scalar(pred_data, true_data, inc_summary=False, name='pixel_error'):
    pixel_error = tf.reduce_mean(tf.abs(true_data - pred_data))
    return __cond_scalar_summary(pixel_error, inc_summary, name)


def tf_cross_correlation_scalar(pred_data, true_data, inc_summary=False, name='cross_correlation'):
    x_corr = tf.reduce_mean(true_data * pred_data)
    return __cond_scalar_summary(x_corr, inc_summary, name)


def tf_l2_loss_scalar(pred_data, true_data, inc_summary=False, name='l2_loss'):
    l2_loss = tf.nn.l2_loss(true_data - pred_data)
    return __cond_scalar_summary(l2_loss, inc_summary, name)


def tf_cross_entropy_scalar(logits, labels, inc_summary=False, name='cross_entropy'):
    x_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return __cond_scalar_summary(x_entropy, inc_summary, name)
