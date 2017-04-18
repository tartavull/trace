import tensorflow as tf
import alignment as al
import math
import tifffile as tiff
import numpy as np

# print('Loading data')
# train_stack = np.expand_dims(tiff.imread('./isbi/train-input.tif'), axis=3)
#
# with tf.Graph().as_default():
#     with tf.Session() as sess:
#         rot = True
#         max_angle = math.pi / 10
#         trans = True
#         max_shift = 25
#         print('Creating sampler')
#         lsamp = al.LocalizationSampler(train_stack, rotation_aug=rot, max_angle=max_angle, translation_aug=trans,
#                                        max_shift=max_shift)
#
#         ref_op, sec_op, true_sec_op = lsamp.get_sample_funcs()
#
#         ref_op = tf.cast(ref_op, dtype=tf.float32) / 255.0
#         sec_op = tf.cast(sec_op, dtype=tf.float32) / 255.0
#         true_sec_op = tf.cast(true_sec_op, dtype=tf.float32) / 255.0
#
#         print('Creating transformer')
#         aff_trans = al.AffineSpatialTransformer(774)
#
#         trans_op = aff_trans(ref_op, sec_op)
#         print('Initializing')
#         sess.run(tf.global_variables_initializer())
#         #     res = sess.run(trans2)
#         print('Running transformer')
#         res, ref, sec, tsec = sess.run([trans_op, ref_op, sec_op, true_sec_op])
#         print('Done')

tf.reset_default_graph()
print('load dataset')
train_stack = np.expand_dims(tiff.imread('./isbi/train-input.tif'), axis=3) / 255.0
dim = int(train_stack.shape[1] * (math.sqrt(2)) + 100)

print('create transformer')
aff_trans = al.AffineSpatialTransformer(dim)

print('create trainer')
trainer = al.LocalizationTrainer(aff_trans, './realignment/spatial_transformer/affine/test7/')

print('train')
with tf.Session() as sess:
    trainer.train(sess, train_stack, 1010)
