import tensorflow as tf
import alignment as al
import math
import tifffile as tiff
import numpy as np

rotation = False
max_angle = math.pi / 10
translation = True
max_shift = 10

tf.reset_default_graph()
print('load dataset')
train_stack = np.expand_dims(tiff.imread('./isbi/train-input.tif'), axis=3) / 255.0
dim = int(train_stack.shape[1])

print('create transformer')
aff_trans = al.ConvTranslationSpatialTransformer(in_dim=dim, l1_n=24, l2_n=36, l3_n=48, l4_n=64, fc1_n=512, max_shift=max_shift, trainable=True)
# aff_trans = al.FCTranslationSpatialTransformer(in_dim=dim, fc1_n=512, max_shift=max_shift, trainable=True)
# aff_trans = al.FCTranslationalSpatialTransformer(dim, max_shift)
# aff_trans = al.ConvAffineSpatialTransformer(dim, 50)

print('create sampler')
sampler = al.LocalizationSampler(train_stack, rotation_aug=False, max_angle=max_angle, translation_aug=True, max_shift=max_shift)

print('create trainer')
trainer = al.LocalizationTrainer(aff_trans, './realignment/spatial_transformer/trans_conv_sm/test1')

print('train')
with tf.Session() as sess:
    trainer.train(sess, 100000, sampler, lr=0.00001)
