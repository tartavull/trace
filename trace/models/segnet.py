import tensorflow as tf

from common import conv2d, bias_variable, weight_variable, max_pool

SMALL_LAYER = 2
LARGE_LAYER = 3

# taken from https://arxiv.org/pdf/1511.00561.pdf
def default_SegNet():
    params = {
        's1': [
            [4, 4, 1, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4]
        ],
        's2': [
            [4, 4, 4, 8],
            [4, 4, 8, 8],
            [4, 4, 8, 8]
        ],
        's3': [
            [4, 4, 8, 16],
            [4, 4, 16, 16],
            [4, 4, 16, 16]
        ],
        's4': [
            [4, 4, 16, 32],
            [4, 4, 32, 32],
            [4, 4, 32, 32]
        ],
        's5': [
            [4, 4, 32, 64],
            [4, 4, 64, 64],
            [4, 4, 64, 64]
        ],

        'fc': 200,
        'lr': 0.001,
        'out': 101
    }
    return SegNet(params)

# convolution, batch normalization, and ReLU layer
def conv_norm_relu(inlayer, shape, dilation):
    # convolution
    W_conv  = weight_variable(shape)
    s_conv  = conv2d(inlayer, W_conv, dilation=dilation)

    # batch normalization
    mean_bn, var_bn = tf.nn.moments(s_conv, [0])
    offset_bn = tf.Variable(tf.zeros([shape[-1]]))
    scale_bn = tf.Variable(tf.ones([shape[-1]]))
    bn_conv = tf.nn.batch_normalization(s_conv, mean=mean_bn, variance=var_bn, 
        offset=offset_bn, scale=scale_bn, variance_epsilon=.0005)

    # ReLU
    return tf.nn.relu(bn_conv)

class SegNet:
    def __init__(self, params):

        # Hyperparameters
        s1 = params['s1']
        s2 = params['s2']
        s3 = params['s3']
        s4 = params['s4']
        s5 = params['s5']

        fc = params['fc']
        learning_rate = params['lr']

        self.out = params['out']
        self.fov = 95
        self.inpt = self.fov + 2 * (self.out // 2)

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        # layer set 1: 2 Conv/Batch/ReLU and 1 pool
        down_conv1 = self._downsample_layer(inlayer=self.image, 
            shapes=s1, dilation=1, cbr_layers=2)

        # layer set 2: 2 Conv/Batch/ReLU and 1 pool
        down_conv2 = self._downsample_layer(inlayer=down_conv1, 
            shapes=s2, dilation=2, cbr_layers=2)

        # layer set 3: 3 Conv/Batch/ReLU and 1 pool
        down_conv3 = self._downsample_layer(inlayer=down_conv2, 
            shapes=s3, dilation=4, cbr_layers=3)

        # layer set 4: 3 Conv/Batch/ReLU and 1 pool
        down_conv4 = self._downsample_layer(inlayer=down_conv3, 
            shapes=s4, dilation=8, cbr_layers=3)

        # layer set 5: 3 Conv/Batch/ReLU and 1 pool
        down_conv5 = self._downsample_layer(inlayer=down_conv4, 
            shapes=s5, dilation=16, cbr_layers=3)

        # layer set 6: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        up_conv1 = self._upsample_layer(inlayer=down_conv5, 
            shapes=s5, dilation=16, cbr_layers=3)

        # layer set 7: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        up_conv2 = self._upsample_layer(inlayer=up_conv1, 
            shapes=s4, dilation=8, cbr_layers=3)

        # layer set 8: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        up_conv3 = self._upsample_layer(inlayer=up_conv2, 
            shapes=s3, dilation=4, cbr_layers=3)
        
        # layer set 9: 1 unpool (upsampling) and 2 Conv/Batch/ReLU
        up_conv4 = self._upsample_layer(inlayer=up_conv3, 
            shapes=s2, dilation=2, cbr_layers=2)

        # layer set 10: 1 unpool (upsampling) and 2 Conv/Batch/ReLU
        self.prediction = self._upsample_layer(inlayer=up_conv4, 
            shapes=s1, dilation=1, cbr_layers=2)

        self.softmax = tf.nn.softmax(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.target))
        self.loss_summary = tf.scalar_summary('cross_entropy', self.cross_entropy)
        self.train_step = tf.train.AdagradDAOptimizer(learning_rate).minimize(self.cross_entropy)

        self.binary_prediction = tf.round(self.softmax)
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))
        self.pixel_error_summary = tf.summary.scalar('pixel_error', self.pixel_error)
        self.validation_pixel_error_summary = tf.summary.scalar('validation pixel_error', self.pixel_error)

        self.rand_f_score = tf.placeholder(tf.float32)
        self.rand_f_score_merge = tf.placeholder(tf.float32)
        self.rand_f_score_split = tf.placeholder(tf.float32)
        self.vi_f_score = tf.placeholder(tf.float32)
        self.vi_f_score_merge = tf.placeholder(tf.float32)
        self.vi_f_score_split = tf.placeholder(tf.float32)

        self.rand_f_score_summary = tf.summary.scalar('rand f score', self.rand_f_score)
        self.rand_f_score_merge_summary = tf.summary.scalar('rand f merge score', self.rand_f_score_merge)
        self.rand_f_score_split_summary = tf.summary.scalar('rand f split score', self.rand_f_score_split)
        self.vi_f_score_summary = tf.summary.scalar('vi f score', self.vi_f_score)
        self.vi_f_score_merge_summary = tf.summary.scalar('vi f merge score', self.vi_f_score_merge)
        self.vi_f_score_split_summary = tf.summary.scalar('vi f split score', self.vi_f_score_split)

        self.score_summary_op = tf.summary.merge([self.rand_f_score_summary,
                                                 self.rand_f_score_merge_summary,
                                                 self.rand_f_score_split_summary,
                                                 self.vi_f_score_summary,
                                                 self.vi_f_score_merge_summary,
                                                 self.vi_f_score_split_summary
                                                 ])

        self.summary_op = tf.summary.merge([self.loss_summary,
                                       self.pixel_error_summary
                                       ])

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        self.model_name = str.format('out-{}_lr-{}_m1-{}_m2-{}_m3-{}_m4-{}_fc-{}', self.out, learning_rate, map_1,
                                     map_2, map_3, map_4, fc)

    def _downsample_layer(self, inlayer, shapes, dilation=1, cbr_layers=2):
        h_conv1 = conv_norm_relu(inlayer=inlayer, shape=shapes[0], dilation=dilation)
        h_conv2 = conv_norm_relu(inlayer=h_conv1, shape=shapes[1], dilation=dilation)
        
        if cbr_layers == SMALL_LAYER:
            return max_pool(h_conv2, strides=[1, 1], dilation=dilation)
        elif cbr_layers == LARGE_LAYER:
            h_conv3 = conv_norm_relu(inlayer=h_conv2, shape=shapes[2], dilation=dilation)
            return max_pool(h_conv3, strides=[1, 1], dilation=dilation)
        else:
            raise ValueError('Illegal number of Conv/Batch/ReLU layers')

    def _upsample_layer(self, inlayer, shapes, dilation=1, cbr_layers=2):
        h_unpool = tf.image.resize_images(inlayer, shapes[0][-2:], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
        h_conv1 = conv_norm_relu(inlayer=h_unpool, shape=shapes[0], dilation=dilation)
        h_conv2 = conv_norm_relu(inlayer=h_conv1, shape=shapes[1], dilation=dilation)

        if cbr_layers == SMALL_LAYER:
            return h_conv2
        elif cbr_layers == LARGE_LAYER:
            h_conv3 = conv_norm_relu(inlayer=h_conv2, shape=shapes[2], dilation=dilation)
            return h_conv3
        else:
            raise ValueError('Illegal number of Conv/Batch/ReLU layers')