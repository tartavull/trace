import tensorflow as tf

from common import conv2d, bias_variable, weight_variable, max_pool, conv_norm_relu

SMALL_LAYER = 2
LARGE_LAYER = 3

# taken from https://arxiv.org/pdf/1511.00561.pdf
def default_SegNet():
    params = {
        'd1': 48,
        'd2': 48,
        'd3': 48,
        'd4': 48,
        'd5': 48,

        'u1': 48,
        'u2': 48,
        'u3': 48,
        'u4': 48,
        'u5': 48,

        'fc': 200,
        'lr': 0.001,
        'out': 101
    }
    return SegNet(params)


class SegNet:
    def __init__(self, params):

        # Hyperparameters
        down_1 = params['d1']
        down_2 = params['d2']
        down_3 = params['d3']
        down_4 = params['d4']
        down_5 = params['d5']

        up_1 = params['u1']
        up_2 = params['u2']
        up_3 = params['u3']
        up_4 = params['u4']
        up_5 = params['u5']

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
            shape=[4, 4, 1, m1], dilation=1, cbr_layers=2)

        # layer set 2: 2 Conv/Batch/ReLU and 1 pool
        down_conv2 = self._downsample_layer(inlayer=down_conv1, 
            shape=[4, 4, m1, m2], dilation=2, cbr_layers=2)

        # layer set 3: 3 Conv/Batch/ReLU and 1 pool
        down_conv3 = self._downsample_layer(inlayer=down_conv2, 
            shape=[4, 4, m2, m3], dilation=4, cbr_layers=3)

        # layer set 4: 3 Conv/Batch/ReLU and 1 pool
        down_conv4 = self._downsample_layer(inlayer=down_conv3, 
            shape=[4, 4, m3, m4], dilation=8, cbr_layers=3)

        # layer set 5: 3 Conv/Batch/ReLU and 1 pool
        down_conv5 = self._downsample_layer(inlayer=down_conv4, 
            shape=[4, 4, m4, m5], dilation=16, cbr_layers=3)

        # layer set 6: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        

        # layer set 7: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        

        # layer set 8: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        
        
        # layer set 9: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        

        # layer set 10: 1 unpool (upsampling) and 3 Conv/Batch/ReLU
        
        
        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, fc, 2])
        b_fc2 = bias_variable([2])
        self.prediction = conv2d(h_fc1, W_fc2, dilation=16) + b_fc2

        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.prediction, self.target))
        self.loss_summary = tf.scalar_summary('cross_entropy', self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        self.binary_prediction = tf.round(self.sigmoid_prediction)
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

    def _downsample_layer(self, inlayer=self.image, 
        shape=[4, 4, 1, 48], dilation=1, cbr_layers=2):

        h_conv1 = conv_norm_relu(inlayer=inlayer, shape=shape, dilation=dilation)
        h_conv2 = conv_norm_relu(inlayer=h_conv1, shape=shape, dilation=dilation)
        
        if cbr_layers == SMALL_LAYER:
            return max_pool(h_conv2, strides=[1, 1], dilation=dilation)
        elif cbr_layers == LARGE_LAYER:
            h_conv3 = conv_norm_relu(inlayer=h_conv2, shape=shape, dilation=dilation)
            return max_pool(h_conv3, strides=[1, 1], dilation=dilation)
        else:
            raise ValueError('Incorrect number of Conv/Batch/ReLU layers')

    def _upsample_layer(self):
        h_conv14  = conv_norm_relu(inlayer=h_unpool1, shape=[4, 4, 1, map_1], dilation=16)
        h_conv15  = conv_norm_relu(inlayer=h_conv14, shape=[4, 4, 1, map_2], dilation=16)
        h_conv16  = conv_norm_relu(inlayer=h_conv15, shape=[4, 4, 1, map_2], dilation=16)