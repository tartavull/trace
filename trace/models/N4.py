import tensorflow as tf

from common import conv2d, bias_variable, weight_variable, max_pool
from params import N4_params

def default_N4():
    required =['m1', 'm2', 'm3', 'm4', 'fc', 'lr']
    for param in required:
        if param not in N4_params:
            raise ValueError('Incorrect parameter map (Missing \
                parameter: {})'.format(param))
    return N4(N4_params)

class N4:
    def __init__(self, params):

        # Hyperparameters
        map_1 = params['m1']
        map_2 = params['m2']
        map_3 = params['m3']
        map_4 = params['m4']
        fc = params['fc']
        learning_rate = params['lr']

        self.out = params['out']
        self.fov = 95
        self.inpt = self.fov + 2 * (self.out // 2)

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        # layer 1 - original stride 1
        W_conv1  = weight_variable([4, 4, 1, map_1])
        s_conv1  = tf.matmul(self.image, W_conv1)

        mean_bn1, var_bn1 = tf.nn.moments(s_conv1, [0])
        offset_bn1 = tf.Variable(tf.zeros([map_1]))
        scale_bn1  = tf.Variable(tf.ones([map_1]))
        bn_conv1 = tf.nn.batch_normalization(s_conv1, mean=mean_bn1, variance=var_bn1, 
            offset=offset_bn1, scale=scale_bn1, variance_epsilon=.0005)
        h_conv1 = tf.nn.relu(bn_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, map_1, map_2])
        s_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2))

        mean_bn2, var_bn2 = tf.nn.moments(s_conv2, [0])
        offset_bn2 = tf.Variable(tf.zeros([map_2]))
        scale_bn2  = tf.Variable(tf.ones([map_2]))
        bn_conv2 = tf.nn.batch_normalization(s_conv2, mean=mean_bn2, variance=var_bn2, 
            offset=offset_bn2, scale=scale_bn2, variance_epsilon=.0005)
        h_conv2  = tf.nn.relu(s_conv2)
        
        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, map_2, map_3])
        s_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4))

        mean_bn3, var_bn3 = tf.nn.moments(s_conv3, [0])
        offset_bn3 = tf.Variable(tf.zeros([map_3]))
        scale_bn3  = tf.Variable(tf.ones([map_3]))
        bn_conv3 = tf.nn.batch_normalization(s_conv3, mean=mean_bn3, variance=var_bn3, 
            offset=offset_bn3, scale=scale_bn3, variance_epsilon=.0005)
        h_conv3 = tf.nn.relu(s_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=4)

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, map_3, map_4])
        s_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8))

        mean_bn4, var_bn4 = tf.nn.moments(s_conv4, [0])
        offset_bn4 = tf.Variable(tf.zeros([map_4]))
        scale_bn4  = tf.Variable(tf.ones([map_4]))
        bn_conv4 = tf.nn.batch_normalization(s_conv4, mean=mean_bn4, variance=var_bn4, 
            offset=offset_bn4, scale=scale_bn4, variance_epsilon=.0005)
        h_conv4 = tf.nn.relu(s_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1, 1], dilation=8)

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, map_4, fc])
        b_fc1 = bias_variable([fc])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=16) + b_fc1)

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
