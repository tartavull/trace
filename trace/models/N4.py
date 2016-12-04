import tensorflow as tf

from common import conv2d, bias_variable, weight_variable, max_pool


def default_N4():
    params = {
        'm1': 48,
        'm2': 48,
        'm3': 48,
        'm4': 48,
        'fc': 200,
        'lr': 0.001,
        'out': 101
    }
    return N4(params)


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
        W_conv1 = weight_variable([4, 4, 1, map_1])
        b_conv1 = bias_variable([map_1])
        h_conv1 = tf.nn.relu(conv2d(self.image, W_conv1, dilation=1) + b_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, map_1, map_2])
        b_conv2 = bias_variable([map_2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=2) + b_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=2)

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, map_2, map_3])
        b_conv3 = bias_variable([map_3])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=4) + b_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=4)

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, map_3, map_4])
        b_conv4 = bias_variable([map_4])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=8) + b_conv4)

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


class N5:
    def __init__(self, params):

        # Hyperparameters
        map_pre = 48
        map_1 = params['m1']
        map_2 = params['m2']
        map_3 = params['m3']
        map_4 = params['m4']
        map_5 = params['m5']
        fc = params['fc']
        learning_rate = params['lr']
        self.out = params['out']
        self.fov = 95
        self.inpt = self.fov + 2 * (self.out // 2)

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        dilations = [1, 3, 9, 27, 81]

        W_conv_pre = weight_variable([3,3,1,map_pre])
        b_conv_pre = bias_variable([map_pre])
        h_conv_pre = tf.nn.relu(conv2d(self.image, W_conv_pre, dilation=1, padding='SAME') + b_conv_pre)

        # layer 1 - original stride 1
        W_conv1 = weight_variable([4, 4, map_pre, map_1])
        b_conv1 = bias_variable([map_1])
        h_conv1 = tf.nn.relu(conv2d(h_conv_pre, W_conv1, dilation=dilations[0]) + b_conv1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(h_conv1, strides=[1, 1], dilation=dilations[0])

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, map_1, map_2])
        b_conv2 = bias_variable([map_2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, dilation=dilations[1]) + b_conv2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(h_conv2, strides=[1, 1], dilation=dilations[1])

        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, map_2, map_3])
        b_conv3 = bias_variable([map_3])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, dilation=dilations[2]) + b_conv3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(h_conv3, strides=[1, 1], dilation=dilations[2])

        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, map_3, map_4])
        b_conv4 = bias_variable([map_4])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, dilation=dilations[3]) + b_conv4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(h_conv4, strides=[1, 1], dilation=dilations[3])

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, map_4, fc])
        b_fc1 = bias_variable([fc])
        h_fc1 = tf.nn.relu(conv2d(h_pool4, W_fc1, dilation=dilations[4]) + b_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, fc, 2])
        b_fc2 = bias_variable([2])

        prediction = conv2d(h_fc1, W_fc2, dilation=dilations[4]) + b_fc2

##TODO HAVE TO TRACK MEANS THROUGHUOT TRAINING
class N4_BN:
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
        epsilon = 1e-3 #batch normalization

        # layer 0
        # Normally would have shape [1, inpt, inpt, 1], but None allows us to have a flexible validation set
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        # should this be before layer 1? I'm not sure
        batch_mean0, batch_var0 = tf.nn.moments(self.image,[0])
        scale0 = tf.Variable(tf.ones([1])) # Is 1 the correct value?
        beta0 = tf.Variable(tf.zeros([1]))
        BN0 = tf.nn.batch_normalization(self.image,batch_mean0,batch_var0,beta0,scale0,epsilon)

        # layer 1 - original stride 1
        W_conv1 = weight_variable([4, 4, 1, map_1])
        b_conv1 = bias_variable([map_1])
        h_conv1 = conv2d(BN0, W_conv1, dilation=1) + b_conv1
        # BN before relu and pooling
        batch_mean1, batch_var1 = tf.nn.moments(h_conv1,[0])
        scale1 = tf.Variable(tf.ones([map_1]))
        beta1 = tf.Variable(tf.zeros([map_1]))
        BN1 = tf.nn.batch_normalization(h_conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
        relu1 = tf.nn.relu(BN1)

        # layer 2 - original stride 2
        h_pool1 = max_pool(relu1, strides=[1, 1], dilation=1)

        # layer 3 - original stride 1
        W_conv2 = weight_variable([5, 5, map_1, map_2])
        b_conv2 = bias_variable([map_2])
        h_conv2 = conv2d(h_pool1, W_conv2, dilation=2) + b_conv2

        batch_mean2, batch_var2 = tf.nn.moments(h_conv2,[0])
        scale2 = tf.Variable(tf.ones([map_2]))
        beta2 = tf.Variable(tf.zeros([map_2]))
        BN2 = tf.nn.batch_normalization(h_conv2,batch_mean2,batch_var2,beta2,scale2,epsilon)
        relu2 = tf.nn.relu(BN2)

        # layer 4 - original stride 2
        h_pool2 = max_pool(relu2, strides=[1, 1], dilation=2)


        # layer 5 - original stride 1
        W_conv3 = weight_variable([4, 4, map_2, map_3])
        b_conv3 = bias_variable([map_3])
        h_conv3 = conv2d(h_pool2, W_conv3, dilation=4) + b_conv3

        batch_mean3, batch_var3 = tf.nn.moments(h_conv3,[0])
        scale3 = tf.Variable(tf.ones([map_3]))
        beta3 = tf.Variable(tf.zeros([map_3]))
        BN3 = tf.nn.batch_normalization(h_conv3,batch_mean3,batch_var3,beta3,scale3,epsilon)
        relu3 = tf.nn.relu(BN3)

        # layer 6 - original stride 2
        h_pool3 = max_pool(relu3, strides=[1, 1], dilation=4)


        # layer 7 - original stride 1
        W_conv4 = weight_variable([4, 4, map_3, map_4])
        b_conv4 = bias_variable([map_4])
        h_conv4 = conv2d(h_pool3, W_conv4, dilation=8) + b_conv4
        
        batch_mean4, batch_var4 = tf.nn.moments(h_conv4,[0])
        scale4 = tf.Variable(tf.ones([map_4]))
        beta4 = tf.Variable(tf.zeros([map_3]))
        BN4 = tf.nn.batch_normalization(h_conv4,batch_mean4,batch_var4,beta4,scale4,epsilon)
        relu4 = tf.nn.relu(B4)

        # layer 8 - original stride 2
        h_pool4 = max_pool(relu4, strides=[1, 1], dilation=8)

        # layer 9 - original stride 1
        W_fc1 = weight_variable([3, 3, map_4, fc])
        b_fc1 = bias_variable([fc])
        h_fc1 conv2d(BN4, W_fc1, dilation=16) + b_fc1

        batch_mean_fc1, batch_var_fc1 = tf.nn.moments(h_fc1,[0])
        scale_fc1 = tf.Variable(tf.ones([fc]))
        beta_fc1 = tf.Variable(tf.zeros([fc]))
        BN_fc1 = tf.nn.batch_normalization(h_fc1,batch_mean_fc1,batch_var_fc1,beta_fc1,scale_fc1,epsilon)
        relu_fc = tf.nn.relu(BN_fc1)

        # layer 10 - original stride 2
        W_fc2 = weight_variable([1, 1, fc, 2])
        b_fc2 = bias_variable([2])
        self.prediction = conv2d(relu_fc, W_fc2, dilation=16) + b_fc2


        self.sigmoid_prediction = tf.nn.sigmoid(self.prediction)
        # Replace with BN_fc2?
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


















