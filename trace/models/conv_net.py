import tensorflow as tf

DEFAULT_PARAMS = {
    'model_name': 'N4_widened',
    'fov': 95,  # Sanity check
    'input': 195,
    'output': 101,
    'learning_rate': 0.0001,
    'layers': [
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 48, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 5, 'n_feature_maps': 68, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 100, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 128, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 200, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 1, 'n_feature_maps': 2, 'activation_fn': lambda x: x},
    ]
}

DEEPER_PARAMS = {
    'model_name': 'N4_widened',
    'fov': 95,  # Sanity check
    'input': 195,
    'output': 101,
    'learning_rate': 0.0001,
    'layers': [
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 48, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 5, 'n_feature_maps': 68, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 100, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 200, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 250, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 4, 'n_feature_maps': 300, 'activation_fn': tf.nn.relu},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 400, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 1, 'n_feature_maps': 2, 'activation_fn': lambda x: x},
    ]
}

VD2D = {
    'model_name': 'VD2D',
    'fov': 109,  # Sanity check
    'input': 209,
    'output': 101,
    'learning_rate': 0.001,
    'layers': [
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 24, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 24, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 2, 'n_feature_maps': 24, 'activation_fn': tf.nn.tanh},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 36, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 36, 'activation_fn': tf.nn.tanh},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 48, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 48, 'activation_fn': tf.nn.tanh},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 60, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 60, 'activation_fn': tf.nn.tanh},
        {'type': 'pool', 'filter_size': 2},
        {'type': 'conv2d', 'filter_size': 3, 'n_feature_maps': 200, 'activation_fn': tf.nn.relu},
        {'type': 'conv2d', 'filter_size': 1, 'n_feature_maps': 2, 'activation_fn': lambda x: x},
    ]

}


class ConvNet:
    def __init__(self, params):

        print(params['model_name'])

        learning_rate = params['learning_rate']

        # Define the inputs
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        n_poolings = 0
        receptive_field = 1

        prev_layer = self.image
        prev_n_feature_maps = 1

        layer_num = 0

        for layer in params['layers']:

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            # Convolutional layer
            if layer['type'] == 'conv2d':
                # Extract parameters
                filter_size = layer['filter_size']
                n_feature_maps = layer['n_feature_maps']
                activation_fn = layer['activation_fn']

                # Create the tensorflow variables
                filters_shape = [filter_size, filter_size, prev_n_feature_maps, n_feature_maps]
                filters = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1))
                bias = tf.Variable(tf.constant(0.1, shape=[n_feature_maps]))

                # Perform a dilated convolution on the previous layer, where the dilation rate is dependent on the
                # number of poolings so far.
                convolution = tf.nn.convolution(prev_layer, filters, strides=[1, 1], padding='VALID',
                                                dilation_rate=[dilation_rate, dilation_rate])

                # Apply the activation function
                output_layer = activation_fn(convolution + bias)

                # Prepare the next values in the loop
                prev_layer = output_layer
                prev_n_feature_maps = n_feature_maps
                receptive_field = (filter_size * receptive_field) - (receptive_field - dilation_rate) * (
                    filter_size - 1)

            elif layer['type'] == 'pool':
                filter_size = layer['filter_size']
                # Max pool
                output_layer = tf.nn.pool(prev_layer, window_shape=[filter_size, filter_size],
                                          dilation_rate=[dilation_rate, dilation_rate], strides=[1, 1], padding='VALID',
                                          pooling_type='MAX')

                prev_layer = output_layer
                n_poolings += 1
                receptive_field = (filter_size * receptive_field) - (receptive_field - dilation_rate) * (
                    filter_size - 1)

            # Debugging
            layer_num += 1
            print("Layer %d,\ttype: %s,\tfilter: [%d, %d],\tFOV: %d" %
                  (layer_num, layer['type'], filter_size, filter_size, receptive_field))

        assert (receptive_field == params['fov'])

        self.fov = receptive_field
        self.input = params['input']
        self.output = params['output']

        # Predictions
        self.prediction = tf.nn.sigmoid(prev_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Evaluation
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prev_layer, self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        # Placeholders that we will input externally, but useful when visualizing on tensorboard
        self.rand_f_score = tf.placeholder(tf.float32)
        self.rand_f_score_merge = tf.placeholder(tf.float32)
        self.rand_f_score_split = tf.placeholder(tf.float32)
        self.vi_f_score = tf.placeholder(tf.float32)
        self.vi_f_score_merge = tf.placeholder(tf.float32)
        self.vi_f_score_split = tf.placeholder(tf.float32)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        # Summary operations
        self.training_summaries = tf.summary.merge(
            [tf.summary.scalar('cross_entropy', self.cross_entropy),
             tf.summary.scalar('pixel_error', self.pixel_error),
             ])

        self.validation_summaries = tf.summary.merge(
            [tf.summary.scalar('rand_score', self.rand_f_score),
             tf.summary.scalar('rand_merge_score', self.rand_f_score_merge),
             tf.summary.scalar('rand_split_score', self.rand_f_score_split),
             tf.summary.scalar('vi_score', self.vi_f_score),
             tf.summary.scalar('vi_merge_score', self.vi_f_score_merge),
             tf.summary.scalar('vi_split_score', self.vi_f_score_split),
             ])

        self.saver = tf.train.Saver()
        self.model_name = params['model_name']
