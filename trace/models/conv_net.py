import tensorflow as tf


class Layer(object):
    def __init__(self, filter_size, n_feature_maps, activation_fn=lambda x: x):
        self.filter_size = filter_size
        self.n_feature_maps = n_feature_maps
        self.activation_fn = activation_fn

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        raise NotImplementedError("Abstract Class!")


class Conv2DLayer(Layer):
    layer_type = 'conv2d'

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Create the tensorflow variables
        filters_shape = [self.filter_size, self.filter_size, prev_n_feature_maps, self.n_feature_maps]
        filters = tf.Variable(tf.truncated_normal(filters_shape, stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[self.n_feature_maps]))

        # Perform a dilated convolution on the previous layer, where the dilation rate is dependent on the
        # number of poolings so far.
        convolution = tf.nn.convolution(prev_layer, filters, strides=[1, 1], padding='VALID',
                                        dilation_rate=[dilation_rate, dilation_rate])

        # Apply the activation function
        output_layer = self.activation_fn(convolution + bias)

        # Prepare the next values in the loop
        return output_layer, self.n_feature_maps


class PoolLayer(Layer):
    layer_type = 'pool'

    def __init__(self, filter_size):
        super(PoolLayer, self).__init__(filter_size, 0)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):
        # Max pool
        output_layer = tf.nn.pool(prev_layer, window_shape=[self.filter_size, self.filter_size],
                                  dilation_rate=[dilation_rate, dilation_rate], strides=[1, 1],
                                  padding='VALID',
                                  pooling_type='MAX')

        return output_layer, prev_n_feature_maps


class BNLayer(Layer):
    layer_type = 'bn_conv2d'

    def __init__(self, activation_fn=lambda x: x):
        super(BNLayer, self).__init__(1, 0, activation_fn)

    def connect(self, prev_layer, prev_n_feature_maps, dilation_rate, is_training):

        # Apply batch normalization
        bn_conv = tf.contrib.layers.batch_norm(prev_layer, center=True, scale=True, is_training=is_training, scope='bn')

        # Apply the activation function
        output_layer = self.activation_fn(bn_conv)

        # Prepare the next values in the loop
        return output_layer, prev_n_feature_maps


class ConvArchitecture:
    def __init__(self, model_name, layers):
        self.model_name = model_name
        self.layers = layers

        n_poolings = 0
        self.receptive_field = 1

        for layer in self.layers:

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            if type(layer) is PoolLayer:
                n_poolings += 1

            # Calculate the receptive field
            self.receptive_field = (layer.filter_size * self.receptive_field) - (self.receptive_field - dilation_rate) * (layer.filter_size - 1)

    def print_arch(self):

        print(self.model_name)

        n_poolings = 0
        receptive_field = 1
        layer_num = 0

        for layer in self.layers:

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            if type(layer) is PoolLayer:
                n_poolings += 1

            # Calculate the receptive field
            receptive_field = (layer.filter_size * receptive_field) - (receptive_field - dilation_rate) * \
                                                                      (layer.filter_size - 1)

            layer_num += 1
            print("Layer %d,\ttype: %s,\tfilter: [%d, %d],\tFOV: %d" % (layer_num, layer['type'], layer.filter_size, layer.filter_size, receptive_field))

N4 = ConvArchitecture(
    model_name='n4',
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=1, n_feature_maps=2),
    ]
)

N4_WIDENED = ConvArchitecture(
    model_name='N4_widened',
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=128, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=1, n_feature_maps=2),
    ]
)

N4_DEEPER = ConvArchitecture(
    model_name='N4_deeper',
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=4, n_feature_maps=250, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=300, activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=400, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=1, n_feature_maps=2),
    ]
)


VD2D = ConvArchitecture(
    model_name='VD2D',
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24, activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=1, n_feature_maps=2),
    ]
)

BN_VD2D = ConvArchitecture(
    model_name='bn_VD2D',
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2), BNLayer(),
    ]
)

BN_VD2D_RELU = ConvArchitecture(
    model_name='bn_VD2D_relu',
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2), BNLayer(),
    ]
)


class ConvNet:
    def __init__(self, architecture, is_training=False):

        # Save the architecture
        self.architecture = architecture
        self.model_name = self.architecture.model_name
        self.fov = architecture.receptive_field

        # Define the inputs
        self.image = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.target = tf.placeholder(tf.float32, shape=[None, None, None, 2])

        n_poolings = 0

        prev_layer = self.image
        prev_n_feature_maps = 1

        layer_num = 0

        for layer in architecture.layers:

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            with tf.variable_scope('layer' + str(layer_num)):

                prev_layer, prev_n_feature_maps = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate, is_training)

                if type(layer) is PoolLayer:
                    n_poolings += 1

            layer_num += 1


        # Predictions
        self.prediction = tf.nn.sigmoid(prev_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prev_layer, self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()
