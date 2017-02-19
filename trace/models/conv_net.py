import em_dataset as em

from .common import *


class ConvArchitecture(Architecture):
    def __init__(self, model_name, output_mode, layers):
        super(ConvArchitecture, self).__init__(model_name, output_mode)

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

            if layer.layer_type == 'pool':
                n_poolings += 1

            # Calculate the receptive field
            receptive_field = (layer.filter_size * receptive_field) - (receptive_field - dilation_rate) * (layer.filter_size - 1)

            layer_num += 1
            print("Layer %d,\ttype: %s,\tfilter: [%d, %d],\tFOV: %d" % (layer_num, layer.layer_type, layer.filter_size, layer.filter_size, receptive_field))

N4 = ConvArchitecture(
    model_name='n4',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

N4_WIDENED = ConvArchitecture(
    model_name='N4_widened',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=128, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

N4_DEEPER = ConvArchitecture(
    model_name='N4_deeper',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=4, n_feature_maps=250, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=300, activation_fn=tf.nn.relu, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=400, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)


VD2D = ConvArchitecture(
    model_name='VD2D',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=2, n_feature_maps=24, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

VD2D_BOUNDARIES = ConvArchitecture(
    model_name='VD2D_boundaries',
    output_mode=em.BOUNDARIES_MODE,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=2, n_feature_maps=24, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.tanh, is_valid=True),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=1, is_valid=True),
    ]
)

BN_VD2D = ConvArchitecture(
    model_name='bn_VD2D',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2, is_valid=True), BNLayer(),
    ]
)

BN_VD2D_RELU = ConvArchitecture(
    model_name='bn_VD2D_relu',
    output_mode=em.AFFINITIES_2D_MODE,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        PoolLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2, is_valid=True), BNLayer(),
    ]
)





class ConvNet(Model):
    def __init__(self, architecture, is_training=False):
        super(ConvNet, self).__init__(architecture)

        # Standardize each input image, using map because per_image_standardization takes one image at a time
        standardized_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.image)

        n_poolings = 0

        prev_layer = standardized_image
        prev_n_feature_maps = 1

        for layer_num, layer in enumerate(self.architecture.layers):

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            with tf.variable_scope('layer' + str(layer_num)):
                layer.depth = layer_num
                prev_layer, prev_n_feature_maps = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate, is_training)

                if type(layer) is PoolLayer:
                    n_poolings += 1

        # Predictions
        self.prediction = tf.nn.sigmoid(prev_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prev_layer, self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()
