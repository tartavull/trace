from .common import *
from utils import *
from augmentation import *



class ConvArchitecture(Architecture):
    def __init__(self, model_name, output_mode, layers, architecture_type='2D'):
        super(ConvArchitecture, self).__init__(model_name, output_mode, architecture_type)

        self.layers = layers

        n_poolings = 0
        self.receptive_field = 1
        self.z_receptive_field = 1

        dilation_rate = 1
        z_dilation_rate = 1

        for layer in self.layers:

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            if issubclass(type(layer), PoolLayer):
                n_poolings += 1

            # Calculate the receptive field
            self.receptive_field += dilation_rate * (layer.filter_size - 1)
            if self.architecture_type == '3D':
                self.z_receptive_field += z_dilation_rate * (layer.z_filter_size - 1)

        self.fov = self.receptive_field
        self.z_fov = self.z_receptive_field

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
            receptive_field = (layer.filter_size * receptive_field) - (receptive_field - dilation_rate) * \
                                                                      (layer.filter_size - 1)

            layer_num += 1
            print("Layer %d,\ttype: %s,\tfilter: [%d, %d],\tFOV: %d" % (layer_num, layer.layer_type, layer.filter_size,
                                                                        layer.filter_size, receptive_field))


N4 = ConvArchitecture(
    model_name='n4',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

N4_3D = ConvArchitecture(
    model_name='n4_3D',
    output_mode=AFFINITIES_3D,
    layers=[
        Conv3DLayer(filter_size=4, z_filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool3DLayer(filter_size=2),
        Conv3DLayer(filter_size=5, z_filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool3DLayer(filter_size=2),
        Conv3DLayer(filter_size=4, z_filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool3DLayer(filter_size=2),
        Conv3DLayer(filter_size=4, z_filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool3DLayer(filter_size=2),
        Conv3DLayer(filter_size=3, z_filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv3DLayer(filter_size=1, z_filter_size=1, n_feature_maps=3, is_valid=True),
    ],
    architecture_type='3D'
)

N4_WIDENED = ConvArchitecture(
    model_name='N4_widened',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=128, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

N4_DEEPER = ConvArchitecture(
    model_name='N4_deeper',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=4, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=5, n_feature_maps=68, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=100, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=4, n_feature_maps=250, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=4, n_feature_maps=300, activation_fn=tf.nn.relu, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=400, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

VD2D = ConvArchitecture(
    model_name='VD2D',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=2, n_feature_maps=24, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=2, is_valid=True),
    ]
)

VD2D_BOUNDARIES = ConvArchitecture(
    model_name='VD2D_boundaries',
    output_mode=BOUNDARIES,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=24, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=2, n_feature_maps=24, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=36, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=48, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=3, n_feature_maps=60, activation_fn=tf.nn.tanh, is_valid=True),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, activation_fn=tf.nn.relu, is_valid=True),
        Conv2DLayer(filter_size=1, n_feature_maps=1, is_valid=True),
    ]
)

BN_VD2D = ConvArchitecture(
    model_name='bn_VD2D',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.tanh),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2, is_valid=True), BNLayer(),
    ]
)

BN_VD2D_RELU = ConvArchitecture(
    model_name='bn_VD2D_relu',
    output_mode=AFFINITIES_2D,
    layers=[
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=2, n_feature_maps=24, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=36, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=48, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=60, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Pool2DLayer(filter_size=2),
        Conv2DLayer(filter_size=3, n_feature_maps=200, is_valid=True), BNLayer(activation_fn=tf.nn.relu),
        Conv2DLayer(filter_size=3, n_feature_maps=2, is_valid=True), BNLayer(),
    ]
)


class ConvNet(Model):
    def __init__(self, architecture, is_training=False):
        super(ConvNet, self).__init__(architecture)

        n_poolings = 0

        prev_layer = self.image
        prev_n_feature_maps = 1

        z_dilation_rate = 1

        for layer_num, layer in enumerate(self.architecture.layers):

            # Double the dilation rate for a given layer every time we pool.
            dilation_rate = 2 ** n_poolings

            with tf.variable_scope('layer' + str(layer_num)):
                layer.depth = layer_num
                prev_layer, prev_n_feature_maps = layer.connect(prev_layer, prev_n_feature_maps, dilation_rate,
                                                                is_training, z_dilation_rate=z_dilation_rate)

                if issubclass(type(layer), PoolLayer):
                    n_poolings += 1

        # Predictions
        self.prediction = tf.nn.sigmoid(prev_layer)
        self.binary_prediction = tf.round(self.prediction)

        # Loss
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prev_layer,
                                                                                    labels=self.target))
        self.pixel_error = tf.reduce_mean(tf.cast(tf.abs(self.binary_prediction - self.target), tf.float32))

        self.saver = tf.train.Saver()

    # def predict_with_evaluation(self, session, inputs, metrics, labels, pred_batch_shape, mirror_inputs=True):
    #     if mirror_inputs:
    #         inputs = mirror_across_borders_3d(inputs, self.fov, self.z_fov)


    def predict(self, session, inputs, pred_batch_shape, mirror_inputs=True):
        if mirror_inputs:
            inputs = mirror_across_borders_3d(inputs, self.fov, self.z_fov)

        return self.__predict_with_evaluation(session, inputs, None, pred_batch_shape, mirror_inputs)

    def __predict_with_evaluation(self, session, inputs, metrics, pred_batch_shape, mirror_inputs=True):
        # Extract the tile sizes from the argument
        z_patch, y_patch, x_patch = pred_batch_shape[0], pred_batch_shape[1], pred_batch_shape[2]
        z_inp_patch, y_inp_patch, x_inp_patch = z_patch + self.z_fov - 1, y_patch + self.fov - 1, x_patch + self.fov - 1

        # Extract the input size, so we can reduce to output
        z_inp_size, y_inp_size, x_inp_size = inputs.shape[1], inputs.shape[2], inputs.shape[3]

        # Create a holder for the output
        all_preds = np.zeros(
            shape=[inputs.shape[0], z_inp_size - self.z_fov + 1, y_inp_size - self.fov + 1,
                   x_inp_size - self.fov + 1,
                   self.architecture.n_outputs], dtype=np.float16)

        for i, _ in enumerate(inputs):

            # Iterate over each batch
            for z in range(0, all_preds.shape[1], z_patch):
                for y in range(0, all_preds.shape[2], y_patch):
                    for x in range(0, all_preds.shape[3], x_patch):
                        print('z=%d, y=%d, x=%d' % (z, y, x))
                        # Get the appropriate patch
                        input_image = np.expand_dims(inputs[i,
                                                            z: z + z_inp_patch,
                                                            y: y + y_inp_patch,
                                                            x: x + x_inp_patch,
                                                            :], axis=0)

                        pred = session.run(self.prediction, feed_dict={self.example: input_image})

                        # Fill in the output
                        all_preds[i, z: z + z_patch, y: y + y_patch, x: x + x_patch, :] = pred

        return all_preds
