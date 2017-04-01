import em_dataset as em
import models.conv_net as conv
import tensorflow as tf


class InferenceParams(object):
    def __init__(self, patch_shape):
        """Constructor for InferenceParams

        :param patch_shape: Shape of the patch used for inference
        """
        self.patch_shape = patch_shape


class TrainingParams(object):
    def __init__(self, optimizer, learning_rate, n_iterations, patch_shape, batch_size):
        """Constructor for TrainingParams

        :param optimizer: Optimizer used for learning
        :param learning_rate: Learning rate for the optimizer
        :param n_iterations: Number of iterations for training
        :param patch_shape: Shape of the output patch, a list of form [z_size, y_size, x_size]
        :param batch_size: Number of examples per batch
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.patch_shape = patch_shape
        self.z_output_shape, self.y_output_shape, self.x_output_shape = patch_shape[0], patch_shape[1], patch_shape[2]
        self.batch_size = batch_size


class PipelineConfig(object):
    '''Everything you might need to define a complete training and inference pipeline

    '''

    def __init__(self, data_path, dataset_constructor, model_constructor, model_arch, training_params,
                 inference_params, pipeline_name=None):
        """

        :param data_path: Path where the training data is located
        :param dataset_constructor: Constructor for the type of dataset proposed
        :param model_constructor: Model constructor (i.e. ConvNet, UNet, etc.)
        :param model_arch: Specific architecture for a model class
        :param training_params: Parameters used for training
        :param inference_params: Parameters used for inference
        :param pipeline_name: Name of the pipeline. When None is specified, use the architecture name.
        """
        self.data_path = data_path
        self.dataset_constructor = dataset_constructor
        self.model_constructor = model_constructor
        self.model_arch = model_arch
        self.training_params = training_params
        self.inference_params = inference_params
        self.pipeline_name = pipeline_name

        if self.pipeline_name is None:
            self.pipeline_name = model_arch.model_name


vd2d_pipeline = PipelineConfig(
    data_path='./isbi/',
    dataset_constructor=em.ISBIDataset,
    model_constructor=conv.ConvNet,
    model_arch=conv.VD2D,
    training_params=TrainingParams(
        optimizer=tf.train.AdamOptimizer,
        learning_rate=0.0001,
        n_iterations=10000,
        patch_shape=[1, 160, 160],
        batch_size=1,
    ),
    inference_params=InferenceParams(
        patch_shape=[1, 512, 512]
    )
)
