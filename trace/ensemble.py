import models as conv_net
import tensorflow as tf
import learner
import numpy as np
import utils

import em_dataset as em


class ComponentParams:
    def __init__(self, id, model, architecture, training_params):
        self.id = id
        self.model = model
        self.architecture = architecture
        self.training_params = training_params


N4_3_TEST = [
    ComponentParams(
        id='n4_1',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=500,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_2',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=500,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_3',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=500,
            output_size=101,
        )
    )
]

N4_3 = [
    ComponentParams(
        id='n4_1',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=30000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_2',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=30000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='n4_3',
        model=conv_net.ConvNet,
        architecture=conv_net.N4,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=30000,
            output_size=101,
        )
    )
]

VD2D_5 = [
    ComponentParams(
        id='vd2d_1',
        model=conv_net.ConvNet,
        architecture=conv_net.VD2D,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_2',
        model=conv_net.ConvNet,
        architecture=conv_net.VD2D,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_3',
        model=conv_net.ConvNet,
        architecture=conv_net.VD2D,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_4',
        model=conv_net.ConvNet,
        architecture=conv_net.VD2D,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=50000,
            output_size=101,
        )
    ),
    ComponentParams(
        id='vd2d_5',
        model=conv_net.ConvNet,
        architecture=conv_net.VD2D,
        training_params=learner.TrainingParams(
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            n_iter=50000,
            output_size=101,
        )
    ),
]

ENSEMBLE_PARAMS_DICT = {
    'n4_3': N4_3,
    'n4_3_test': N4_3_TEST,
    'vd2d_5': VD2D_5,
}


class ModelAverager:
    name = 'model_averaging'

    def __init__(self, ensembler_folder):
        self.ensembler_folder = ensembler_folder

    # No training needed
    def train(self, classifiers):
        pass

    def predict(self, predictions):
        with tf.Session() as sess:
            # Initialize the variable
            sess.run(tf.global_variables_initializer())

            # num models, num slices, dim, dim, channels
            outputs = tf.placeholder(tf.float32, shape=[None, None, None, None, None])

            # Take the average of the outputs along the first dimension
            averaged = tf.reduce_mean(outputs, axis=0)

            # Run the prediction
            result = sess.run(averaged, feed_dict={outputs: np.asarray(predictions)})

        return result


ENSEMBLE_METHOD_DICT = {
    ModelAverager.name: ModelAverager
}


class EnsembleLearner:
    def __init__(self, model_configs, model_configs_name, ensemble_method_constructor, data_folder, run_name):
        self.model_configs = model_configs
        self.data_folder = data_folder

        self.results_folder = data_folder + 'results/ensemble/' + ensemble_method_constructor.name + '/' \
                              + model_configs_name + '/run-' + run_name + '/'
        self.ensembler_folder = self.results_folder + 'ensembler/'

        self.ensemble_method = ensemble_method_constructor(self.ensembler_folder)

    def train(self, dataset):

        # Keep track of every ongoing session
        classifiers = []

        # Train every model
        for config in self.model_configs:
            # Store each model copy in its own folder
            ckpt_folder = self.results_folder + config.id + '/'

            # Reset the default graph so that we emit the same variables
            # DON'T DO THIS IN PREDICTION TIME, AS IT WILL DESTROY ALL OTHER GRAPHS
            tf.reset_default_graph()

            # Create the classifier and persist it
            model = config.model(config.architecture)
            classifier = learner.Learner(model, ckpt_folder)
            classifiers.append(classifier)

            # Determine the input size to be sampled from the dataset
            input_size = config.training_params.output_size + model.fov - 1

            # Create the sampler
            dset_sampler = em.EMDatasetSampler(dataset, input_size, batch_size=config.training_params.batch_size,
                                               label_output_type=model.architecture.output_mode)
            hooks = [
                learner.LossHook(10, model),
                learner.ModelSaverHook(1000, ckpt_folder),
                learner.ValidationHook(500, dset_sampler, model, self.data_folder),
            ]

            classifier.train(config.training_params, dset_sampler, hooks)

        # Train the ensembler
        self.ensemble_method.train(classifiers)

    def predict(self, dataset, images, split='test'):

        predictions = []

        # Make all the predictions
        for config in self.model_configs:
            # Read model from respective checkpoint folder
            ckpt_folder = self.results_folder + config['id'] + '/'

            # Reset the graph, for now....
            tf.reset_default_graph()

            # Restore the classifier
            model = config.model(config.architecture)
            classifier = learner.Learner(model, ckpt_folder)
            classifier.restore()

            print("Predicting for this model")
            pred = classifier.predict(images)

            print("Saving this model's predictions")
            # Save their prediction for posterity
            dataset.prepare_predictions_for_submission(ckpt_folder, split, pred[0],
                                                       label_type=config.architecture.output_mode)

            predictions.append(pred)

        # Return the ensembled predictions
        return self.ensemble_method.predict(predictions)
