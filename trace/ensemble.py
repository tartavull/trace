from models import conv_net
import tensorflow as tf
import learner
import numpy as np
import utils

N4_3_TEST = [
    {
        'id': 'n4_1',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 500,
    },
    {
        'id': 'n4_2',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 500,
    },
    {
        'id': 'n4_3',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 500,
    }
]

N4_3 = [
    {
        'id': 'n4_1',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 30000,
    },
    {
        'id': 'n4_2',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 30000,
    },
    {
        'id': 'n4_3',
        'model': conv_net.ConvNet,
        'params': conv_net.DEFAULT_PARAMS,
        'epochs': 30000,
    }
]

ENSEMBLE_PARAMS_DICT = {
    'n4_3': N4_3,
    'n4_3_test': N4_3_TEST,
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
    def __init__(self, model_configs, ensemble_method_constructor, data_folder, run_name):
        self.model_configs = model_configs
        self.data_folder = data_folder

        self.results_folder = data_folder + 'results/ensemble/' + ensemble_method_constructor.name + '/run-' + run_name + '/'
        self.ensembler_folder = self.results_folder + 'ensembler/'

        self.ensemble_method = ensemble_method_constructor(self.ensembler_folder)

    def train(self, data_provider):

        # Keep track of every ongoing session
        classifiers = []

        # Train every model
        for config in self.model_configs:

            # Store each model copy in its own folder
            ckpt_folder = self.results_folder + config['id'] + '/'

            # Extract parameters
            model_type = config['model']
            params = config['params']
            epochs = config['epochs']

            # Reset the default graph so that we emit the same variables
            # DON'T DO THIS IN PREDICTION TIME, AS IT WILL DESTROY ALL OTHER GRAPHS
            tf.reset_default_graph()

            # Create the classifier and persist it
            model = model_type(params)
            classifier = learner.Learner(model, ckpt_folder)
            classifiers.append(classifier)

            hooks = [
                learner.LossHook(10),
                learner.ModelSaverHook(1000, ckpt_folder),
                learner.ValidationHook(500, data_provider, model, self.data_folder),
            ]

            classifier.train(data_provider, hooks, epochs)

        # Train the ensembler
        self.ensemble_method.train(classifiers)

    def predict(self, images, split='test'):

        predictions = []

        # Make all the predictions
        for config in self.model_configs:

            # Read model from respective checkpoint folder
            ckpt_folder = self.results_folder + config['id'] + '/'

            # Extract parameters
            model_type = config['model']
            params = config['params']

            # Restore the classifier
            model = model_type(params)
            classifier = learner.Learner(model, ckpt_folder)
            classifier.restore()

            print("Predicting for this model")
            pred = classifier.predict(images)

            print("Saving this model's predictions")
            # Save their prediction for posterity
            utils.generate_files_from_predictions(ckpt_folder, split, pred)

            predictions.append(pred)

        # Return the ensembled predictions
        return self.ensemble_method.predict(predictions)





