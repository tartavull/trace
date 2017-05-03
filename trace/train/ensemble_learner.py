
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
        self.ensemble_method.train_on_images(classifiers)

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
        return self.ensemble_method.predict(None, predictions)
