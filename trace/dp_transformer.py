import configparser as cp
import os
import dataprovider.data_provider as dp
import numpy as np
import h5py

BOUND = 'boundaries'
XY_AFF = 'xy-aff'
XYZ_AFF = 'xyz-aff'


class DPTransformer:
    def __init__(self, data_folder, spec_fn, target_type=XY_AFF):
        if target_type == BOUND:
            self.target_dim = 1
        elif target_type == XY_AFF:
            self.target_dim = 2
        elif target_type == XYZ_AFF:
            self.target_dim = 3
        else:
            raise Exception('Invalid target_type provided: must be %s, %s, or %s.' % (BOUND, XY_AFF, XYZ_AFF))

        self.spec_path = data_folder + spec_fn
        self.data_folder = data_folder

        config = cp.RawConfigParser()
        config.read(self.spec_path)

        # Make sure our specfile is appropriately configured
        spec_fn_base = os.path.splitext(spec_fn)[0]
        config.set('files', 'img', self.data_folder + spec_fn_base + '-input.h5')
        config.set('files', 'lbl', self.data_folder + spec_fn_base + '-labels.h5')

        # Writing our modified configuration file
        with open(self.spec_path, 'wb') as f:
            config.write(f)

    def batch_iterator(self, fov, output_patch, input_patch):

        net_spec = {
            'label': (1, input_patch, input_patch),
            'input': (1, input_patch, input_patch)
        }

        params = {
            # 'border': 'mirror',
            'augment': [
                {'type': 'warp'},
                {'type': 'grey', 'mode': '2D'},
                {'type': 'flip'}
            ],
            'drange': [0]
        }

        print('Loading from: ' + self.spec_path)
        provider = dp.VolumeDataProvider(self.spec_path, net_spec, params)

        while True:
            sample = provider.random_sample()
            inpt, label = sample['input'], sample['label']

            # Number of slices, patch length, patch width, values per pixel
            inpt = inpt.reshape(1, input_patch, input_patch, 1)

            # Extract the labels, which will have x, y, and z values for each pixel
            label = label[0:self.target_dim, 0, fov // 2:fov // 2 + output_patch, fov // 2:fov // 2 + output_patch]

            # Reshape the label for use in our model
            reshaped_label = np.zeros(shape=(1, output_patch, output_patch, self.target_dim))

            # Central output patch, only the values that we care about
            for i in range(self.target_dim):
                reshaped_label[0, :, :, i] = label[i]

            yield inpt, reshaped_label

    def dataset_from_h5py(self, data_prefix):
        assert(data_prefix == 'train' or data_prefix == 'validation' or data_prefix == 'test')

        input_fn = self.data_folder + data_prefix + '-input.h5'
        with h5py.File(input_fn, 'r') as input_file:
            inpt = input_file['main'][:].astype(np.float32) / 255.0
            reshaped_input = inpt.reshape(inpt.shape + (1,))

        reshaped_label_affinities = None

        # TODO(beisner): instead of loading the affinities here, we should load the dataset and then affinitize

        if data_prefix != 'test':
            label_fn = self.data_folder + data_prefix + '-affinities.h5'
            print('opening at ' + label_fn)
            with h5py.File(label_fn, 'r') as label_affinities_file:
                labels = label_affinities_file['main']
                reshaped_label_affinities = np.einsum('dzyx->zyxd', labels[0:2])

        return reshaped_input, reshaped_label_affinities










