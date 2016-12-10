# -*- coding: utf-8 -*-
# https://papers.nips.cc/paper/4741-deep-neural-networks-segment-neuronal-membranes-in-electron-microscopy-images.pdf
from __future__ import print_function
from __future__ import division

import h5py
import tensorflow as tf
import numpy as np
import subprocess
import tifffile

import os
import sys
import IPython as ipy

import models
from models.tf_unet import unet
from models.tf_unet.unet import create_conv_net
from dataprovider.data_provider import VolumeDataProvider
from augmentation import set_path_to_config



# number of layers in unet maps to input_patch - output_patch
fov_dict = {1:4,2:16,3:40,4:88,5:184,6:376}


class DPu():
    def __init__(self, config, input_patch,fov):
        self.config = config
        self.input_patch = input_patch
        self.output_patch = input_patch - fov
        self.fov = fov

        split = 'train'
        dataset_prefix = config.folder + split
        set_path_to_config(dataset_prefix)

        spec = dataset_prefix + '.spec'
        net_spec = {
            'label': (1, input_patch, input_patch),
            'input': (1, input_patch, input_patch)
        }

        params = {
            'augment': [],
            'drange': [0]
        }

        self.dp = VolumeDataProvider(spec, net_spec, params)


    def sample_train(self, batch_size):
        
        output_patch = input_patch - self.fov
        fov = self.fov

        input_batch = np.zeros([batch_size, input_patch, input_patch,1])
        label_batch = np.zeros([batch_size, output_patch, output_patch, 2])
        
        for i in range(batch_size):
            sample = self.dp.random_sample()
            inpt, label = sample['input'], sample['label']
            inpt = inpt.reshape(1,input_patch,input_patch,1)

            label = label[0:2,0,self.fov//2:self.fov//2+self.output_patch,self.fov//2:self.fov//2+self.output_patch]
            reshapedLabel = np.zeros(shape=(1, self.output_patch, self.output_patch, 2))

            reshapedLabel[0,:,:,0] = label[0]
            reshapedLabel[0,:,:,1] = label[1]

            input_batch[i,:,:,:] = inpt
            label_batch[i,:,:,:] = reshapedLabel

        return input_batch,label_batch

    def sample_validation(self):
        
        validation_input_file = h5py.File(config.folder + config.validation_input_h5, 'r')
        validation_input = validation_input_file['main'][:5,:,:].astype(np.float32) / 255.0
        validation_input_file.close()
        c, size, _ = validation_input.shape
        rs_validation_input = validation_input.reshape(c,size,size,1)

        validation_label_file = h5py.File(config.folder + 'validation-affinities.h5','r')
        validation_labels = validation_label_file['main']
        reshaped_labels = np.einsum('dzyx->zyxd', validation_labels[0:2])


        return rs_validation_input,reshaped_labels


def unet_train(config, n_iterations=10000, validation=True, input_patch = 195, layers = 4):
    model = unet.Unet(nx = None, ny = None, channels = 1, layers = layers, n_class = 2, add_regularizers = True, class_weights = None)

    ckpt_folder = config.folder + model.model_name + '/'
    # nx, ny fixed input/output sizes
    # look in tf_unet.unet for documentation

    data_provider = DPu(config, input_patch, fov_dict[layers])
    trainer = unet.Trainer(model, batch_size=1, optimizer="momentum")
    trainer.train(data_provider, ckpt_folder, training_iters=1, epochs=1, dropout=0.2, display_step=10, restore=False)

    return scores



if __name__ == '__main__':
    from cli import config_dict
    dataset = 'isbi'
    config = config_dict(dataset)

    layers = 3
    input_patch = 196

    model = unet.Unet(nx = None, ny = None, channels = 1, layers = layers, n_class = 2, add_regularizers = True, class_weights = None)
    ckpt_folder = config.folder + model.model_name + '/'
    data_provider = DPu(config, input_patch, fov_dict[layers])
    trainer = unet.Trainer(model, batch_size=5, optimizer="adam", learning_rate = .001)

    trainer.train(data_provider, ckpt_folder, training_iters=10, epochs=100, dropout=1, display_step=10, restore=False)
    ipy.embed()




































