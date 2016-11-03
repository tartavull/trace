from __future__ import print_function
from __future__ import division
from ConfigParser import RawConfigParser
import os.path

import h5py
import numpy as np

import snemi3d
from dataprovider.data_provider import VolumeDataProvider

def set_path_to_config(dataset):
    config = RawConfigParser()
    configpath = snemi3d.folder()+dataset+'.spec'
    config.read(configpath)
    config.set('files','img',snemi3d.folder()+dataset+'-input.h5')
    config.set('files','lbl',snemi3d.folder()+dataset+'-labels.h5')
    # Writing our modified configuration file
    with open(configpath, 'wb') as f:
        config.write(f)


def maybe_create_affinities(dataset):
    """
    Args:
        dataset (str): either train or test

        It should recompute them in the case that the labels changes
    """
    if os.path.exists(snemi3d.folder()+dataset+'-affinities.h5'):
        print ('Affinties already exists.')
        return

    if not os.path.exists(snemi3d.folder()+dataset+'-labels.h5'):
        print ('No labels from where to compute affinities.')
        return
    #there is a little bug in DataProvider that doesn't let us do it
    #for the actual size of the dataset 100,1024,1024
    patch_size = (99,1023,1023)
    net_spec = {'label':patch_size}
    params = {'augment': [] , 'drange':[0]}
    set_path_to_config(dataset)
    spec = snemi3d.folder()+dataset+'.spec'
    dp = VolumeDataProvider(spec, net_spec, params)
    affinities =  dp.random_sample()['label']
    with h5py.File(snemi3d.folder()+dataset+'-affinities.h5','w') as f:
        f.create_dataset('main',data=affinities)


def batch_iterator(fov, output_patch, input_patch):
    dataset = 'train'
    net_spec = {'label':(1,input_patch,input_patch),'input':(1,input_patch,input_patch)}
    params = {'augment': [] , 'drange':[0]}
    set_path_to_config(dataset)
    spec = snemi3d.folder()+dataset+'.spec'
    dp = VolumeDataProvider(spec, net_spec, params)

    while True:
        sample = dp.random_sample()
        inpt, label = sample['input'], sample['label']
        inpt = inpt.reshape(1,input_patch,input_patch,1)
        label = label[0:2,0,fov//2:fov//2+output_patch,fov//2:fov//2+output_patch]
        label = label.reshape(1, output_patch,output_patch,2) #central output patch, only x,y affinities
        yield inpt, label
        
if __name__ == '__main__':
    batch_iterator(1).next()